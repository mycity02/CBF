"""State-gated physical rear protection; exactly one control filter per step."""
import numpy as np
from dataclasses import replace

from bmt.runtime_assurance.adaptive_plugin import FSMHOCBFPlugin, RearEscapeCBFPlugin
from bmt.runtime_assurance.cbf2d import _projected_half_extents
from bmt.runtime_assurance.metadrive_plugin import MetaDriveCBFPlugin
from bmt.runtime_assurance.rear_escape import RearEscapeCBF


class RearPhysicalGuardPlugin:
    name = 'fsm_rear_physical_guard'

    def __init__(self, physical_config, legacy_config):
        if not physical_config.rear_use_native_dynamics:
            raise ValueError('The physical guard requires a native-dynamics configuration')
        self.config = physical_config
        self.legacy = FSMHOCBFPlugin(legacy_config)
        self.physical = RearEscapeCBFPlugin(physical_config)
        self.observer = MetaDriveCBFPlugin(legacy_config)
        self.assessor = RearEscapeCBF(replace(legacy_config, rear_trigger_ttc=physical_config.rear_trigger_ttc,
            rear_guard_allow_slow_approach=physical_config.rear_guard_allow_slow_approach,
            rear_require_aligned_heading=physical_config.rear_require_aligned_heading,
            fsm_enter_h=physical_config.fsm_enter_h, fsm_exit_h=physical_config.fsm_exit_h,
            fsm_enter_psi1=physical_config.fsm_enter_psi1, fsm_exit_psi1=physical_config.fsm_exit_psi1))
        self.active = False
        self.clear_steps = 0
        self.tracked = set()

    def reset(self, env):
        for component in [self.legacy, self.physical, self.observer]:
            component.reset(env)
        self.assessor.reset()
        self.active = False
        self.clear_steps = 0
        self.tracked.clear()

    def describe(self):
        return dict(name=self.name, config=self.config.to_dict(),
                    legacy=self.legacy.describe(),
                    selection='current rear approach plus bounded physical-road entry and safe nominal handoff; no candidate actions')

    def _ego_vehicle(self, env):
        return self.observer._ego_vehicle(env)

    def _action_to_control(self, action, ego):
        return (self.physical if self.active else self.legacy)._action_to_control(action, ego)

    def filter_action(self, env, nominal_action):
        ego_object = self._ego_vehicle(env)
        objects = self.observer._vehicle_objects(env)
        ego = self.observer._vehicle_state(ego_object)
        others = [self.observer._vehicle_state(obj) for obj in objects if obj.name != ego_object.name]
        road = (self.physical if self.config.rear_use_physical_road_geometry
                else self.observer)._road_geometry(ego_object)
        entering = []
        for other in self.assessor._rear_candidates(ego, others, road):
            diagnostic = self.assessor._vehicle_barrier(ego, other, road).diagnostic
            closing = float((other.velocity-ego.velocity)@road.tangent)
            # The low-speed supervisor already treats a close rear vehicle as
            # a risk. A second closing-speed gate must not silently suppress
            # that decision while the legacy backend starts its own maneuver.
            slow_close_gap = (self.config.rear_guard_allow_slow_approach and ego.speed < 3.
                              and diagnostic.h <= self.config.fsm_enter_h)
            if (closing > .1 or slow_close_gap) and self.assessor._rear_entering(ego, other, road, diagnostic):
                entering.append(other)
        road_h = min((b.diagnostic.h for b in self.physical.filter._road_barriers(ego, road)),
                     default=float('inf'))
        reason = 'hold_physical' if self.active else 'legacy_no_rear_trigger'
        if not self.active and entering:
            if road_h >= -self.config.rear_guard_road_entry_slack-self.config.constraint_tolerance:
                self.physical.reset(env)
                self.active = True
                self.tracked = {other.name for other in entering}
                self.clear_steps = 0
                reason = ('rear_approach_in_physical_road_set' if road_h >= 0.
                          else 'rear_approach_in_bounded_road_recovery_set')
            else:
                reason = 'rear_approach_outside_physical_road_set'
        elif self.active:
            self.tracked.update(other.name for other in entering)
            all_clear = True
            for other in others:
                if other.name not in self.tracked:
                    continue
                relative = other.position-ego.position
                longitudinal, lateral = float(relative@road.tangent), float(relative@road.left_normal)
                ego_long, ego_lat = _projected_half_extents(ego, road.tangent, road.left_normal)
                other_long, other_lat = _projected_half_extents(other, road.tangent, road.left_normal)
                lateral_rate = float((other.velocity-ego.velocity)@road.left_normal)
                closing = float((other.velocity-ego.velocity)@road.tangent)
                separated = abs(lateral) > ego_lat+other_lat+self.config.lateral_margin
                separating = lateral*lateral_rate >= -.1
                passed = longitudinal > ego_long+other_long+1.
                falling_back = longitudinal < -(ego_long+other_long+5.) and closing <= 0.
                all_clear &= passed or (separated and separating) or falling_back
            heading_error = np.arctan2(np.sin(ego.heading-road.heading), np.cos(ego.heading-road.heading))
            stable = abs(heading_error) < .1 and abs(float(ego.velocity@road.left_normal)) < .4
            nominal_legacy, _ = self.legacy._action_to_control(nominal_action, ego_object)
            old_active = self.assessor._current_active_obstacles(ego, others, road)
            old_barriers = self.assessor._build_barriers(ego, old_active, road)
            handoff_safe = all(b.constraint.residual(nominal_legacy.as_array()) >= -self.config.constraint_tolerance
                               for b in old_barriers)
            self.clear_steps = self.clear_steps+1 if all_clear and stable and not entering and handoff_safe else 0
            if self.clear_steps >= max(self.config.fsm_clear_steps, 3):
                self.legacy.reset(env)
                self.active = False
                self.tracked.clear()
                reason = 'rear_clear_and_lateral_motion_stable'
        selected = self.physical if self.active else self.legacy
        # Only the selected controller solves a QP. The observer evaluates
        # current geometry but does not generate, project or compare actions.
        action, telemetry = selected.filter_action(env, nominal_action)
        telemetry['rta_selected_plugin'] = telemetry['rta_plugin']
        telemetry['rta_plugin'] = self.name
        telemetry['rta_physical_guard_active'] = self.active
        telemetry['rta_rear_guard_reason'] = reason
        telemetry['rta_physical_road_entry_h'] = road_h if np.isfinite(road_h) else None
        self.observer._remember_velocities([ego_object, *objects])
        self.observer._episode_steps += 1
        return action, telemetry
