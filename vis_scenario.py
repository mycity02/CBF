import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('generated_scenarios/sd_10af3d70d93ef629_SCGEN_all_TF_except_adv_mode_0.pkl', 'rb') as f:
    data = pickle.load(f)

tracks = data.get('tracks', {})
sdc_id = data.get('metadata', {}).get('sdc_id', None)

fig, ax = plt.subplots(1, 1, figsize=(12, 12))

for track_name, track in tracks.items():
    states = track.get('state', {})
    positions = states.get('position', None)
    if positions is None:
        continue
    if hasattr(positions, 'shape'):
        x = positions[:, 0]
        y = positions[:, 1]
        valid = ~np.isnan(x)
        x = x[valid]
        y = y[valid]
    else:
        x = [p[0] for p in positions if p is not None and len(p) > 0]
        y = [p[1] for p in positions if p is not None and len(p) > 0]
    if len(x) == 0:
        continue
    color = 'red' if str(track_name) == str(sdc_id) else 'blue'
    label = 'SDC (Ego)' if str(track_name) == str(sdc_id) else None
    ax.plot(x, y, color=color, alpha=0.6, linewidth=1.5, label=label)
    ax.scatter(x[0], y[0], color='green', s=30, zorder=5)
    ax.scatter(x[-1], y[-1], color='red', s=30, zorder=5)

ax.set_title('Adv-BMT Generated Scenario')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('generated_scenario_vis.png', dpi=150, bbox_inches='tight')
print('Saved to generated_scenario_vis.png')
