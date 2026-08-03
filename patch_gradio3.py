path = r'D:\SpecializedSoftware\Anaconda\envs\adv-bmt\lib\site-packages\gradio_client\utils.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if line.strip() == 'def get_type(schema):' and i + 1 < len(lines):
        new_lines.append(line)
        indent = '    '
        new_lines.append(indent + 'if not isinstance(schema, dict):\n')
        new_lines.append(indent + '    return "Any"\n')
    else:
        new_lines.append(line)

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Patched successfully')
