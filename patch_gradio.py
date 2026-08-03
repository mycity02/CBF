path = r'D:\SpecializedSoftware\Anaconda\envs\adv-bmt\lib\site-packages\gradio_client\utils.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
import re
content = re.sub(r'if "(\w+)" in schema:', r'if isinstance(schema, dict) and "\1" in schema:', content)
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Patched successfully')
