path = r'D:\SpecializedSoftware\Anaconda\envs\adv-bmt\lib\site-packages\gradio_client\utils.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Patch get_type function: add guard for non-dict schema
old_get_type = '''def get_type(schema):
    if "const" in schema:'''
new_get_type = '''def get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    if "const" in schema:'''
content = content.replace(old_get_type, new_get_type)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('Patched successfully')
