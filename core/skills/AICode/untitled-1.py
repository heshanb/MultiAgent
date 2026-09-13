file_path = r'D:\Python_project\MultiAgent\core\agents\routes.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到 _rag_judge_llm_lock 行
insert_idx = -1
for i, line in enumerate(lines):
    if '_rag_judge_llm_lock = threading.Lock()' in line:
        insert_idx = i + 1
        break

if insert_idx == -1:
    print('ERROR: Could not find insertion point')
    exit(1)

# 插入缺失的缓存变量
new_vars = '''
# 意图分类缓存
_intent_cache = {}
_intent_cache_lock = threading.Lock()

# 查询改写缓存
_query_rewrite_cache = {}
_query_rewrite_cache_lock = threading.Lock()
'''

lines.insert(insert_idx, new_vars)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f'OK: Added missing cache variables after line {insert_idx}')
print('Done! 重启服务后，所有"未解析的引用"报错将消失。')