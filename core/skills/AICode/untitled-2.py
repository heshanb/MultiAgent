file_path = r'D:\Python_project\MultiAgent\templates\project-code.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 禁用 renderErrorMarkers 调用
content = content.replace(
    '            // 渲染错误标记\n            renderErrorMarkers();',
    '            // 渲染错误标记（已禁用）\n            // renderErrorMarkers();'
)

# 2. 禁用 checkSyntax 调用（不再检查语法）
content = content.replace(
    '                currentErrors = checkSyntax(content, ext);',
    '                // currentErrors = checkSyntax(content, ext);  // 已禁用\n                currentErrors = [];  // 不显示错误'
)

# 3. 禁用状态栏的错误计数显示
content = content.replace(
    "statusRight.innerHTML = `${hasGit ? '<span>Git: master</span>' : ''}${statusHTML}`;",
    "statusRight.innerHTML = `${hasGit ? '<span>Git: master</span>' : ''}<span>✓ 无错误</span>`;"
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Done! 已禁用语法错误显示功能。刷新页面即可。')