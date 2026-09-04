# Ngrok 快速使用指南

## 第一步：注册Ngrok账号

1. 访问：https://dashboard.ngrok.com/signup
2. 使用邮箱注册（或使用Google/GitHub账号）
3. 验证邮箱

## 第二步：获取Authtoken

1. 登录：https://dashboard.ngrok.com
2. 点击左侧菜单 "Your Authtoken"
3. 复制你的Authtoken（一串长字符）

## 第三步：配置Authtoken

打开PowerShell，运行：

```powershell
cd D:\Python_project\MultiAgent
.\ngrok.exe config add-authtoken 你的TOKEN
```

例如：
```powershell
.\ngrok.exe config add-authtoken 2aBCdefGHIjklMNOpqrSTUvwxYZ1234567890
```

## 第四步：启动MultiAgent服务

确保你的Python服务正在运行：

```powershell
python main.py
```

服务应该在 http://localhost:5001 运行

## 第五步：启动Ngrok

打开新的PowerShell窗口，运行：

```powershell
cd D:\Python_project\MultiAgent
.\ngrok.exe http 5001
```

## 第六步：获取访问链接

Ngrok启动后，会显示类似这样的信息：

```
Forwarding  https://abc123def456.ngrok-free.app -> http://localhost:5001
```

将这个HTTPS链接分享给其他人，他们就可以访问你的服务了！

## 注意事项

### 免费版限制
- 每次重启Ngrok，链接会变化
- 有带宽限制（约40GB/月）
- 有连接数限制
- 首次访问需要点击"Visit Site"按钮

### 保持运行
- Ngrok窗口不能关闭
- 关闭后链接失效

### 安全提示
- 不要将敏感数据暴露到公网
- 演示结束后关闭Ngrok
- 可以在Ngrok Dashboard查看访问日志

## 常见问题

### Q: 链接能固定吗？
A: 免费版每次重启会变化，付费版可以固定域名

### Q: 能自定义域名吗？
A: 需要付费版才支持自定义域名

### Q: 安全吗？
A: 
- 使用HTTPS加密传输
- 但服务本身没有密码保护
- 建议演示期间使用，不要长时间开放

### Q: 如何查看访问统计？
A: 
- 访问 https://dashboard.ngrok.com
- 查看请求日志和统计

## 快速启动脚本

创建一个批处理文件 `start-ngrok.bat`：

```batch
@echo off
cd /d D:\Python_project\MultiAgent
echo 正在启动Ngrok...
ngrok.exe http 5001
pause
```

双击即可启动（需要先配置Authtoken）