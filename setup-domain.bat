@echo off
chcp 65001 >nul
echo ========================================
echo MultiAgent 域名配置工具
echo ========================================
echo.
echo 本脚本将帮助你配置域名访问
echo.
echo 请选择方案：
echo [1] Cloudflare Tunnel（免费，推荐）
echo [2] Ngrok（免费，快速测试）
echo [3] 查看完整配置指南
echo.
set /p choice=请输入选项 (1/2/3): 

if "%choice%"=="1" goto cloudflare
if "%choice%"=="2" goto ngrok
if "%choice%"=="3" goto guide
echo 无效选项
pause
exit /b

:cloudflare
echo.
echo ========================================
echo 方案A: Cloudflare Tunnel
echo ========================================
echo.
echo 步骤1: 安装 cloudflared
echo.
echo 请选择安装方式：
echo [1] 使用 winget 安装（推荐）
echo [2] 手动下载安装
echo.
set /p cf_choice=请输入选项 (1/2): 

if "%cf_choice%"=="1" goto cf_winget
if "%cf_choice%"=="2" goto cf_manual
echo 无效选项
pause
exit /b

:cf_winget
echo.
echo 正在使用 winget 安装 cloudflared...
winget install --id Cloudflare.cloudflared
if %errorlevel% equ 0 (
    echo ✓ cloudflared 安装成功
) else (
    echo ✗ 安装失败，请手动下载
    echo 下载地址: https://github.com/cloudflare/cloudflared/releases/latest
)
goto cf_setup

:cf_manual
echo.
echo 请手动下载 cloudflared:
echo https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.msi
echo.
echo 下载后双击安装
pause
goto cf_setup

:cf_setup
echo.
echo ========================================
echo 步骤2: 配置 Cloudflare Tunnel
echo ========================================
echo.
echo 请先完成以下准备：
echo 1. 注册 Cloudflare 账号: https://dash.cloudflare.com
echo 2. 购买域名并添加到 Cloudflare
echo 3. 修改域名 DNS 服务器为 Cloudflare 提供的地址
echo.
pause

echo.
echo 正在登录 Cloudflare...
cloudflared tunnel login
if %errorlevel% neq 0 (
    echo ✗ 登录失败
    pause
    exit /b 1
)

echo.
echo 正在创建 Tunnel...
cloudflared tunnel create multiagent
if %errorlevel% neq 0 (
    echo ✗ 创建 Tunnel 失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 步骤3: 配置路由
echo ========================================
echo.
set /p domain=请输入你的域名 (例如: yourdomain.com): 

echo 正在配置 DNS 路由...
cloudflared tunnel route dns multiagent %domain%
if %errorlevel% neq 0 (
    echo ✗ 配置路由失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 步骤4: 启动 Tunnel
echo ========================================
echo.
echo 正在启动 Tunnel，将 http://localhost:5001 暴露到 https://%domain%
echo.
echo 按 Ctrl+C 停止 Tunnel
echo.
pause

cloudflared tunnel run --url http://localhost:5001 multiagent
pause
exit /b

:ngrok
echo.
echo ========================================
echo 方案B: Ngrok
echo ========================================
echo.
echo 步骤1: 下载 Ngrok
echo 下载地址: https://ngrok.com/download
echo.
echo 步骤2: 注册账号并获取 Authtoken
echo 注册地址: https://dashboard.ngrok.com/signup
echo.
echo 步骤3: 配置 Authtoken
echo ngrok config add-authtoken YOUR_TOKEN
echo.
echo 步骤4: 启动 Ngrok
echo ngrok http 5001
echo.
echo 启动后会显示一个 HTTPS 链接，例如:
echo https://abc123.ngrok-free.app
echo.
pause
exit /b

:guide
echo.
echo 正在打开配置指南...
start "" "%~dp0域名配置指南.md"
pause
exit /b