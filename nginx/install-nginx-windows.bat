@echo off
chcp 65001 >nul
echo ========================================
echo Nginx Windows安装脚本
echo ========================================
echo.

echo [1/4] 检查是否已安装Nginx...
where nginx >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Nginx已安装
    goto :configure
)

echo ✗ Nginx未安装，开始下载...
echo.

echo [2/4] 下载Nginx...
powershell -Command "& {Invoke-WebRequest -Uri 'https://nginx.org/download/nginx-1.25.4.zip' -OutFile 'nginx.zip'}"

if not exist nginx.zip (
    echo ✗ 下载失败，请手动下载
    echo 下载地址: https://nginx.org/download/nginx-1.25.4.zip
    pause
    exit /b 1
)

echo [3/4] 解压Nginx...
powershell -Command "& {Expand-Archive -Path 'nginx.zip' -DestinationPath '.' -Force}"

echo [4/4] 清理临时文件...
del nginx.zip

:configure
echo.
echo ========================================
echo 配置Nginx...
echo ========================================
echo.

REM 复制配置文件
if exist ..\nginx\nginx.conf (
    copy /Y ..\nginx\nginx.conf nginx\conf\nginx.conf
    echo ✓ 配置文件已复制
) else (
    echo ✗ 未找到配置文件 nginx.conf
)

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 下一步：
echo 1. 编辑 nginx\conf\nginx.conf
echo 2. 将 server_name 改为你的域名
echo 3. 将 SSL证书放到 nginx\cert\ 目录
echo 4. 运行 start-nginx.bat 启动Nginx
echo.
pause