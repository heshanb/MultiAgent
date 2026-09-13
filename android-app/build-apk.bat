@echo off
chcp 65001 >nul
echo ========================================
echo   智能体 Android 应用快速构建脚本
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] 清理旧构建文件...
if exist "app\build" (
    rmdir /s /q "app\build"
    echo 清理完成
) else (
    echo 无需清理
)
echo.

echo [2/3] 开始构建 Debug APK...
echo 这可能需要几分钟，请耐心等待...
echo.

if exist "gradlew.bat" (
    call gradlew.bat assembleDebug
) else (
    echo 错误：未找到 gradlew.bat
    echo 请使用 Android Studio 打开项目进行构建
    pause
    exit /b 1
)

echo.
echo [3/3] 构建完成！
echo.

if exist "app\build\outputs\apk\debug\app-debug.apk" (
    echo APK 文件位置：
    echo %CD%\app\build\outputs\apk\debug\app-debug.apk
    echo.
    echo 是否打开文件夹？(Y/N)
    set /p open_folder=
    if /i "%open_folder%"=="Y" (
        explorer "app\build\outputs\apk\debug"
    )
) else (
    echo 错误：未找到 APK 文件
    echo 请检查构建日志
)

echo.
echo ========================================
echo 安装说明：
echo 1. 将 APK 文件传输到手机
echo 2. 在手机上允许"安装未知来源应用"
echo 3. 点击 APK 文件进行安装
echo ========================================
echo.
pause