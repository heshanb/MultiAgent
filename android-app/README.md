# 智能体 Android WebView 应用构建指南

##  项目概述

这是一个基于 WebView 的 Android 应用，将您的 MultiAgent 智能体系统打包成安卓应用。

## 📋 前置要求

### 1. 安装 Android Studio
- 下载地址：https://developer.android.com/studio
- 推荐版本：Android Studio Hedgehog 或更高版本

### 2. 安装必要组件
打开 Android Studio 后，确保安装以下组件：
- Android SDK Platform 34 (API Level 34)
- Android SDK Build-Tools
- Android Emulator（可选，用于测试）

## 🚀 构建步骤

### 方法一：使用 Android Studio（推荐）

#### 步骤 1：打开项目
1. 启动 Android Studio
2. 选择 "Open an Existing Project"
3. 选择目录：`D:\Python_project\MultiAgent\android-app`

#### 步骤 2：同步 Gradle
- Android Studio 会自动同步 Gradle
- 如果未自动同步，点击 "File" → "Sync Project with Gradle Files"

#### 步骤 3：修改服务器地址
打开 `app/src/main/java/com/multiagent/webview/MainActivity.java`
找到第 34 行：
```java
private static final String SERVER_URL = "http://127.0.0.1:5001";
```
修改为您的服务器地址，例如：
```java
private static final String SERVER_URL = "https://your-domain.com";
```

#### 步骤 4：构建 APK
1. 点击菜单 "Build" → "Build Bundle(s) / APK(s)" → "Build APK(s)"
2. 等待构建完成
3. 构建成功后，会弹出通知，点击 "locate" 找到 APK 文件
4. APK 位置：`android-app/app/build/outputs/apk/debug/app-debug.apk`

#### 步骤 5：安装到手机
**方式 A：通过 USB 连接**
1. 手机开启开发者选项和 USB 调试
2. 连接手机到电脑
3. 在 Android Studio 中点击 "Run" 按钮

**方式 B：直接安装 APK**
1. 将 APK 文件传输到手机
2. 在手机上允许"安装未知来源应用"
3. 点击 APK 文件进行安装

### 方法二：使用命令行构建

#### 步骤 1：安装 Gradle
```bash
# Windows PowerShell
# 下载 Gradle: https://gradle.org/install/
# 或使用 Chocolatey
choco install gradle
```

#### 步骤 2：构建 APK
```powershell
cd D:\Python_project\MultiAgent\android-app

# 清理并构建
gradlew.bat clean assembleDebug

# 或者构建 Release 版本（需要签名）
gradlew.bat clean assembleRelease
```

#### 步骤 3：查找 APK
```
Debug APK: android-app\app\build\outputs\apk\debug\app-debug.apk
Release APK: android-app\app\build\outputs\apk\release\app-release.apk
```

## 🔧 配置说明

### 1. 服务器地址配置
```java
// MainActivity.java 第 34 行
private static final String SERVER_URL = "http://127.0.0.1:5001";
```

**生产环境建议：**
- 使用 HTTPS 协议
- 配置域名和 SSL 证书
- 部署到云服务器（阿里云、腾讯云等）

### 2. 权限说明
应用请求以下权限：
- **INTERNET**: 访问网络（必需）
- **CAMERA**: 拍照和扫描二维码
- **STORAGE**: 文件上传和下载
- **LOCATION**: 地理位置服务
- **RECORD_AUDIO**: 语音输入

### 3. 应用图标
替换应用图标：
1. 准备图标文件（推荐 512x512 PNG）
2. 在 Android Studio 中右键 `res` → "New" → "Image Asset"
3. 选择您的图标文件
4. 自动生成各种尺寸的图标

##  后端部署

### 本地测试
```bash
# 在 MultiAgent 项目根目录运行
python main.py
```
应用访问：`http://127.0.0.1:5001`

### 云服务器部署（推荐）

#### 使用 Docker 部署
```bash
# 1. 创建 Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["python", "main.py"]
EOF

# 2. 构建镜像
docker build -t multiagent .

# 3. 运行容器
docker run -d -p 5001:5001 --name multiagent multiagent
```

#### 使用 Nginx 反向代理（HTTPS）
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:5001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 支持
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## 📦 发布到应用商店

### 1. 生成签名密钥
```bash
# 在 Android Studio 中
# Build → Generate Signed Bundle / APK
# 选择 APK → 创建新密钥
```

### 2. 配置签名
在 `app/build.gradle` 中添加：
```gradle
android {
    signingConfigs {
        release {
            storeFile file('your-keystore.jks')
            storePassword 'your-store-password'
            keyAlias 'your-key-alias'
            keyPassword 'your-key-password'
        }
    }
    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled true
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }
}
```

### 3. 构建 Release APK
```bash
gradlew assembleRelease
```

### 4. 上架应用商店
- **Google Play**: https://play.google.com/console
- **华为应用市场**: https://developer.huawei.com
- **小米应用商店**: https://dev.mi.com
- **应用宝**: https://open.qq.com

## 🔍 调试技巧

### 1. Chrome 远程调试
1. 手机连接电脑，开启 USB 调试
2. 在 Chrome 浏览器地址栏输入：`chrome://inspect`
3. 找到您的设备，点击 "inspect"

### 2. 查看日志
```bash
# 使用 adb 查看日志
adb logcat | grep -i multiagent

# 或使用 Android Studio 的 Logcat 窗口
```

### 3. 常见问题

**问题 1：白屏或无法加载**
- 检查服务器是否运行
- 检查网络连接
- 检查 AndroidManifest.xml 中的 INTERNET 权限

**问题 2：文件上传失败**
- 检查存储权限是否授予
- 检查服务器文件大小限制

**问题 3：HTTPS 混合内容错误**
- 确保服务器使用 HTTPS
- 或临时允许混合内容（不推荐生产环境）

## 📝 版本更新

### 更新应用版本
修改 `app/build.gradle`：
```gradle
defaultConfig {
    versionCode 2        // 每次更新递增
    versionName "1.1"    // 用户可见版本号
}
```

### 更新服务器地址
修改 `MainActivity.java` 中的 `SERVER_URL`，重新构建 APK。

##  优化建议

### 1. 添加启动画面
创建 `SplashActivity`，显示应用 Logo 和加载动画

### 2. 离线支持
- 添加 Service Worker 缓存静态资源
- 实现离线提示页面

### 3. 推送通知
集成 Firebase Cloud Messaging (FCM) 实现消息推送

### 4. 生物识别登录
集成指纹或面部识别，提升安全性

## 📞 技术支持

如有问题，请检查：
1. Android Studio 日志
2. 服务器日志
3. Chrome 开发者工具控制台

---

**祝您构建顺利！** 🎉