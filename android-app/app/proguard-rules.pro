# 添加项目混淆规则
-keepattributes *Annotation*
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}

# 保留 WebView 相关类
-keepclassmembers class * {
    @android.webkit.JavascriptInterface <methods>;
}

# 保留 JavaScript 接口
-keep class com.multiagent.webview.** { *; }