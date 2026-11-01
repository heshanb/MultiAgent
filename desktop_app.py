"""
小智 AI 助手 - 桌面应用启动器
使用 pywebview 将 Web 应用封装为桌面窗口
"""
import os
import sys
import threading
import time
import uvicorn
import webview

def start_server():
    """启动 FastAPI 服务器"""
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=5001,
        log_level="info"
    )

def main():
    """主函数"""
    print(" 启动小智 AI 助手...")
    
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    time.sleep(2)
    
    window = webview.create_window(
        title='小智 AI 助手',
        url='http://127.0.0.1:5001',
        width=1200,
        height=800,
        min_size=(800, 600),
        resizable=True,
        fullscreen=False
    )
    
    webview.start()

if __name__ == '__main__':
    main()