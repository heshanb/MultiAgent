-- MySQL数据库初始化脚本
-- 使用方法：
-- Windows: mysql -u root -p < init_mysql_db.sql
-- 或在MySQL客户端中执行

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS multiagent_db 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_unicode_ci;

-- 创建专用用户（生产环境建议使用专用用户而非root）
-- CREATE USER 'multiagent'@'localhost' IDENTIFIED BY 'your_secure_password';
-- GRANT ALL PRIVILEGES ON multiagent_db.* TO 'multiagent'@'localhost';
-- FLUSH PRIVILEGES;

-- 使用数据库
USE multiagent_db;

-- 验证数据库创建
SELECT 'Database created successfully!' AS status;
SHOW TABLES;