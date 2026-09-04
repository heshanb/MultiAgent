"""
生成自签名SSL证书脚本
用于启用HTTPS，使其他IP的电脑能使用File System Access API
"""
import os
import sys
import ipaddress
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

def generate_cert_cryptography(cert_path, key_path):
    """使用cryptography库生成证书"""
    print("使用cryptography库生成证书...")
    
    # 生成私钥
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    
    # 构建证书主题
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "MultiAgent"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    
    # 构建证书
    cert = x509.CertificateBuilder().subject_name(
        subject
    ).issuer_name(
        issuer
    ).public_key(
        key.public_key()
    ).serial_number(
        x509.random_serial_number()
    ).not_valid_before(
        datetime.now(timezone.utc)
    ).not_valid_after(
        datetime.now(timezone.utc) + timedelta(days=3650)
    ).add_extension(
        x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.DNSName("127.0.0.1"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ]),
        critical=False,
    ).sign(key, hashes.SHA256())
    
    # 写入私钥
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    
    # 写入证书
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    
    print("✓ 使用cryptography成功生成证书")
    return True

def main():
    cert_dir = Path(__file__).parent
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"
    
    if cert_path.exists() and key_path.exists():
        print("证书已存在，如需重新生成请先删除现有证书")
        print(f"  证书: {cert_path}")
        print(f"  私钥: {key_path}")
        return
    
    if not HAS_CRYPTOGRAPHY:
        print("✗ cryptography库未安装")
        print("请运行: pip install cryptography")
        sys.exit(1)
    
    print("开始生成SSL证书...")
    if generate_cert_cryptography(cert_path, key_path):
        print(f"\n✓ 证书生成成功！")
        print(f"  证书: {cert_path}")
        print(f"  私钥: {key_path}")
        print(f"\n现在可以运行 main.py 启用HTTPS")
        print(f"\n注意: 这是自签名证书，浏览器会显示安全警告")
        print(f"      请点击'高级' -> '继续访问'以正常使用")
    else:
        print("\n✗ 证书生成失败")
        sys.exit(1)

if __name__ == "__main__":
    main()