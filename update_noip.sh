#!/bin/bash
# No-IP DDNS自动更新脚本

USERNAME="x4tt9g8"
PASSWORD="jjb85oPTx2ff"
HOSTNAME="cultural-relics.serveftp.com"  # 例如：cultural-relics.ddns.net

# 更新DDNS
curl -s -u "$USERNAME:$PASSWORD" "https://dynupdate.no-ip.com/nic/update?hostname=$HOSTNAME"

echo ""
echo "DDNS updated at $(date)"