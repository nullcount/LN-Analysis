#!/bin/bash

sudo apt update
sudo apt install wireguard

echo Paste Node Private Key: 
read private_key

echo Paste Proxy Public Key:
read public_key

echo Paste Proxy IP:
read server_ip

echo Saving configuration in /etc/wiregaurd/wg0.conf

tee -a /etc/wireguard/wg0.conf << END
[Interface]
PrivateKey = $private_key
Address = 192.168.4.2

[Peer]
PublicKey = $public_key
AllowedIPs = 192.168.4.1/32
Endpoint = $server_ip:55107
PersistentKeepalive = 30

END

echo Starting interface...
sudo systemctl start wg-quick@wg0

echo Enabling autoconnect...
sudo systemctl enable wg-quick@wg0

echo Pinging proxy over interface 192.168.4.1
ping -w 5 192.168.4.1 

sudo wg show

