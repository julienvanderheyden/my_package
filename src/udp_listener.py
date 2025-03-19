#! /usr/bin/env python3
import rospy
import socket

# Receiver settings
HOST = '172.29.130.141'  # Listen on all available interfaces
PORT = 25342       # UDP port to listen on

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((HOST, PORT))

print(f"Listening on {HOST}:{PORT}...")

# Receive UDP packet
while True:
    data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
    print(f"Received message: {data.decode()} from {addr}")
