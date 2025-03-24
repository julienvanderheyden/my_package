#! /usr/bin/env python3
import socket

# Sender settings
HOST = '127.0.0.1'  # IP address of the receiver (localhost in this case)
PORT = 12345        # UDP port to send data to

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Send message
message = "Hello, Julia! (FROM PYTHON)"
sock.sendto(message.encode(), (HOST, PORT))

print(f"Sent message: {message} to {HOST}:{PORT}")
sock.close()
