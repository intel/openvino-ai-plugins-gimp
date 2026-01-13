#!/usr/bin/env python3
# Copyright(C) 2022-2023 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0

import socket
import sys
import os

sys.path.extend([os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "..")])
from gimpopenvino import config

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((config.DEFAULT_HOST, config.SERVER_PORT))
    s.sendall(b"Hello, GIMP")
    data = s.recv(config.SOCKET_BUFFER_SIZE)

print(f"Received {data!r}")
