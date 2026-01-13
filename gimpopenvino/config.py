#!/usr/bin/env python3
# Copyright(C) 2022-2025 Intel Corporation
# SPDX - License - Identifier: Apache - 2.0

"""
Centralized configuration module for GIMP-OpenVINO.

This module provides a single source of truth for all configuration values
used throughout the GIMP-OpenVINO plugin suite. Configuration values can be
overridden via environment variables to support multi-instance scenarios and
custom deployments.

Environment Variables:
    GIMP_OV_SERVER_PORT: Main Stable Diffusion server port (default: 65432)
    GIMP_OV_HANDSHAKE_PORT: Server handshake port (default: 65433)
    GIMP_OV_MODEL_MGMT_PORT: Model management server port (default: 65434)
    GIMP_OV_HOST: Server host address (default: 127.0.0.1)
    GIMP_OV_SOCKET_TIMEOUT: Socket connection timeout in seconds (default: 0.1)
    GIMP_OV_SOCKET_BUFFER_SIZE: Socket receive buffer size in bytes (default: 1024)
    GIMP_OV_SOCKET_BUFFER_SIZE_LARGE: Large buffer size for bulk data (default: 4096)
    GIMP_OV_SERVER_BIND_RETRIES: Number of times to retry binding server socket (default: 15)
    GIMP_OV_SERVER_RETRY_DELAY: Delay in seconds between bind retries (default: 5)
"""

import os

# ==============================================================================
# Network Configuration
# ==============================================================================

# Host Configuration
DEFAULT_HOST = os.getenv("GIMP_OV_HOST", "127.0.0.1")
"""
Default host address for socket communication.
This is the standard loopback interface (localhost).
Override with GIMP_OV_HOST environment variable.
"""

# Port Configuration
SERVER_PORT = int(os.getenv("GIMP_OV_SERVER_PORT", "65432"))
"""
Main Stable Diffusion server port.
Used for primary inference requests and model communication.
Override with GIMP_OV_SERVER_PORT environment variable to avoid port conflicts.
"""

HANDSHAKE_PORT = int(os.getenv("GIMP_OV_HANDSHAKE_PORT", "65433"))
"""
Server handshake port for initialization signaling.
Used to signal when the server is ready to accept connections.
Override with GIMP_OV_HANDSHAKE_PORT environment variable.
"""

MODEL_MANAGEMENT_PORT = int(os.getenv("GIMP_OV_MODEL_MGMT_PORT", "65434"))
"""
Model management server port.
Used for model installation, updates, and management operations.
Override with GIMP_OV_MODEL_MGMT_PORT environment variable.
"""

# ==============================================================================
# Socket Configuration
# ==============================================================================

SOCKET_TIMEOUT = float(os.getenv("GIMP_OV_SOCKET_TIMEOUT", "0.1"))
"""
Socket connection timeout in seconds (default: 100ms).
Short timeout to quickly detect if server is not running.
Override with GIMP_OV_SOCKET_TIMEOUT environment variable.
"""

SOCKET_BUFFER_SIZE = int(os.getenv("GIMP_OV_SOCKET_BUFFER_SIZE", "1024"))
"""
Standard socket receive buffer size in bytes.
Used for most socket recv() operations.
Override with GIMP_OV_SOCKET_BUFFER_SIZE environment variable.
"""

SOCKET_BUFFER_SIZE_LARGE = int(os.getenv("GIMP_OV_SOCKET_BUFFER_SIZE_LARGE", "4096"))
"""
Large buffer size for receiving bulk data.
Used when larger payloads are expected.
Override with GIMP_OV_SOCKET_BUFFER_SIZE_LARGE environment variable.
"""

# ==============================================================================
# Server Configuration
# ==============================================================================

SERVER_BIND_RETRIES = int(os.getenv("GIMP_OV_SERVER_BIND_RETRIES", "15"))
"""
Number of times to retry binding the server socket.
Helps handle cases where the port is temporarily in use.
Override with GIMP_OV_SERVER_BIND_RETRIES environment variable.
"""

SERVER_RETRY_DELAY = int(os.getenv("GIMP_OV_SERVER_RETRY_DELAY", "5"))
"""
Delay in seconds between server bind retry attempts.
Allows time for the port to be released.
Override with GIMP_OV_SERVER_RETRY_DELAY environment variable.
"""

# ==============================================================================
# Legacy Aliases (for backward compatibility)
# ==============================================================================

HOST = DEFAULT_HOST
"""Legacy alias for DEFAULT_HOST. Use DEFAULT_HOST in new code."""

PORT = SERVER_PORT
"""Legacy alias for SERVER_PORT. Use SERVER_PORT in new code."""
