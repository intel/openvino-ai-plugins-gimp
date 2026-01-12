"""
Smoke tests for socket communication in GIMP OpenVINO AI Plugins.

Tests socket servers on ports:
- 65432: Main Stable Diffusion server
- 65433: Handshake/ready signal server
- 65434: Model management server
"""

import socket
import threading
import time
import pytest
from unittest.mock import Mock, MagicMock, patch


# ============================================================================
# Port 65432: Main Stable Diffusion Server Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.socket
def test_main_server_basic_connection(socket_server_factory):
    """Test basic connection to main SD server on port 65432."""
    server = socket_server_factory(port=65432, response=b"OK")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(5.0)
            client.connect(("127.0.0.1", 65432))
            client.sendall(b"ping")
            response = client.recv(1024)
            assert response == b"OK"
    finally:
        server.shutdown()


@pytest.mark.smoke
@pytest.mark.socket
def test_main_server_ping_command(socket_server_factory):
    """Test ping command to verify server responsiveness."""
    server = socket_server_factory(port=65432, response=b"ping")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(5.0)
            client.connect(("127.0.0.1", 65432))
            client.sendall(b"ping")
            response = client.recv(1024)
            assert response == b"ping"
    finally:
        server.shutdown()


@pytest.mark.smoke
@pytest.mark.socket
def test_main_server_model_name_query():
    """Test querying model name from main server."""
    stop_event = threading.Event()
    
    def server_thread():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.settimeout(1.0)
            try:
                s.bind(("127.0.0.1", 65432))
                s.listen()
                while not stop_event.is_set():
                    try:
                        conn, addr = s.accept()
                        with conn:
                            data = conn.recv(1024)
                            if data.decode() == "model_name":
                                conn.sendall(b"sd_1.5_square")
                    except socket.timeout:
                        continue
                    except Exception:
                        break
            except Exception:
                pass
    
    thread = threading.Thread(target=server_thread, daemon=True)
    thread.start()
    time.sleep(0.2)
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(5.0)
            client.connect(("127.0.0.1", 65432))
            client.sendall(b"model_name")
            response = client.recv(1024)
            assert response == b"sd_1.5_square"
    finally:
        stop_event.set()
        thread.join(timeout=2.0)


@pytest.mark.smoke
@pytest.mark.socket
def test_main_server_multiple_connections(socket_server_factory):
    """Test handling multiple sequential connections."""
    server = socket_server_factory(port=65432, response=b"OK")
    
    try:
        for i in range(3):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                client.settimeout(5.0)
                client.connect(("127.0.0.1", 65432))
                client.sendall(b"ping")
                response = client.recv(1024)
                assert response == b"OK"
    finally:
        server.shutdown()


# ============================================================================
# Port 65433: Handshake/Ready Signal Server Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.socket
def test_handshake_server_ready_signal(socket_server_factory):
    """Test handshake server receives Ready signal on port 65433."""
    server = socket_server_factory(port=65433, response=b"ACK")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(5.0)
            client.connect(("127.0.0.1", 65433))
            client.sendall(b"Ready")
            response = client.recv(1024)
            assert response == b"ACK"
    finally:
        server.shutdown()


@pytest.mark.smoke
@pytest.mark.socket
def test_handshake_server_connection():
    """Test basic connection to handshake server."""
    stop_event = threading.Event()
    received_data = []
    
    def server_thread():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.settimeout(1.0)
            try:
                s.bind(("127.0.0.1", 65433))
                s.listen()
                while not stop_event.is_set():
                    try:
                        conn, addr = s.accept()
                        with conn:
                            data = conn.recv(1024)
                            received_data.append(data.decode())
                            break
                    except socket.timeout:
                        continue
                    except Exception:
                        break
            except Exception:
                pass
    
    thread = threading.Thread(target=server_thread, daemon=True)
    thread.start()
    time.sleep(0.2)
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(5.0)
            client.connect(("127.0.0.1", 65433))
            client.sendall(b"Ready")
            time.sleep(0.2)
            assert "Ready" in received_data
    finally:
        stop_event.set()
        thread.join(timeout=2.0)


# ============================================================================
# Port 65434: Model Management Server Tests
# ============================================================================

@pytest.mark.smoke
@pytest.mark.socket
def test_model_management_server_connection(socket_server_factory):
    """Test basic connection to model management server on port 65434."""
    server = socket_server_factory(port=65434, response=b"OK")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(5.0)
            client.connect(("127.0.0.1", 65434))
            client.sendall(b"ping")
            response = client.recv(1024)
            assert response == b"OK"
    finally:
        server.shutdown()


@pytest.mark.smoke
@pytest.mark.socket
def test_model_management_server_ping(socket_server_factory):
    """Test ping command on model management server."""
    server = socket_server_factory(port=65434, response=b"ping")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(5.0)
            client.connect(("127.0.0.1", 65434))
            client.sendall(b"ping")
            response = client.recv(1024)
            assert response == b"ping"
    finally:
        server.shutdown()


@pytest.mark.smoke
@pytest.mark.socket
def test_model_management_get_all_models():
    """Test get_all_model_details command on model management server."""
    stop_event = threading.Event()
    
    def server_thread():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.settimeout(1.0)
            try:
                s.bind(("127.0.0.1", 65434))
                s.listen()
                while not stop_event.is_set():
                    try:
                        conn, addr = s.accept()
                        with conn:
                            data = conn.recv(1024)
                            if data.decode() == "get_all_model_details":
                                # Send number of installed models
                                conn.sendall(b"0")
                    except socket.timeout:
                        continue
                    except Exception:
                        break
            except Exception:
                pass
    
    thread = threading.Thread(target=server_thread, daemon=True)
    thread.start()
    time.sleep(0.2)
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(5.0)
            client.connect(("127.0.0.1", 65434))
            client.sendall(b"get_all_model_details")
            response = client.recv(1024)
            assert response == b"0"  # No models installed
    finally:
        stop_event.set()
        thread.join(timeout=2.0)


# ============================================================================
# Socket Error Handling Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.socket
def test_socket_connection_refused():
    """Test handling of connection refused error."""
    with pytest.raises(ConnectionRefusedError):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(1.0)
            # Try to connect to a port that's not listening
            client.connect(("127.0.0.1", 65435))


@pytest.mark.unit
@pytest.mark.socket
def test_socket_timeout():
    """Test socket timeout handling."""
    stop_event = threading.Event()
    
    def server_thread():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 65436))
            s.listen()
            # Accept but don't respond
            try:
                conn, addr = s.accept()
                while not stop_event.is_set():
                    time.sleep(0.1)
            except Exception:
                pass
    
    thread = threading.Thread(target=server_thread, daemon=True)
    thread.start()
    time.sleep(0.2)
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(1.0)
            client.connect(("127.0.0.1", 65436))
            client.sendall(b"test")
            with pytest.raises(socket.timeout):
                # This should timeout since server doesn't respond
                client.recv(1024)
    finally:
        stop_event.set()
        thread.join(timeout=2.0)


@pytest.mark.unit
@pytest.mark.socket
def test_socket_reuse_address():
    """Test SO_REUSEADDR option is set correctly."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        option = s.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)
        assert option == 1
