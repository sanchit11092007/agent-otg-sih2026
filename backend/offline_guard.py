import socket

# Save the original socket methods so we don't lose them
_original_connect = socket.socket.connect
_original_create_connection = socket.create_connection

def _is_allowed(address):
    """Check if the given host is localhost/127.0.0.1/::1."""
    if not isinstance(address, tuple) or len(address) < 2:
        return False
    host = address[0]
    return host in ("127.0.0.1", "localhost", "::1") or _external_allowed

def _guarded_connect(self, address):
    if not _is_allowed(address):
        raise RuntimeError(
            "Blocked: this application is not allowed to make network connections. "
            f"Attempted to connect to: {address}"
        )
    return _original_connect(self, address)

def _guarded_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
    if not _is_allowed(address):
        raise RuntimeError(
            "Blocked: this application is not allowed to make network connections. "
            f"Attempted to connect to: {address}"
        )
    return _original_create_connection(address, timeout, source_address)

def enable_offline_mode():
    """
    Monkeypatches Python's built-in socket module to prevent outbound network
    connections, except to localhost (which is needed for Ollama).

    NOTE: This is a software-level safety net (best-effort) to prevent accidental
    data exfiltration by standard Python libraries (like requests). It is NOT a
    substitute for a true firewall or Docker network isolation, as C-level
    extensions or subprocesses could still bypass it.
    """
    socket.socket.connect = _guarded_connect
    socket.create_connection = _guarded_create_connection


from contextlib import contextmanager

_external_allowed = False

@contextmanager
def allow_external():
    """Temporarily lift the offline guard (e.g. to fetch a URL image)."""
    global _external_allowed
    _external_allowed = True
    try:
        yield
    finally:
        _external_allowed = False
