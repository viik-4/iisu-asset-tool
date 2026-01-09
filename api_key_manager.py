"""Encrypted local storage for API keys."""
import json
import os
import base64
from pathlib import Path


def _get_embedded_tgdb_key() -> str:
    """Retrieve embedded public API key for TheGamesDB."""
    # Key is split and encoded to prevent simple string searching
    # This is a public API key provided for this application
    _p = [0x65, 0x32, 0x36, 0x38, 0x30, 0x36, 0x36, 0x37]
    _q = [0x30, 0x36, 0x39, 0x37, 0x61, 0x31, 0x65, 0x39]
    _r = [0x64, 0x61, 0x39, 0x36, 0x39, 0x39, 0x36, 0x31]
    _s = [0x62, 0x63, 0x66, 0x62, 0x62, 0x39, 0x61, 0x33]
    _t = [0x64, 0x61, 0x34, 0x35, 0x33, 0x38, 0x32, 0x65]
    _u = [0x65, 0x38, 0x30, 0x32, 0x61, 0x62, 0x31, 0x38]
    _v = [0x62, 0x65, 0x66, 0x37, 0x33, 0x33, 0x35, 0x66]
    _w = [0x35, 0x62, 0x39, 0x61, 0x31, 0x34, 0x39, 0x64]
    return ''.join(chr(c) for c in _p + _q + _r + _s + _t + _u + _v + _w)


class APIKeyManager:
    """Manages API keys with local encrypted storage.

    Uses a simple obfuscation method that doesn't require external dependencies.
    For a production app with high security needs, consider using the `cryptography` package.
    """

    def __init__(self):
        self.config_dir = Path.home() / ".iisu_asset_tool"
        self.config_dir.mkdir(exist_ok=True)
        self.keys_file = self.config_dir / "keys.dat"

        # Get machine-specific key for obfuscation
        self._key = self._get_machine_key()

    def _get_machine_key(self) -> bytes:
        """Get a machine-specific key for obfuscation."""
        import platform
        import uuid

        # Combine machine attributes
        machine_info = f"{platform.node()}-{uuid.getnode()}-iisu-asset-tool"
        # Create a simple hash
        key_bytes = machine_info.encode('utf-8')
        # Pad/trim to 32 bytes
        while len(key_bytes) < 32:
            key_bytes = key_bytes + key_bytes
        return key_bytes[:32]

    def _xor_encrypt(self, data: str) -> str:
        """Simple XOR encryption for obfuscation."""
        data_bytes = data.encode('utf-8')
        encrypted = bytes([b ^ self._key[i % len(self._key)] for i, b in enumerate(data_bytes)])
        return base64.b64encode(encrypted).decode('utf-8')

    def _xor_decrypt(self, encrypted: str) -> str:
        """Simple XOR decryption."""
        try:
            encrypted_bytes = base64.b64decode(encrypted.encode('utf-8'))
            decrypted = bytes([b ^ self._key[i % len(self._key)] for i, b in enumerate(encrypted_bytes)])
            return decrypted.decode('utf-8')
        except Exception:
            return ""

    def save_keys(self, keys_dict: dict):
        """Save API keys (encrypted)."""
        # Encrypt each value
        encrypted_dict = {k: self._xor_encrypt(v) if v else "" for k, v in keys_dict.items()}
        json_data = json.dumps(encrypted_dict)
        self.keys_file.write_text(json_data, encoding='utf-8')

    def load_keys(self) -> dict:
        """Load and decrypt API keys."""
        if not self.keys_file.exists():
            return {}

        try:
            json_data = self.keys_file.read_text(encoding='utf-8')
            encrypted_dict = json.loads(json_data)
            # Decrypt each value
            return {k: self._xor_decrypt(v) if v else "" for k, v in encrypted_dict.items()}
        except Exception:
            return {}

    def get_key(self, service: str) -> str:
        """Get a specific API key.

        First checks environment variables, then falls back to stored keys,
        then falls back to embedded keys for supported services.
        If found, also sets the environment variable.
        """
        env_mapping = {
            "steamgriddb": "SGDB_API_KEY",
            "igdb_client_id": "IGDB_CLIENT_ID",
            "igdb_client_secret": "IGDB_CLIENT_SECRET",
            "thegamesdb": "TGDB_API_KEY"
        }

        env_key = env_mapping.get(service)

        # Environment variables take precedence
        if env_key and os.environ.get(env_key):
            return os.environ.get(env_key, "")

        # Fall back to stored keys
        keys = self.load_keys()
        stored_key = keys.get(service, "")

        # If we found a stored key, also set the environment variable
        if stored_key and env_key:
            os.environ[env_key] = stored_key
            return stored_key

        # Fall back to embedded keys for supported services
        if service == "thegamesdb":
            embedded_key = _get_embedded_tgdb_key()
            if embedded_key and env_key:
                os.environ[env_key] = embedded_key
            return embedded_key

        return stored_key

    def set_key(self, service: str, key: str):
        """Set a specific API key."""
        # TheGamesDB uses a built-in key exclusively - don't allow overrides
        if service == "thegamesdb":
            return

        keys = self.load_keys()
        keys[service] = key
        self.save_keys(keys)

        # Also set environment variable for current session
        env_mapping = {
            "steamgriddb": "SGDB_API_KEY",
            "igdb_client_id": "IGDB_CLIENT_ID",
            "igdb_client_secret": "IGDB_CLIENT_SECRET",
            "thegamesdb": "TGDB_API_KEY"
        }

        env_key = env_mapping.get(service)
        if env_key:
            if key:
                os.environ[env_key] = key
            elif env_key in os.environ:
                del os.environ[env_key]


# Global instance for easy access
_manager = None

def get_manager() -> APIKeyManager:
    """Get the global API key manager instance."""
    global _manager
    if _manager is None:
        _manager = APIKeyManager()
    return _manager
