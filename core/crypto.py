"""
crypto.py — AES-256-CBC payload encryption.

KNOWN LIMITATIONS, deliberately left in place.

These are real weaknesses, but changing them would alter payload sizes and the
key schedule and so invalidate comparison with the published benchmark. They are
documented here and listed as future work rather than silently fixed:

  1. derive_key is a bare SHA-256 of the password: no salt, no iteration count,
     no memory hardness. It is not a password-based key derivation function and
     offers no resistance to offline brute force or precomputation. A real
     deployment wants PBKDF2-HMAC-SHA256, scrypt or Argon2 with a stored salt.
  2. The same password seeds BOTH the AES key and the embedding position map, so
     one compromised secret yields both the plaintext and the location of every
     modified sample. Independent subkeys derived from a single master secret
     (e.g. HKDF with distinct info strings) would separate the two roles.
  3. CBC provides confidentiality but no integrity. The ciphertext is malleable
     and there is no MAC, so tampering is undetectable and the only failure
     signal is an unpadding error — a padding-oracle shape. Encrypt-then-MAC, or
     an AEAD mode such as AES-GCM, would fix this.
"""

import hashlib
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


def derive_key(password: str) -> bytes:
    """
    Derives a 32-byte AES-256 key from the password using SHA-256.

    NOT a password-based KDF — no salt, no iterations. See the module docstring.
    """
    return hashlib.sha256(password.encode('utf-8')).digest()


def derive_seed(password: str) -> int:
    """
    Derives an integer seed for random position generation.
    Uses SHA-256 hash of password converted to integer.
    """
    hash_bytes = hashlib.sha256(password.encode('utf-8')).digest()
    return int.from_bytes(hash_bytes[:8], byteorder='big')


def encrypt_message(message: str, password: str) -> bytes:
    """
    Encrypts a plaintext message string using AES-256 CBC mode.
    Returns: IV (16 bytes) + encrypted ciphertext
    """
    key = derive_key(password)
    iv = os.urandom(16)                          # Random 16-byte IV
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded = pad(message.encode('utf-8'), AES.block_size)
    ciphertext = cipher.encrypt(padded)
    return iv + ciphertext                       # Prepend IV to ciphertext


def decrypt_message(encrypted_data: bytes, password: str) -> str:
    """
    Decrypts AES-256 CBC encrypted data back to plaintext string.
    Expects: IV (16 bytes) + ciphertext
    """
    key = derive_key(password)
    iv = encrypted_data[:16]                     # Extract IV from first 16 bytes
    ciphertext = encrypted_data[16:]             # Rest is ciphertext
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded = cipher.decrypt(ciphertext)
    return unpad(padded, AES.block_size).decode('utf-8')


def get_encrypted_length(message: str) -> int:
    """
    Returns the byte length of encrypted message (for capacity checking).
    AES CBC pads to nearest 16-byte block + 16 bytes for IV.
    """
    msg_len = len(message.encode('utf-8'))
    padded_len = ((msg_len // 16) + 1) * 16     # Padded to block size
    return 16 + padded_len                       # IV + ciphertext