"""
Zugang zur Mapillary-API: Token aus .env, Sessions mit Wiederholung.

Bisher stand das in 03 und noch einmal in jedem Experiment, das Detections
abruft.
"""

import os
import threading

import requests
from requests.adapters import HTTPAdapter, Retry

_thread_local = threading.local()


def load_token(root):
    """
    MAPILLARY_TOKEN aus der Umgebung, sonst aus <root>/.env.

    Die .env ist gitignored; das Token faellt nie in ein Notebook oder ein
    Ergebnis.
    """
    env_file = root / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    token = os.environ.get("MAPILLARY_TOKEN", "")
    if not token.startswith("MLY|"):
        raise RuntimeError(
            "Kein Mapillary-Token. Datei .env im Projektwurzelverzeichnis anlegen:\n"
            "  MAPILLARY_TOKEN=MLY|dein|token"
        )
    return token


def make_session(pool_size=16):
    """
    Session mit Wiederholung bei 429 und 5xx. Bei 429 haelt sie sich an
    Retry-After, sonst exponentiell.
    """
    session = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5, status=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=True,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry,
                          pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_session(pool_size=16):
    """Eine Session je Thread -- requests.Session ist nicht threadsicher."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = make_session(pool_size)
    return _thread_local.session
