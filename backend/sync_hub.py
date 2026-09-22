"""
sync_hub.py — Hotspot Port Forwarding & Multi-Device Content Hub for Agent OTG
=============================================================================

Enables running the AI server on the main PC while multiple receiver devices
(connected via Wi-Fi hotspot or local network) can retrieve live streams,
answers, code, and generated files simply by entering a short 4-digit ID or
auto-syncing in real time.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import random
import socket
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


def get_local_ip_addresses() -> Dict[str, Any]:
    """
    Detect local network IPv4 addresses (Hotspot, Wi-Fi, Ethernet).
    Returns recommended access URLs for receiver devices.
    """
    candidates: List[str] = []

    # Method 1: Hostname resolution with gethostbyname_ex (fast and reliable on Windows)
    try:
        hostname = socket.gethostname()
        _, _, ip_list = socket.gethostbyname_ex(hostname)
        for ip in ip_list:
            if ip and ip != "127.0.0.1" and not ip.startswith("169.254.") and ip not in candidates:
                candidates.append(ip)
    except Exception:
        pass

    # Method 2: Dummy socket connection (no packet sent)
    if not candidates:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and ip != "127.0.0.1" and not ip.startswith("169.254.") and ip not in candidates:
                candidates.append(ip)
        except Exception:
            pass

    # Fallback to localhost if no LAN interface found
    if not candidates:
        candidates.append("127.0.0.1")

    # Prioritize 192.168.137.x (standard Windows Mobile Hotspot IP) or 192.168.x.x
    def _priority(ip: str) -> int:
        if ip.startswith("192.168.137."):
            return 0  # Windows Hotspot adapter default
        if ip.startswith("192.168."):
            return 1
        if ip.startswith("10."):
            return 2
        if ip.startswith("172."):
            return 3
        if ip == "127.0.0.1":
            return 99
        return 10

    candidates.sort(key=_priority)
    primary_ip = candidates[0]

    return {
        "primary_ip": primary_ip,
        "all_ips": candidates,
        "frontend_port": 5173,
        "backend_port": 8000,
        "recommended_frontend_url": f"http://{primary_ip}:5173",
        "recommended_backend_url": f"http://{primary_ip}:8000",
        "is_hotspot": primary_ip.startswith("192.168.137."),
    }


class SyncItem:
    def __init__(self, sync_id: str, query: str):
        self.sync_id: str = sync_id
        self.query: str = query
        self.status: str = "streaming"  # "streaming" | "completed" | "error"
        self.content: str = ""
        self.stages: List[Dict[str, Any]] = []
        self.artifact: Optional[Dict[str, Any]] = None
        self.sources: List[Any] = []
        self.category: str = "general"
        self.model_used: str = ""
        self.time_seconds: float = 0.0
        self.error: Optional[str] = None
        self.created_at: str = _now_iso()
        self.updated_at: str = _now_iso()
        self.listeners: List[asyncio.Queue] = []
        self._lock = threading.Lock()

    def append_stage(self, stage: Dict[str, Any]):
        with self._lock:
            self.stages.append(stage)
            self.updated_at = _now_iso()
        self._broadcast({"type": "stage", **stage})

    def append_content(self, token: str):
        with self._lock:
            self.content += token
            self.updated_at = _now_iso()
        self._broadcast({"type": "token", "content": token})

    def complete(self, meta: Optional[Dict[str, Any]] = None):
        with self._lock:
            self.status = "completed"
            self.updated_at = _now_iso()
            if meta:
                if "model_used" in meta:
                    self.model_used = meta["model_used"]
                if "time_seconds" in meta:
                    self.time_seconds = meta["time_seconds"]
                if "artifact" in meta:
                    self.artifact = meta["artifact"]
                if "sources" in meta:
                    self.sources = meta["sources"]
                if "category" in meta:
                    self.category = meta["category"]
        self._broadcast({
            "type": "done",
            "status": "completed",
            "model_used": self.model_used,
            "time_seconds": self.time_seconds,
            "artifact": self.artifact,
            "sources": self.sources,
            "category": self.category,
        })

    def fail(self, error_message: str):
        with self._lock:
            self.status = "error"
            self.error = error_message
            self.updated_at = _now_iso()
        self._broadcast({"type": "error", "detail": error_message})

    def _broadcast(self, event: Dict[str, Any]):
        dead = []
        for q in list(self.listeners):
            try:
                q.put_nowait(event)
            except Exception:
                dead.append(q)
        for q in dead:
            if q in self.listeners:
                self.listeners.remove(q)

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "sync_id": self.sync_id,
                "query": self.query,
                "status": self.status,
                "content": self.content,
                "stages": self.stages,
                "artifact": self.artifact,
                "sources": self.sources,
                "category": self.category,
                "model_used": self.model_used,
                "time_seconds": self.time_seconds,
                "error": self.error,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }


class SyncHub:
    """Central registry and broadcast manager for multi-device sync."""

    def __init__(self, max_items: int = 100):
        self._items: Dict[str, SyncItem] = {}
        self._order: List[str] = []
        self._max_items = max_items
        # create_item() generates an ID while holding this lock.  Use an RLock
        # because generate_id() also protects the item registry.
        self._lock = threading.RLock()
        self._global_listeners: List[asyncio.Queue] = []
        self._latest_sync_id: Optional[str] = None

    def generate_id(self) -> str:
        """Generate a user-friendly 4-digit numeric PIN for fast entry on phones/tablets."""
        with self._lock:
            for _ in range(1000):
                pin = f"{random.randint(1000, 9999)}"
                if pin not in self._items:
                    return pin
            return f"{random.randint(10000, 99999)}"

    def create_item(self, query: str, sync_id: Optional[str] = None) -> SyncItem:
        with self._lock:
            if not sync_id:
                sync_id = self.generate_id()
            item = SyncItem(sync_id=sync_id, query=query)
            self._items[sync_id] = item
            self._order.insert(0, sync_id)
            self._latest_sync_id = sync_id

            # Evict oldest items if exceeding limit
            while len(self._order) > self._max_items:
                old_id = self._order.pop()
                self._items.pop(old_id, None)

        self._broadcast_global({
            "type": "new_broadcast",
            "sync_id": sync_id,
            "query": query,
            "created_at": item.created_at,
        })
        return item

    def get_item(self, sync_id: str) -> Optional[SyncItem]:
        with self._lock:
            return self._items.get(sync_id)

    def get_latest_item(self) -> Optional[SyncItem]:
        with self._lock:
            if self._latest_sync_id and self._latest_sync_id in self._items:
                return self._items[self._latest_sync_id]
            if self._order:
                return self._items.get(self._order[0])
            return None

    def get_recent_items(self, limit: int = 15) -> List[Dict[str, Any]]:
        with self._lock:
            recent_ids = self._order[:limit]
            return [self._items[sid].to_dict() for sid in recent_ids if sid in self._items]

    def _broadcast_global(self, event: Dict[str, Any]):
        dead = []
        for q in list(self._global_listeners):
            try:
                q.put_nowait(event)
            except Exception:
                dead.append(q)
        for q in dead:
            if q in self._global_listeners:
                self._global_listeners.remove(q)

    def subscribe_global(self) -> asyncio.Queue:
        q = asyncio.Queue()
        self._global_listeners.append(q)
        return q

    def unsubscribe_global(self, q: asyncio.Queue):
        if q in self._global_listeners:
            self._global_listeners.remove(q)


# Singleton Hub Instance
hub = SyncHub()
