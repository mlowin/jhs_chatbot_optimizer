import portalocker
from tinydb import TinyDB, Query
import os

class StateStore:
    def __init__(self, path="state_store.json"):
        self.path = path
        self.lock_path = path + ".lock"
        self._ensure_lockfile()

    def _ensure_lockfile(self):
        if not os.path.exists(self.lock_path):
            open(self.lock_path, "w").close()

    def _lock(self):
        return portalocker.Lock(
            self.lock_path,
            mode="r+",
            timeout=10,
            flags=portalocker.LOCK_EX
        )

    # Jede Operation öffnet TinyDB neu → kein Shared Cache
    def _open(self):
        return TinyDB(self.path)

    # ----- PUBLIC API -----

    def set(self, key, value):
        with self._lock():
            db = self._open()
            table = db.table("states")
            q = Query()
            table.upsert({"key": key, "value": value}, q.key == key)
            db.close()

    def get(self, key, default=None):
        db = self._open()
        table = db.table("states")
        q = Query()
        entry = table.get(q.key == key)
        db.close()
        return entry["value"] if entry else default
    
    def get_all(self):
        db = self._open()
        table = db.table("states")
        entries = table.all()
        db.close()

        # optional: als dict zurückgeben statt Liste
        return {item["key"]: item["value"] for item in entries}

    def delete(self, key):
        with self._lock():
            db = self._open()
            table = db.table("states")
            q = Query()
            table.remove(q.key == key)
            db.close()

    def exists(self, key):
        db = self._open()
        table = db.table("states")
        q = Query()
        exists = table.contains(q.key == key)
        db.close()
        return exists

    def keys(self):
        db = self._open()
        table = db.table("states")
        ks = [e["key"] for e in table.all()]
        db.close()
        return ks
