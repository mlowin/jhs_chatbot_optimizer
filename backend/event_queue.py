from tinydb import TinyDB, Query
from tinydb.operations import delete
import threading
import portalocker
import os

class EventQueue:
    def __init__(self, path="task_db.json"):
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

    def push(self, item):
        """Adds a task to the queue."""
        with self._lock():
            db = self._open()
            table = db.table("events")
            table.insert(item)            
            db.close()

    def get_by_uid(self, uid):
        with self._lock():
            db = self._open()
            table = db.table("events")            
            Task = Query()
            result = table.get(Task.uid == uid)        
            db.close()
        return result

    
    def get_all(self):
        with self._lock():
            db = self._open()
            table = db.table("events")            
            entries = table.all() 
            db.close()
        return entries#{item["key"]: item["value"] for item in entries}
    
    def get_current_item(self):
        current_min = 0
        current_lst = []
        
        with self._lock():
            db = self._open()
            table = db.table("events")            
            all_items = table.all()   
            db.close()

        if not all_items:
            return None
        
        # scan all items regarding priority
        for stack_item in all_items:
            if stack_item['priority'] > current_min:
                current_min = stack_item['priority']
                current_lst = []
            if current_min == stack_item['priority']:
                current_lst.append(stack_item)
        
        # if only one item exists with highest priority, return this one
        if len(current_lst) == 1:
            if 'alternate_weight' in current_lst[0]:
                del current_lst[0]['alternate_weight']
            return current_lst[0]

        lowest_alternate_weight = -1
        lowest_alternate_item = None
        
        for stack_item in current_lst:
                if 'alternate_weight' not in stack_item:
                    stack_item['alternate_weight'] = 0
                if lowest_alternate_weight == -1 or stack_item['alternate_weight'] < lowest_alternate_weight:
                    lowest_alternate_weight = stack_item['alternate_weight']
                    lowest_alternate_item = stack_item
        
        return lowest_alternate_item
    
    def delete(self, doc_id):
        with self._lock():
            db = self._open()
            table = db.table("events")
            table.remove(doc_ids=[doc_id])        
            db.close()

    
    def update(self, item):        
        with self._lock():
            db = self._open()
            table = db.table("events")
            table.update(item, doc_ids=[item.doc_id])   
            db.close()

    def size(self):
        with self._lock():
            db = self._open()
            table = db.table("events")
            size = len(table)
            db.close()
        return size
    
