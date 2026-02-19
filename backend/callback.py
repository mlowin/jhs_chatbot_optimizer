from datetime import datetime, timezone
import string
import random
import json

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

class CallbackEvent:

    def __init__(self, event_name, args = {}, result = None):
        self._event_name = event_name
        self._args = args
        self._result = result
        self._id = self._event_name+"_"+id_generator()

    def get_callback_event(self):
        obj = {
            "type":"event",
            "callback": self._event_name,
            "id": self._id,
            "callback_args":self._args,
            "priority":self._result
        }
        return obj

class NextProcess:

    def __init__(self, process_name):
        self.next_process = process_name

class StartLLM:

    def __init__(self, task, prompt, items, priority=1, callback = None):
        self._task = task
        self._prompt = prompt
        self._items = items
        self._priority = priority
        self._callback = callback
        self.set_id()
        with open('llm_inputs/input_'+self._id+".json", 'w') as f:
            f.write(json.dumps(items))
    
    def set_id(self):
        self._id = self._task+"_"+str((datetime.now(timezone.utc)).timestamp() * 1e3)+"_"+id_generator()


    def get_event_queue_element(self):
        obj = {
            "type":"llm",
            "uid":self._id,
            "prompt":self._prompt,
            "current_item": 0,
            "item_size": len(self._items),
            "priority": self._priority,            
        }
        if self._callback is not None:
            obj['callback'] = self._callback
        return obj

class StartEvent:

    def __init__(self, event_handle, trigger_uid=None, uid=None):
        self._event_handle = event_handle

        if trigger_uid is not None:
            self._trigger_uid = trigger_uid
        else:
            self._trigger_uid = event_handle.replace(":","_")+"_"+str((datetime.now(timezone.utc)).timestamp() * 1e3)+"_"+id_generator()
        
        if uid is not None:
            self._uid = uid
        else:
            self._uid = self._trigger_uid

    def get_event_handle(self):
        return self._event_handle
    
    def get_event_queue_element(self):
        return {
            "type":"event",
            "callback": self._event_handle,
            "priority": 2,
            "trigger_uid": self._trigger_uid,
            "uid": self._uid
        }