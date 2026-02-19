import json
import pandas as pd
from tasks.select_file import SelectFile
from flask import request, render_template, redirect
from datetime import datetime, timezone
import string
import random
from state_store import StateStore
from event_queue import EventQueue
import importlib
import callback
from config import Config


state_store = StateStore()
event_queue = EventQueue()

def get_status():
    if not state_store.exists('integration_current_process'):
        return 'select_file','initial'
    
    return state_store.get('integration_current_process'),state_store.get('integration_current_status')



def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def add_llm_task(id, prompt, items, priority=1):
    id += "_"+str((datetime.now(timezone.utc)).timestamp() * 1e3)+"_"+id_generator()
    

    obj = {
        "type":"llm",
        "uid":id,
        "prompt":prompt,
        "current_item": 0,
        "item_size": len(items),
        "priority": priority
    }
    event_queue.push(obj)
    return id

def renderProgress(task_id):
    return render_template("progress_template.htm", task_id = task_id)

def execute_callback(cb_name, args = {}):
    module_name, class_name, method_name = cb_name.split(":")

    # Modul laden (tasks/modulname.py)
    module = importlib.import_module(f"tasks.{module_name}")

    # Klasse holen
    cls = getattr(module, class_name)

    # Instanz erstellen
    instance = cls()

    # Methode holen (pre, do oder post)
    method = getattr(instance, method_name)

    # Methode ausführen
    return method(**args)

def get_current_page():
    current_process, current_status = get_status()
    class_name = Config.PROCESS_FLOW[current_process]
    print("get_current_page",current_process, current_status)
    
    if current_status == 'initial':    
        print("INITIAL")
        result = execute_callback(current_process+":"+class_name+":pre")
        state_store.set('integration_current_status',result)
        print("RESULT",result)
        current_status = result
        print("STATUS",state_store.get('integration_current_status'))

    if current_status == 'show':    
        print("STATUS",state_store.get('integration_current_status'))  
        if request.method == 'GET':      
            result = execute_callback(current_process+":"+class_name+":show")
            
            print("STATUS",state_store.get('integration_current_status'))
            return result
        elif request.method == 'POST':
            result = execute_callback(current_process+":"+class_name+":post")
            if type(result) == callback.NextProcess:
                state_store.set('integration_current_process',result.next_process)
                state_store.set('integration_current_status','initial')
                return redirect("/integrations/")
            else:
                return result

    elif current_status == 'event':
        return renderProgress(state_store.get('integration_current_id'))