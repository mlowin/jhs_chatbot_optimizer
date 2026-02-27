from openai import OpenAI, RateLimitError
import pandas as pd
import numpy as np
import json
import random
import time
import requests
import os
import importlib
from datetime import datetime, timezone
import string
import random
from event_queue import EventQueue
from state_store import StateStore
import callback
from config import Config
import re

with open('llm_config.json') as config_file:
    config = []
    for llm in json.loads(config_file.read()):
        if llm['use_backend']:
            config.append(llm)
    



client = OpenAI(
 base_url=config[0]["base_url"],
 api_key=config[0]["model_token"]
)


def execute_callback(cb_name, trigger_uid):#, args, result):
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
    return method(trigger_uid)#result=result, **args)


def invoke_llm(system, user,temperature=0, num_retries = 0):
    if system is None: 
        system = ''
    try:
        completion = client.chat.completions.create(
            model=config[0]["model_name"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=temperature
        )
        output = completion.choices[0].message.content
        output = re.sub(r"^```(?:json)?\s*|\s*```$", "", output.strip(), flags=re.DOTALL)
        return output
    
    except RateLimitError as e:
        error_message = str(e)
        # Wartezeit aus Fehlermeldung extrahieren
        match = re.search(r"try again in ([\d\.]+)s", error_message)

        if match:
            wait_time = float(match.group(1))
        else:
            wait_time = 5  # Fallback

        print(f"Rate limit erreicht. Warte {wait_time:.2f}s...")
        time.sleep(wait_time)
        return invoke_llm(system, user,temperature, num_retries +1)
    


input_cache = dict()
output_cache = dict()

event_queue = EventQueue()
print("RUN")
while True:
    if event_queue.size() > 0:
        stack_item = event_queue.get_current_item()
        print("working on",stack_item)        

        if stack_item['type'] == 'llm':
            if stack_item['uid'] not in input_cache:
                with open('llm_inputs/input_'+stack_item['uid']+".json", 'r') as f:
                    input_cache[stack_item['uid']] = json.loads(f.read())
                print("load from scratch",stack_item['current_item'])
                if stack_item['current_item'] > 0:
                    print("pop")
                    input_cache[stack_item['uid']] = input_cache[stack_item['uid']][stack_item['current_item']:]
                if os.path.exists('llm_outputs/output_'+stack_item['uid']+".json"):                
                    with open('llm_outputs/output_'+stack_item['uid']+".json", 'r') as f:
                        output_cache[stack_item['uid']] = json.loads(f.read())
                else:
                    output_cache[stack_item['uid']] = []
            
            user_message = input_cache[stack_item['uid']].pop(0)
            print(user_message)
            result = invoke_llm(stack_item['prompt'], user_message)
            print("_____")
            print(result)
            pattern = r"```json\s*(\{.*?\})\s*```"
            match = re.search(pattern, result, re.DOTALL)

            if match:
                result = match.group(1)
            print("?",match)
            
            output_cache[stack_item['uid']].append(result)
            with open('llm_outputs/output_'+stack_item['uid']+".json", 'w') as f:
                f.write(json.dumps(output_cache[stack_item['uid']]))
            stack_item['current_item'] += 1
            
        
            if stack_item['current_item'] == stack_item['item_size']:
                del input_cache[stack_item['uid']]
                del output_cache[stack_item['uid']]
                event_queue.delete(stack_item.doc_id)
            else:
                if 'alternate_weight' in stack_item:
                    stack_item['alternate_weight'] += 1
                event_queue.update(stack_item)

                
            if 'callback' in stack_item and stack_item['current_item'] == stack_item['item_size']:
                start_event = callback.StartEvent(stack_item['callback'],stack_item['uid'])
                event_queue.push(start_event.get_event_queue_element())
        
        
        else: # callback
            state_store = StateStore()
            state_store.set('integration_current_process',stack_item["callback"].split(":")[0])
            state_store.set('integration_current_status','event')

            response = execute_callback(
                stack_item["callback"],
                stack_item['trigger_uid']
            )
            print("set stack_item['trigger_uid']",stack_item)
            state_store.set('integration_current_id',stack_item['trigger_uid'])
            event_queue.delete(stack_item.doc_id)

            if type(response) == callback.NextProcess:
                state_store.set('integration_current_process',response.next_process)
                state_store.set('integration_current_status','event')    
                
                response = execute_callback(
                    response.next_process+":"+Config.PROCESS_FLOW[response.next_process]+":pre",
                    stack_item['trigger_uid']
                )
                
            

    else:
        print("sleep")
        time.sleep(5)
        
