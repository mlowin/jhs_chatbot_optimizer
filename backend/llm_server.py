from openai import OpenAI
import pandas as pd
import numpy as np
import json
import random
import time
import requests

client = OpenAI(
 base_url="http://10.10.0.20:7999/v1",
 api_key="token-ml-gpt-oss-2025",
# base_url="http://10.10.0.20:8000/v1",
# api_key="token-ml-llama3.1-2024",
)

def invoke_llm(system, user,temperature=0):
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        # model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=temperature
    )
    output = completion.choices[0].message.content

    return output

def get_current_item(llm_stack):
    current_min = 0
    current_lst = []
    for uid in llm_stack:
        stack_item = llm_stack[uid]
        if stack_item['priority'] > current_min:
            current_min = stack_item['priority']
            current_lst = []
        if current_min == stack_item['priority']:
            current_lst.append(uid)

    current_min = -1
    current_min_uid = None

    for uid in current_lst:
        stack_item = llm_stack[uid]
        if current_min == -1 or stack_item['current_item'] < current_min:
            current_min = stack_item['current_item']
            current_min_uid = uid
    
    return current_min_uid


input_cache = dict()
output_cache = dict()

while True:
    with open('llm_stack.json', 'r') as f:
        llm_stack = json.loads(f.read())
    if len(llm_stack) > 0:
        uid = get_current_item(llm_stack)
        item = llm_stack[uid]
        if uid not in input_cache:
            with open('llm_inputs/'+uid+".json", 'r') as f:
                input_cache[uid] = json.loads(f.read())
                output_cache[uid] = []
        result = invoke_llm(item['prompt'], input_cache[uid].pop(0))
        output_cache[uid].append(result)
        with open('llm_outputs/'+uid+".json", 'w') as f:
            f.write(json.dumps(output_cache[uid]))
        item['current_item'] += 1

        if item['current_item'] == item['item_size']:
            r = requests.get(item['endpoint'])
            del input_cache[uid]
            del output_cache[uid]
            del llm_stack[uid]
        else:
            llm_stack[uid] = item

        with open('llm_stack.json', 'w') as f:
            f.write(json.dumps(llm_stack[uid]))        
    else:
        time.sleep(5)
        
