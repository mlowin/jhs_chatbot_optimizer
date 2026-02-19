from werkzeug.utils import secure_filename
from tasks.jhs_task import JHSTask
from flask import render_template, request
import os
import pandas as pd
from config import Config
from callback import StartEvent
import json
from state_store import StateStore
from event_queue import EventQueue
from callback import StartLLM, CallbackEvent, NextProcess
import math

class CreatePersona(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    def pre(self):
        prompt = self._state_store.get('integration_persona_prompt')
        persona_number = str(self._state_store.get('integration_persona_number'))

        llm_task = StartLLM("create_persona",prompt, ["Generiere "+persona_number+" verschiedene Personas."],callback='create_persona:CreatePersona:post')
        llm_task_queue = llm_task.get_event_queue_element()
        event_queue = EventQueue()
        event_queue.push(llm_task_queue)
        self._state_store.set('integration_current_id',llm_task_queue['uid'])
        return "event"
      

    
    def post(self, trigger_uid):   
        print("post ist triggered, alles korrekt!")   
        with open('llm_outputs/output_'+trigger_uid+'.json', 'r') as f:
            persona_list = json.loads(f.read())
        self._state_store.set('integration_persona_list',persona_list)
        self._state_store.set('integration_current_process',"edit_persona")
        self._state_store.set('integration_current_status','initial')
        return True