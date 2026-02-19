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

class ApplyFiltersOnUserQueries(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    def pre(self):        
        filters = self._state_store.get("integration_content_filters")
        user_queries = self._state_store.get("integration_user_queries")
        
        with open('prompts/apply_filters.txt','r') as f:
            prompt = f.read()
            prompt = prompt.replace("%CONTENT_FILTERS%",json.dumps(filters))


        llm_task = StartLLM("apply_filters_on_user_queries",prompt, user_queries,callback='apply_filters_on_user_queries:ApplyFiltersOnUserQueries:post')
        llm_task_queue = llm_task.get_event_queue_element()
        event_queue = EventQueue()
        event_queue.push(llm_task_queue)
        self._state_store.set('integration_current_id',llm_task_queue['uid'])
        return "event"
      

    
    def post(self, trigger_uid):      
        self._state_store.set('integration_user_queries_labels','llm_outputs/output_'+trigger_uid+'.json')
        return NextProcess("fine_tune_model")