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
from tasks.extract_filters import ExtractFilters
        
class ApplyFiltersOnProducts(JHSTask):

    def __init__(self):
        self._state_store = StateStore()
        
    
    def get_all_articles(self, df):        
        column_definition = self._state_store.get("integration_column_definition")
        lst_txt_columns = [col["name"] for col in column_definition if col['column'] in ['title','description']]
        return df[lst_txt_columns].astype(str).apply(lambda r: " - ".join(r),axis=1).tolist()
    
    def pre(self):        
        filters = self._state_store.get("integration_content_filters")
        ef = ExtractFilters()
        
        filepath = self._state_store.get("integration_filename")
        df = ef.read_file(filepath)
        lst_articles = self.get_all_articles(df)
        
        with open('prompts/apply_filters.txt','r') as f:
            prompt = f.read()
            prompt = prompt.replace("%CONTENT_FILTERS%",json.dumps(filters))


        llm_task = StartLLM("apply_filters_on_products",prompt, lst_articles,callback='apply_filters_on_products:ApplyFiltersOnProducts:post')
        llm_task_queue = llm_task.get_event_queue_element()
        event_queue = EventQueue()
        event_queue.push(llm_task_queue)
        self._state_store.set('integration_current_id',llm_task_queue['uid'])
        return "event" 

    
    def post(self, trigger_uid):      
        self._state_store.set('integration_product_labels','llm_outputs/output_'+trigger_uid+'.json')
        self._state_store.set('integration_current_process',"create_vectordatabase")
        self._state_store.set('integration_current_status','initial')
        return True