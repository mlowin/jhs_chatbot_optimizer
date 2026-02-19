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

class MergeExtractedFilters(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    def pre(self, trigger):        
        filename = self._state_store.get("integration_filters_output")
        article_type = self._state_store.get("integration_filters_article_type")
        dataset_filters = self._state_store.get("integration_dataset_filters")
        
        with open(filename,'r') as f:
            lst_content_filters = json.loads(f.read())

        with open('prompts/merge_filters.txt','r') as f:
            prompt_merge = f.read()
            prompt_merge = prompt_merge.replace("%%%TYPE%%%", article_type).replace("%%%FILTER_STR%%%",json.dumps(dataset_filters))

        llm_task = StartLLM("merge_extracted_filters_articles",prompt_merge, lst_content_filters,callback='merge_extracted_filters:MergeExtractedFilters:post')
        llm_task_queue = llm_task.get_event_queue_element()
        event_queue = EventQueue()
        event_queue.push(llm_task_queue)
        self._state_store.set('integration_current_id',llm_task_queue['uid'])
        return "event"
      
    
    def post(self, trigger_uid):   
        with open('llm_outputs/output_'+trigger_uid+".json", 'r') as f:
            try:
                content_filters = json.loads(f.read())[0]  
                content_filters = json.loads(content_filters)
            except:
                print("Error parsing JSON")
        self._state_store.set('integration_content_filters',content_filters)        
        self._state_store.set('integration_current_process',"edit_filters")
        self._state_store.set('integration_current_status','initial')