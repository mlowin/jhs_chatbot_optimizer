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

class ExtractFilters(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    def pre(self):        
        article_type = self._state_store.get("integration_filters_article_type")
        article_examples = self._state_store.get("integration_filters_article_examples")
        filepath = self._state_store.get("integration_filename")
        df = self.read_file(filepath)
        
        
        column_definition = self._state_store.get("integration_column_definition")
        print("column definintion",column_definition, type(column_definition))
        dataset_filters = {}
        for col in column_definition:
            if col['column'] == 'filter':
                dataset_filters[col['column']] = list(df[col['name']].value_counts().keys()[:10])
        self._state_store.get("integration_dataset_filters", dataset_filters)     

        with open('prompts/create_filters.txt','r') as f:
            prompt = f.read()
            prompt = prompt.replace("%%%TYPE%%%", article_type).replace("%%%FILTER_STR%%%",json.dumps(dataset_filters))

        lst_txt_columns = [col["name"] for col in column_definition if col['column'] in ['title','description']]
        lst_documents = []
        for i in range(math.ceil(article_examples / 10)):    
            mod = 10 if article_examples % 10 == 0 else article_examples % 10
            df_random = df.sample(n=10 if i < math.ceil(article_examples / 10)-1 else mod)
            lst_documents.append(json.dumps(df_random[lst_txt_columns].astype(str).apply(lambda r: " - ".join(r),axis=1).tolist()))

        llm_task = StartLLM("extract_filters_articles",prompt, lst_documents,callback='extract_filters:ExtractFilters:post')
        llm_task_queue = llm_task.get_event_queue_element()
        event_queue = EventQueue()
        event_queue.push(llm_task_queue)
        self._state_store.set('integration_current_id',llm_task_queue['uid'])
        return "event"
      
    
    def get_extension(self, filename):
        extension = filename.rsplit('.', 1)[1].lower() 
        return extension
    
    def read_file(self, filepath):
        extension = self.get_extension(filepath)
        df = None
        if extension in ['xlsx','xls']:
            df = pd.read_excel(filepath)
        elif extension == 'csv':
            df = pd.read_csv(filepath)
        elif extension == 'json':
            df = pd.read_json(filepath)
        return df
    
    def post(self, trigger_uid):      
        self._state_store.set('integration_filters_output','llm_outputs/output_'+trigger_uid+'.json')
        return NextProcess("merge_extracted_filters")