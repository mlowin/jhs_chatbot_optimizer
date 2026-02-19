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

class ExtractColumns(JHSTask):

    def pre(self):
        state_store = StateStore()
        filepath = state_store.get("integration_filename")
        df = self.read_file(filepath)

        len_df = len(df)
        if len_df > 0:
            dict_df = df.iloc[:2].to_dict(orient='records')
            with open('prompts/read_dataset.txt','r') as f:
                prompt = f.read()
            llm_task = StartLLM("extract_columns",prompt, [json.dumps(dict_df)],callback='extract_columns:ExtractColumns:post')
            llm_task_queue = llm_task.get_event_queue_element()
            event_queue = EventQueue()
            event_queue.push(llm_task_queue)
            state_store.set('integration_current_id',llm_task_queue['uid'])
            return "event"
        else:
            state_store.set('select_file_message',"File must not be empty")
            state_store.set('integration_current_process','select_file')
            state_store.set('integration_current_status','show')
            return "show"
            
        
    
    
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
        state_store = StateStore()
        
        with open('llm_outputs/output_'+trigger_uid+'.json', 'r') as f:
            response = f.read()            
            response = json.loads(response)[0]
        try:
            response = json.loads(response)
            if type(response) == list:
                response = response[0]
            print("set column defintion",response)
            state_store.set('integration_column_definition',response)
        except:
            print("could not parse")            
            state_store.set('integration_column_definition',None)
        
        state_store.set('integration_current_process','edit_columns')
        state_store.set('integration_current_status','initial')   