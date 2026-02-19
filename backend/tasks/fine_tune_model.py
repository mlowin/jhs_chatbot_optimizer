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

from event_queue import EventQueue
from bert_trainer import fine_tune

class FineTuneModel(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    

    def pre(self, trigger_uid):
        llm_task = StartEvent("fine_tune_model:FineTuneModel:post")
        llm_task_queue = llm_task.get_event_queue_element()
        event_queue = EventQueue()
        llm_task_queue['current_item'] = 0
        llm_task_queue['item_size'] = 8 
        event_queue.push(llm_task_queue)
        self._state_store.set('integration_current_id',llm_task_queue['trigger_uid'])
        return "event"
      
    def update(self,uid, iteration):
        event_queue = EventQueue()        
        elem = event_queue.get_by_uid(uid)
        if elem:
            elem['current_item'] = iteration
            event_queue.update(elem)

    
    def post(self, trigger_uid):   
        print("start_fine")
        content_filters = self._state_store.get('integration_content_filters')
        labels = self._state_store.get('integration_user_queries_labels')
        with open(labels,'r') as f:
            labels = json.loads(f.read())        
        
        user_queries = self._state_store.get('integration_user_queries')
        # print(type(labels), len(labels))
        # print(labels)
        lst_out = []
        for i in range(len(user_queries)):
            dict_out = {'user_query': user_queries[i]}
            #try:
            label_row = json.loads(labels[i])
            for filter_cat in content_filters:
                for filter_sub in content_filters[filter_cat]:
                    if filter_cat in label_row and filter_sub in label_row[filter_cat]:
                        dict_out[filter_cat+"_"+filter_sub] = 1
                    else:
                        dict_out[filter_cat+"_"+filter_sub] = 0
            lst_out.append(dict_out)
            # except Exception as e:
            #     print(e)
            #     print("error, could not append")
        print("len",len(lst_out))
        df = pd.DataFrame(lst_out)

        output_path = "models/finetuned_BERT_"+trigger_uid
        os.mkdir(output_path)

        fine_tune(df, output_path, self.update, trigger_uid)
        self._state_store.set('integration_finetuned_model',output_path)
        self._state_store.set('integration_finetuned_variables',list(df.columns)) 
        self._state_store.set('integration_current_process',"finished")
        self._state_store.set('integration_current_status',"initial")
        print("fin")