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
import random
from tasks.extract_filters import ExtractFilters

class CreateUserQueries(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    def get_rand_articles(self, article_examples, df):        
        column_definition = self._state_store.get("integration_column_definition")
        lst_txt_columns = [col["name"] for col in column_definition if col['column'] in ['title','description']]
        lst_documents = []
        for i in range(math.ceil(article_examples / 10)):    
            mod = 10 if article_examples % 10 == 0 else article_examples % 10
            df_random = df.sample(n=10 if i < math.ceil(article_examples / 10)-1 else mod)
            lst_documents.append(json.dumps(df_random[lst_txt_columns].astype(str).apply(lambda r: " - ".join(r),axis=1).tolist()))
        return lst_documents

    def pre(self):
        prompt = self._state_store.get('integration_user_query_prompt')
        user_query_number = int(self._state_store.get('integration_user_query_number_total'))
        articles_number = int(self._state_store.get('integration_user_query_articles_number'))
        personas = self._state_store.get('integration_persona_list')

        ef = ExtractFilters()
        
        filepath = self._state_store.get("integration_filename")
        df = ef.read_file(filepath)
        instances = []
        for i in range(user_query_number):
            articles = self.get_rand_articles(articles_number, df)
            persona = random.choice(personas)
            instances.append("Persona: "+json.dumps(persona)+"\n\nArticles:\n"+json.dumps(articles))

        llm_task = StartLLM("create_user_queries",prompt, instances,callback='create_user_queries:CreateUserQueries:post')
        llm_task_queue = llm_task.get_event_queue_element()
        event_queue = EventQueue()
        event_queue.push(llm_task_queue)
        self._state_store.set('integration_current_id',llm_task_queue['uid'])
        return "event"
      

    
    def post(self, trigger_uid):    
        with open('llm_outputs/output_'+trigger_uid+'.json', 'r') as f:
            user_queries = json.loads(f.read())
        self._state_store.set('integration_user_queries',user_queries)
        self._state_store.set('integration_current_process',"edit_user_queries")
        self._state_store.set('integration_current_status','initial')
        return True