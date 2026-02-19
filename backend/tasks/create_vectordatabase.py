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
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def stringify(dct):
    return "\n".join([key+": "+dct[key] for key in dct])

class CreateVectordatabase(JHSTask):

    def pre(self, trigger_uid=None):
        self._state_store = StateStore()
        llm_task = StartEvent("create_vectordatabase:CreateVectordatabase:post")
        llm_task_queue = llm_task.get_event_queue_element()
        event_queue = EventQueue()
        event_queue.push(llm_task_queue)
        self._state_store.set('integration_current_id',llm_task_queue['trigger_uid'])
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
    
    def post(self, trigger_uid=None):
        state_store = StateStore()
        filepath = state_store.get("integration_filename")
        df = self.read_file(filepath)
        
        documents = [stringify(row) for row in df.astype(str).to_dict(orient="records")]
        Chroma().delete_collection()    
            

         
        persist_directory = '../frontend/chroma'

        import shutil

        try:
            shutil.rmtree(persist_directory)
        except:
            print("No previous Chroma version")
        
        os.mkdir(persist_directory)
        os.chmod(persist_directory, 0o755) 
        embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-m3',model_kwargs={'device': 'cpu'})
        meta = df.to_dict("records")

        
        labels = state_store.get("integration_filters_output")
        content_filters = state_store.get('integration_content_filters')
        with open(labels,'r') as f:
            labels = json.loads(f.read())        
        
        # print(type(labels), len(labels))
        # print(labels)
        lst_out = []
        for i in range(len(labels)):
            #try:
            dict_out = dict()
            label_row = json.loads(labels[i])
            for filter_cat in content_filters:
                for filter_sub in content_filters[filter_cat]:
                    if filter_cat in label_row and filter_sub in label_row[filter_cat]:
                        dict_out[filter_cat+"_"+filter_sub] = 1
                    else:
                        dict_out[filter_cat+"_"+filter_sub] = 0
            meta[i] = meta[i] | dict_out

        print("Documents",documents)
        print("Meta",meta)

        vectorstore = Chroma.from_texts(
            texts=documents,
            metadatas=meta,  
            embedding = embedding,
            persist_directory = persist_directory
        )

        try:
            vectorstore.persist()
        except:
            print("Persist fallback")
            
        
        state_store.set('integration_current_process','select_persona_prompt')
        state_store.set('integration_current_status','initial')          
