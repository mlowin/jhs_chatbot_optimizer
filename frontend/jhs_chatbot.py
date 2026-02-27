#!/usr/bin/env python
# coding: utf-8
import os
import time 
from ml_chatbot import PlainChatbot, CustomCallbackHandler
import pandas as pd
from datetime import datetime, timedelta, timezone, date
from dateutil.relativedelta import relativedelta
import json
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import threading 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import xml.etree.ElementTree as ET
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community import embeddings 
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnableLambda
import numpy as np
from langchain.globals import set_debug
from torch.utils.data.dataset import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import json


set_debug(True)

class JHSChatbot(PlainChatbot):
    
    def __init__(self, llm_retrieval, llm_generation, llm_template, llm_system_prompt, rag_embedding_model, rag_chroma_dir, model_path, model_columns, title_column, rag_k = 3):
        self.llm_retrieval = llm_retrieval
        self.llm_generation = llm_generation
        self.rag_embedding_model = rag_embedding_model
        self.rag_chroma_dir = rag_chroma_dir
        self.rag_k = rag_k
        self.title_column = title_column
        self.retriever = self.get_vectorstore().as_retriever(
            search_type = "similarity",
            search_kwargs = {"k":self.rag_k*3}
        )
        self.model_columns = model_columns


        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)


        PlainChatbot.__init__(self, llm_retrieval, llm_template, llm_system_prompt)
        
   
    def get_vectorstore(self):
        if os.path.isdir(self.rag_chroma_dir):
            vectorstore = Chroma(persist_directory=self.rag_chroma_dir, embedding_function=self.rag_embedding_model)
            print("load chroma from dir")
        
           
        return vectorstore


    def format_docs(self, docs):
        print("FORMAT")
        txt = "\n\n"
        for doc in docs:
            #print(doc.metadata['Titel'])
            for col in doc.metadata:
                txt += col+": "+str(doc.metadata[col])+"\n"
            #txt += "Artikelnummer: {}\nArtikelbezeichnung: {}\nArtikelbeschreibung: {}\n\n".format(doc.metadata['Artikelnummer'],doc.metadata['Titel'],doc.metadata['Beschreibung'])
            
        return txt

    def rag_score(self, items, set_rankers):
        scores = []
        for item in items:
            score = 0
            for cat in set_rankers:
                if cat in item.metadata:
                    score += 1
            scores.append(score)
        return scores
        
    def search_and_format(self, c):
        # get user

        filter_list = []

        inputs = self.tokenizer(c['question'], padding=True, truncation=True, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        probabilities = torch.sigmoid(logits)  # Apply sigmoid to convert logits to probabilities
        i = 0
        set_rankers = {}
        for col in self.model_columns:
            print(col, probabilities[0][i])
            if probabilities[0][i] > 0.5:
                set_rankers[col] = 1
            i += 1


        

        res = self.retriever.get_relevant_documents(
            c['question'],
            #filter={"$and":filter_list} if len(filter_list) > 0 else {}
        )
        scores = np.asarray(self.rag_score(res, set_rankers))
        
        log_filters = []
        for cat in set_rankers:
            log_filters.append(cat)

        print("META")
        print(res)
        print(res[0].metadata)
        print("RES")
        print([doc.metadata[self.title_column] for doc in res])
        print("SCORES", scores)
        res_best = []
        for index in scores.argsort()[-self.rag_k:][::-1]:
            res_best.append(res[index])

        res = res_best
        print("final",[doc.metadata[self.title_column] for doc in res])
        with open("filters.json","w") as f:
            f.write(json.dumps({"$and":filter_list} if len(filter_list) > 0 else {}))
        # print("Die veränderte Frage lautet: ",c['question'])
        # res = self.retriever.invoke(c['question'],search_kwargs={
        #     "k": self.rag_k,
        #     "filter": {"Kategorie": {"$in": ["Kartenspiele"]}}
        # })
        log_numbers = [doc.metadata[self.title_column] for doc in res]
        log_time = datetime.now().replace(microsecond=0).isoformat()

        log_entry = {
            'time': log_time,
            'history': c['history'],
            'question': c['question'],
            'filters': log_filters,
            'articles': log_numbers
        }
        with open('chat_log.json', 'a') as f:
            f.write('\n'+'#'*50+"\n"+json.dumps(log_entry))

        formated = self.format_docs(res)
        
        return formated
   

    def stream_chat(self, history, url=None, stream=True):
        
        if url is None:
            url = 'https://www.wiim.uni-frankfurt.de'
        self.url = url
        formatted = self.format_chat_history(history)
    
        # Vorbereitung des Inputs
        input_dict = {"history": formatted}
    
        # Hole entweder die Frage direkt oder den Chain-Output
        if formatted.count('<|start_header_id|>user') > 1:
            question = self.contextualize_q_chain .invoke(input_dict)
        else:
            txt = formatted.split('|start_header_id|>user<|end_header_id|>\n')[-1]
            question = txt.split('<|eot_id|>')[0]
    
        # Füttere mit der richtigen Frage weiter
        input_dict["question"] = question
        input_dict["context"] = self.search_and_format(input_dict)
        if not stream:
            result = self.chain.invoke(input_dict)
            return result
        else:
            print("STREAM")
            # mit Streaming-Callback
            handler = QueueStreamingHandler(context=input_dict["context"])
            thread = threading.Thread(target=lambda: self.chain.invoke(
                input_dict,
                config={"callbacks": [handler]}
            ))
            thread.start()
    
            def token_generator():
                yield "data: [start]\f\f"
                while True:
                    token = handler.queue.get()
                    print("TOKEN",token)
                    if token is None:
                        break
            
                    if token.startswith("[EXTRA_DATA]"):
                        json_data = token[len("[EXTRA_DATA]"):]
                        yield f"event: extra\fdata: {json_data}\f\f"
                    else:
                        yield f"data: {token}\f\f"
            return token_generator()
        
    def contextualized_question(self, input: dict):   
        if input['history'].count('<|start_header_id|>user') > 1:
            print("context q chain")
            return self.contextualize_q_chain
        else:
            txt = input['history']
            txt = txt.split('|start_header_id|>user<|end_header_id|>\n')[-1]
            txt = txt.split('<|eot_id|>')[0]
            print("OUTPUT",txt)
            return txt
        
    def create_chain(self):
        contextualize_q_system_prompt = """Repeat the last human request only and do not answer the request. Add additional details such as what the main problem is, so one can understand it without any context. Just reformulate the question but do not answer it. Here is the chat history: """
        
           
        template_q = self.llm_template.format(system_prompt=contextualize_q_system_prompt, history='{history} Reformulate this message without answering it!')      
        contextualize_q_prompt = PromptTemplate.from_template(template=template_q)
        self.contextualize_q_chain = contextualize_q_prompt | self.llm_retrieval | StrOutputParser()
        return (
            RunnablePassthrough.assign(
                question = self.contextualized_question 
             
            )
            # |
            # RunnablePassthrough.assign(
            #     context = self.search_and_format      
            # )
            | self.llm_system_prompt 
            | RunnableParallel(
                passed=RunnablePassthrough(),
                output=self.llm_generation#lambda x: x["num"] + 1,
            )
        )
    
        
    
from langchain.callbacks.base import BaseCallbackHandler
from queue import Queue
import json

class QueueStreamingHandler(BaseCallbackHandler):
    def __init__(self, context=None, type=None):
        self.queue = Queue()
        self.buffer = ""  # Hier sammeln wir den kompletten Text
        self.context = context
        self.type = type

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.buffer += token  # Token anhängen
        self.queue.put(token)  # Weiterstreamen

    def on_llm_end(self, *args, **kwargs) -> None:
        # Jetzt haben wir den vollständigen Text:
        full_text = self.buffer
        print("onllmend",full_text)
        
        # Hier kannst du add_links aufrufen:
        #linked_output = self.add_links(full_text)
        linked_output = full_text
        if full_text != "":
            # Dann als separates JSON-Event in die Queue legen:
            self.queue.put("[EXTRA_DATA]" + json.dumps({
                "linked_html": linked_output,
                "type": self.type
            }))
        
            
            # Ende signalisieren
            self.queue.put(None)