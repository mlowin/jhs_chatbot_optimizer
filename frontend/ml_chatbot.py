#!/usr/bin/env python
# coding: utf-8

from langchain.globals import set_verbose
import torch
set_verbose(True)
from langchain_community.vectorstores import Chroma 
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_community.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
#set_llm_cache(InMemoryCache())
from langchain_community.llms import VLLMOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
import pandas as pd
import json
import copy
from langchain_core.prompt_values import StringPromptValue
from typing import Dict, List, Any
from langchain_core.callbacks import BaseCallbackHandler
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_community import embeddings 
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import pickle
from langchain_community.document_loaders import DirectoryLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 


class MLChatbot:
    def __init__(self, llm, llm_template, llm_system_prompt): 
        if '{context}' in llm_template:
            im_template = llm_template.format(system_prompt=llm_system_prompt, history='{history}', context='{context}')
        else:
            im_template = llm_template.format(system_prompt=llm_system_prompt, history='{history}')
        self.llm_system_prompt = PromptTemplate.from_template(im_template)        
        self.llm_template = llm_template
        self.llm = llm
        self.chain = self.create_chain()
        
    def create_chain(self):
        pass  
    
    
    def format_chat_history(self, chat_history):
        formatted_history = ""
        for message in chat_history:
            formatted_history += f"<|start_header_id|>{message['role']}<|end_header_id|>\n{message['message']}<|eot_id|>\n"
        return formatted_history    
    
    def extract_user_query(self, history):
        return history.split('<|start_header_id|>user<|end_header_id|>\n')[-1].split('<|eot_id|>')[0]
    
    def chat(self, history):    
        history = history[-6:]
        print(self.format_chat_history(history)  )      
        invoke_result = self.chain.invoke(
            {
                "history": self.format_chat_history(history)    
            }, 
            config=
            {
                'callbacks': [CustomCallbackHandler()]
            }
        )  
        
        return invoke_result

class CustomCallbackHandler(BaseCallbackHandler):

    def log(self, run_id, parent_run_id, content):
        last_elem = {}
        global buffer
        if run_id in buffer:
            print("Run ID found in buffer")
            last_elem = copy.copy(buffer[run_id])
            del buffer[run_id] 
        if type(content) == dict:
            last_elem = last_elem | content    
        elif type(content) == str and content[0] == '{':
            content = json.loads(content)
            last_elem = last_elem | content
        elif type(content) == str:
            last_elem['response'] = content
        elif type(content) == StringPromptValue:   
            if 'prompts' not in last_elem:
                last_elem['prompts'] = []
            last_elem['prompts'].append(content.text)
        else:
            print(type(content))
            last_elem['unknown'] = content
        if parent_run_id is None:
            try:
                if 'history' in last_elem:
                    
                    file = open("chat_log.obj",'rb')
                    chat_log = pickle.load(file)
                    file.close()
                    
                    last_elem['user_query'] = extract_user_query(last_elem['history'])                
                    key = hash(last_elem['history']+'<|start_header_id|>assistant<|end_header_id|>\n'+last_elem['response']+'<|eot_id|>\n')
                    last_key = hash('<|start_header_id|>user<|end_header_id|>'.join(last_elem['history'].split('<|start_header_id|>user<|end_header_id|>')[:-1]))
                    print("KEY",key,"Last KEY",last_key)
                    if last_key in chat_log:
                        lst = chat_log[last_key]
                        lst.append(last_elem)
                        del chat_log[last_key]
                    else:
                        lst = [last_elem]
                    chat_log[key] = lst
                    filehandler = open("chat_log.obj","wb")
                    pickle.dump(chat_log,filehandler)
                    filehandler.close()
                else:
                    print('NO HISTORY!')
            except Exception as e:
                print("EXCEPT",e)
        else:
            if parent_run_id in buffer:
                last_elem = last_elem | buffer[parent_run_id]
            buffer[parent_run_id] = last_elem
            
        def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
            self.log(kwargs['run_id'], kwargs['parent_run_id'], outputs)#


class PlainChatbot(MLChatbot):

    def __init__(self, llm, llm_template, llm_system_prompt):
        MLChatbot.__init__(self, llm, llm_template, llm_system_prompt)

    def create_chain(self):
        return (
            self.llm_system_prompt 
            | 
            self.llm
        )

class RAGChatbot(MLChatbot):

    def __init__(self, llm_retrieval, llm_generation, llm_template, llm_system_prompt, rag_embedding_model, rag_chroma_dir, rag_documents_dir, rag_k = 3):
        self.rag_embedding_model = rag_embedding_model
        self.rag_chroma_dir = rag_chroma_dir
        self.rag_documents_dir = rag_documents_dir
        self.rag_k = rag_k
        self.llm_retrieval = llm_retrieval
        self.llm_generation = llm_generation
        MLChatbot.__init__(self, llm_generation, llm_template, llm_system_prompt)

    def search_and_format(self, c):
        # get user
        print("Die veränderte Frage lautet: ",c['question'])
        res = self.retriever.invoke(c['question'])
        formated = self.format_docs(res)
        
        return formated
    
    
    
    
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
    
    
    def format_docs(self, docs):
        print("FORMAT")
        txt = "\n\n"
        for doc in docs:
            print(doc.metadata['source'])
            txt += doc.page_content+"\n"
        #print(txt)
        return txt

    def get_vectorstore(self):
        if os.path.isdir(self.rag_chroma_dir):
            vectorstore = Chroma(persist_directory=self.rag_chroma_dir, embedding_function=self.rag_embedding_model)
            print("load chroma from dir")
        else:
            print("create Chroma")
            Chroma().delete_collection()    
            loader = DirectoryLoader(self.rag_documents_dir)
            docs = loader.load()
            print("len",len(docs))
            # Index the data
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 10000,
                chunk_overlap = 200,
                add_start_index = True 
            )
            all_splits = text_splitter.split_documents(docs)

            all_splits = self.add_metadata(all_splits)
            
            # Create Chroma from data
            vectorstore = Chroma.from_documents(
                documents = all_splits,
                embedding = self.rag_embedding_model,
                persist_directory = self.rag_chroma_dir
            )
            
            vectorstore.persist()
        return vectorstore

    def add_metadata(self, all_splits):
        return all_splits
    
    def create_chain(self):        
        self.retriever = self.get_vectorstore().as_retriever(
            search_type = "similarity",
            search_kwargs = {"k":self.rag_k}
        )
    
        
        contextualize_q_system_prompt = """Repeat the last human request only and do not answer the request. Add additional details such as what the main problem is, so one can understand it without any context. Just reformulate the question but do not answer it. Here is the chat history: """
        
           
        template_q = self.llm_template.format(system_prompt=contextualize_q_system_prompt, history='{history} Reformulate this message without answering it!')      
        contextualize_q_prompt = PromptTemplate.from_template(template=template_q)
        self.contextualize_q_chain = contextualize_q_prompt | self.llm_retrieval | StrOutputParser()
        print("ret",self.llm_retrieval)
        rag_chain = self.create_rag_chain()

        return rag_chain

    def create_rag_chain(self):
        print("gen",self.llm_generation)
        return (
            RunnablePassthrough.assign(
                question = self.contextualized_question 
             
            )
            |
            RunnablePassthrough.assign(
                context = self.search_and_format      
            )
            | self.llm_system_prompt 
            | self.llm_generation
        )
        