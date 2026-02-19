from flask import Flask, render_template, request, send_from_directory, Response, stream_with_context, send_file
import json
import os
import random
from jhs_chatbot import JHSChatbot
import requests

from langchain_community.llms import VLLMOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.globals import set_debug


set_debug(True)

from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'

from langchain.callbacks.base import BaseCallbackHandler
from queue import Queue


class QueueStreamingHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = Queue()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put(token)

    def on_llm_end(self, *args, **kwargs) -> None:
        self.queue.put(None)  # Signal für "Ende"

llm_retrieval = VLLMOpenAI(
    model_name="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    base_url="http://10.10.0.20:8000/v1",
    api_key="token-ml-llama3.1-2024",
    verbose=True,
    temperature=0,
    frequency_penalty=0.5,
    top_p=0.95,
    max_tokens=4096,
    streaming=False
)

llm_generation = VLLMOpenAI(    
    model_name="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    base_url="http://10.10.0.20:8000/v1",
    api_key="token-ml-llama3.1-2024",
    verbose=True,
    temperature=0,
    frequency_penalty=0.5,
    top_p=0.95,
    max_tokens=4096,
    streaming=True
)

template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
{history}
<|start_header_id|>assistant<|end_header_id|><|end_of_text|>
"""

rag_system_prompt = """Du bist ein Assistent zum Finden geeigneter Startups aus dem KI-Bereich.
In deiner Datenbank befinden sich rund 1000 Startups und Beschreibungen dazu. Schlage Startups vor, die zur Kundenanfrage passen.
Gerne mehrere Startups, falls diese passen. Verlinke immer die Websites der Startups mit Markdown und schreibe, was die Startups machen.
Nutze zur Beantwortung der Fragen ggf. die folgende Kontextinformation, die Teil einer RAG-Suche ist:
{context}"""

embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-m3',model_kwargs={'device': 'cpu'})
api_url = 'http://127.0.0.1:1338'
persist_directory = './chroma'
import sys, os
sys.path.append(os.path.abspath("../backend"))

from state_store import StateStore

state_store = StateStore(path="../backend/state_store.json")
model_path = "../backend/"+state_store.get("integration_finetuned_model")
model_columns = state_store.get("integration_finetuned_variables")[1:]
columns = state_store.get('integration_column_definition')
title_column = [c["name"] for c in columns if c['column'] == 'title'][0]

chatbot = JHSChatbot(llm_retrieval, llm_generation, template, rag_system_prompt, embedding, persist_directory, model_path, model_columns, title_column)


from flask import Flask, render_template, request, send_from_directory, Response, stream_with_context, jsonify, request
import json
from flask_cors import CORS, cross_origin
import datetime

app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'



@app.route("/")
def home():
    with open('../backend/system_config.json', 'r') as f:
        system_config = json.loads(f.read())
    return render_template("template.htm", logo_path = "WIIM_Logo.png", title=system_config['title'], accent_color=system_config['accent_color'])



@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@cross_origin()
@app.route("/response", methods=["POST"])
def stream_response():    
    userText = request.form.get('history')
    userObj = json.loads(userText)

    if userObj[-1]['role'] == 'user':
        return Response(
            stream_with_context(chatbot.stream_chat(userObj)),
            mimetype='text/event-stream'
        )
    else:
        return json.dumps({'role':'error','message':'Last role was not from user'})
    

if __name__ == "__main__":
  app.run(host='0.0.0.0', port=2339)