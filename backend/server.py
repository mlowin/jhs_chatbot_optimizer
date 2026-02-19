# flask --app server.py --debug run

import time
from flask import Flask, render_template, request, send_from_directory, Response, stream_with_context, send_file
import json
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
from config import Config

app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER

from flask import Flask, render_template, request, send_from_directory, Response, stream_with_context, jsonify, request
import json
from flask_cors import CORS, cross_origin
import datetime
import pandas as pd
import integration_handler
from event_queue import EventQueue
from state_store import StateStore

app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'



@app.route("/")
def page_dashboard():    
    if os.path.exists('llm_config.json'):
        with open('llm_config.json', 'r') as f:
            llm_config = f.read()
        llms = len(json.loads(llm_config))
    else:
        llms = 0

    state_store = StateStore()
    filename = state_store.get('integration_filename')

    if filename is not None:
        df = read_file(filename)
        products = len(df)
    else:
        products = 0

    category_config = state_store.get('integration_content_filters')
    if category_config is not None:
        filters = sum([len(c) for c in category_config])
    else:
        filters = 0

    if os.path.exists('../frontend/chat_log.json'):
        with open('../frontend/chat_log.json', 'r') as f:
                chat_log = f.read().split("\n"+"#"*50+"\n")
        chats = len(chat_log) -1
    else:
        chats = 0

    
    event_queue = EventQueue()

    states = json.dumps({
        "states":state_store.get_all(),
        "events":event_queue.get_all()
    }, indent=4)
    return render_template("home_template.htm", llms=llms, products = products, filters=filters, chats=chats, states=states)

@app.route("/llms/", methods=['GET'])
def page_llms():    
    
    if os.path.exists('llm_config.json'):
        with open('llm_config.json', 'r') as f:
            llm_config = f.read()
    else:
        llm_config = []
    return render_template("llm_template.htm", llm_config=llm_config, llm_stack=json.dumps([]))

@app.route("/llms/", methods=['POST'])
def save_llms():    
    
    with open('llm_config.json', 'w') as f:
        f.write(json.dumps(request.json))
    return "true"


@app.route("/interface_settings/", methods=['GET','POST'])
def interface_settings():        
    with open('system_config.json', 'r') as f:
        system_config = json.loads(f.read())
    
    if 'title' in request.form:
        system_config['title'] = request.form.get('title')
        system_config['accent_color'] = request.form.get('accent_color')
        with open('system_config.json', 'w') as f:
            f.write(json.dumps(system_config))
    return render_template("interface_template.htm", title=system_config['title'], accent_color=system_config['accent_color'])

@app.route("/products/", methods=['GET'])
def page_products():    
    state_store = StateStore()
    
    column_definition = state_store.get("integration_column_definition")
    
    
    filepath = state_store.get("integration_filename")
    if filepath is not None:
        df = read_file(filepath)
    else:
        df = None
    if type(column_definition) == str:
        column_definition = json.loads(column_definition)
    if df is not None and column_definition is not None:
        if len(column_definition) > 0 and type(list(column_definition.keys())[0]) == str:
            lst_col = []
            for col in column_definition:
                lst_col.append({'name':col, 'column':column_definition[col]['column'], 'type':column_definition[col]['type']})
            column_definition = lst_col
        df = df[[col["name"] for col in column_definition if col["column"] != 'no-filter']]
        df = json.dumps(df.to_dict(orient='records'))
    else:
        df = []
        column_definition = []
    
    return render_template("product_template.htm", products=df, file_config=json.dumps(column_definition))

@app.route("/categories/", methods=['GET'])
def page_categories():    
    state_store = StateStore()
    content_filters = state_store.get('integration_content_filters')
    return render_template("category_template.htm", category_config=content_filters, in_process=False)

@app.route("/categories/", methods=['POST'])
def save_categories():        
    with open('content_filters.json', 'w') as f:
        f.write(json.dumps(request.json))
    return "true"

@app.route("/integrations/", methods=['GET', 'POST'])
def page_integrations():    
    return integration_handler.get_current_page()#render_template("integrations_template.htm")


@app.route("/reports/", methods=['GET'])
def page_reports():    
    import re
    if os.path.exists('../frontend/chat_log.json'):
        with open('../frontend/chat_log.json', 'r') as f:
            chat_log = f.read().split("\n"+"#"*50+"\n")
            chats = []
            for chat in chat_log:
                if len(chat.strip()) > 2:
                    chat = json.loads(chat)
                    pattern = re.findall(
                        r"<\|start_header_id\|>(.*?)<\|end_header_id\|>(.*?)<\|eot_id\|>",
                        chat['history'],
                        re.DOTALL
                    )
                    chat['messages'] = len(chat['history'].split('<|start_header_id|>user<|end_header_id|>'))-1
                    chat['chats'] =  pattern
                    chats.append(chat)
            chats.reverse()
    else:
        chats = []
    return render_template("report_template.htm", chats=json.dumps(chats))


def get_extension(filename):
    extension = filename.rsplit('.', 1)[1].lower() 
    return extension

def read_file(filepath):
    extension = get_extension(filepath)
    df = None
    if extension in ['xlsx','xls']:
        df = pd.read_excel(filepath)
    elif extension == 'csv':
        df = pd.read_csv(filepath)
    elif extension == 'json':
        df = pd.read_json(filepath)
    return df



def load_file(filepath):    
    extension = filepath.split('.')[-1].lower()   
    if extension in ['xlsx','xls']:
        df = pd.read_excel(filepath)
    elif extension == 'csv':
        df = pd.read_csv(filepath)
    elif extension == 'json':
        df = pd.read_json(filepath)
    return df

@app.route("/llm_status/", methods=["GET"])
def llm_status():
    uid = request.args.get('uid')

    event_queue = EventQueue()


    elem = event_queue.get_by_uid(uid)
    if elem:
        if 'item_size' in elem:
            return json.dumps([elem['current_item'], elem['item_size']])
        return '[0,1]'
    return '[]'


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=2337)