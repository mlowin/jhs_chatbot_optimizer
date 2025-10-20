# flask --app server.py --debug run

from flask import Flask, render_template, request, send_from_directory, Response, stream_with_context, send_file
import json
import os
from werkzeug.utils import secure_filename


import random
import requests

from flask_cors import CORS, cross_origin
from openai import OpenAI


UPLOAD_FOLDER = 'uploads'

app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from flask import Flask, render_template, request, send_from_directory, Response, stream_with_context, jsonify, request
import json
from flask_cors import CORS, cross_origin
import matplotlib.pyplot as plt
import datetime
import pandas as pd

app = Flask(__name__)
cors = CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'

client = OpenAI(
# base_url="http://10.10.0.20:7999/v1",
# api_key="token-ml-gpt-oss-2025",
base_url="http://10.10.0.20:8000/v1",
api_key="token-ml-llama3.1-2024",
)

def invoke_llm(system, user,temperature=0):
    completion = client.chat.completions.create(
        # model="openai/gpt-oss-120b",
        model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=temperature
    )
    output = completion.choices[0].message.content

    return output


@app.route("/")
def page_dashboard():    
    with open('llm_config.json', 'r') as f:
        llm_config = f.read()
    llms = len(json.loads(llm_config))
    df = pd.read_csv('../frontend/products_with_filter.csv')
    products = len(df)
    with open('content_filters.json', 'r', encoding="utf-8") as f:
        category_config = json.loads(f.read())
    filters = sum([len(c) for c in category_config])
    with open('../frontend/chat_log.json', 'r') as f:
            chat_log = f.read().split("\n"+"#"*50+"\n")
    chats = len(chat_log) -1
    return render_template("home_template.htm", llms=llms, products = products, filters=filters, chats=chats)

@app.route("/llms/", methods=['GET'])
def page_llms():    
    with open('llm_config.json', 'r') as f:
        llm_config = f.read()
    return render_template("llm_template.htm", llm_config=llm_config)

@app.route("/llms/", methods=['POST'])
def save_llms():    
    
    with open('llm_config.json', 'w') as f:
        f.write(json.dumps(request.json))
    return "true"

@app.route("/products/", methods=['GET'])
def page_products():    
    df = pd.read_csv('../frontend/products_with_filter.csv')
    df = df[['Titel', 'Beschreibung', 'Altersempfehlung_ab', 'Altersempfehlung_bis',
       'Anzahl_Spieler_ab', 'Anzahl_Spieler_bis', 'Spieldauer_ab',
       'Spieldauer_bis', 'Preis', 'Altersempfehlung', 'Spieleranzahl',
       'Spieldauer', 'Komplexität', 'Thema', 'Spielmechanik', 'Sprache',
       'Preisspanne', 'Empfehlungen', 'Kategorie', 'Artikelnummer']]
    return render_template("product_template.htm", products=json.dumps(df.to_dict(orient='records')))

@app.route("/categories/", methods=['GET'])
def page_categories():    
    with open('content_filters.json', 'r', encoding="utf-8") as f:
        category_config = f.read()
    return render_template("category_template.htm", category_config=category_config)

@app.route("/categories/", methods=['POST'])
def save_categories():        
    with open('content_filters.json', 'w') as f:
        f.write(json.dumps(request.json))
    return "true"

@app.route("/integrations/", methods=['GET'])
def page_integrations():        
    return render_template("integrations_template.htm")


@app.route("/reports/", methods=['GET'])
def page_reports():    
    with open('../frontend/chat_log.json', 'r') as f:
        chat_log = f.read().split("\n"+"#"*50+"\n")
        chats = []
        for chat in chat_log:
            if len(chat.strip()) > 2:
                chat = json.loads(chat)
                chat['messages'] = len(chat['history'].split('<|start_header_id|>user<|end_header_id|>'))-1
                chats.append(chat)
        chats.reverse()
    return render_template("report_template.htm", chats=json.dumps(chats))




@app.route("/upload_dataset/", methods=["POST"])
def upload_dataset():
    return_obj = {
        'success':False,
        'message':""
    }
    if 'file' not in request.files:
        return_obj['message'] = "file missing"
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':        
        return_obj['message'] = "file empty"
    extension = file.filename.rsplit('.', 1)[1].lower() 
    if file and extension in {'xlsx', 'xls', 'json', 'csv'}:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        if extension in ['xlsx','xls']:
            df = pd.read_excel(filepath)
        elif extension == 'csv':
            df = pd.read_csv(filepath)
        elif extension == 'json':
            df = pd.read_json(filepath)

        len_df = len(df)
        if len_df > 2:
            dict_df = df.iloc[:2].to_dict(orient='records')
            with open('prompts/read_dataset.txt','r') as f:
                prompt = f.read()
            response = invoke_llm(prompt, json.dumps(dict_df))
            # with open('tmp.tmp','r') as f:
            #     response = f.read()
            try:
                response = json.loads(response)
                if type(response) == list:
                    response = response[0]
            except:
                print("could not parse")
        
        return render_template("integrations_dataset_template.htm", dataset=json.dumps({
            'filename':filename,
            'entries':len_df,
            'dump': dict_df,
            'struct': response
        }))

    return_obj['message'] = "filetype not allowed"
    return json.dumps(return_obj)


@app.route("/integrations_filters/", methods=["POST"])
def integrations_filter():

    names = request.form.getlist('name[]')
    types = request.form.getlist('type[]')
    columns = request.form.getlist('column[]')

    file_obj = {'file': request.form.get('filename'),'columns':[]}

    for i in range(len(names)):
        file_obj['columns'].append({'name':names[i], 'type':types[i], 'column':columns[i]})
        
    with open('file_config.json', 'w') as f:
        f.write(json.dumps(file_obj))
    return render_template("integrations_filter_form_template.htm")

@app.route("/integrations_create_filters/", methods=["POST"])
def integrations_create_filters():
    article_type = request.form.get('type')
    article_examples = int(request.form.get('articles'))

    with open('file_config.json', 'r') as f:
        file_config = json.loads(f.read())

    
    extension = file_config['file'].rsplit('.', 1)[1].lower()     
    filepath = os.path.join(UPLOAD_FOLDER, file_config['file'])
    if extension in ['xlsx','xls']:
        df = pd.read_excel(filepath)
    elif extension == 'csv':
        df = pd.read_csv(filepath)
    elif extension == 'json':
        df = pd.read_json(filepath)

    dataset_filters = {}
    for col in file_config['columns']:
        if col['column'] == 'filter':
            dataset_filters[col['column']] = list(df[col['name']].value_counts().keys()[:10])

    with open('prompts/create_filters.txt','r') as f:
        prompt = f.read()
        prompt = prompt.replace("%%%TYPE%%%", article_type).replace("%%%FILTER_STR%%%",json.dumps(dataset_filters))

    lst_txt_columns = [col["name"] for col in file_config['columns'] if col['column'] in ['title','description']]
    lst_content_filters = []
    for i in range(article_examples // 10):    
        df_random = df.sample(n=10)
        documents = df_random[lst_txt_columns].astype(str).apply(lambda r: " - ".join(r),axis=1).tolist()
        lst_content_filters.append(invoke_llm(prompt, "Filter:"+prompt+"\nProdukte:\n"+"\n".join(documents)))

   

    with open('prompts/merge_filters.txt','r') as f:
        prompt_merge = f.read()
        prompt_merge = prompt_merge.replace("%%%TYPE%%%", article_type).replace("%%%FILTER_STR%%%",json.dumps(dataset_filters))

    content_filters = invoke_llm(prompt_merge, json.dumps(lst_content_filters))

    file_config['content_filters'] = json.loads(content_filters)
    file_config['article_type'] = article_type

    with open('file_config.json', 'w') as f:
        f.write(json.dumps(file_config))
    
    return render_template("category_template.htm", category_config=file_config['content_filters'], in_process=True)


@app.route("/integrations_create_dataset/", methods=["POST"])
def integrations_create_dataset():
    json_str = request.form.get('json')
    content_filters = json.loads(json_str)   

    with open('file_config.json', 'r') as f:
        file_config = json.loads(f.read())
    
    file_config['content_filters'] = content_filters
    
    with open('file_config.json', 'w') as f:
        f.write(json.dumps(file_config))
        
    return render_template("category_template.htm", category_config=file_config['content_filters'], in_process=True)

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


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=2337)