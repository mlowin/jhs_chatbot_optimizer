from werkzeug.utils import secure_filename
from tasks.jhs_task import JHSTask
from flask import render_template, request, redirect
import os
import pandas as pd
from config import Config
from callback import StartEvent, NextProcess
import json
from state_store import StateStore

class SelectFile(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    def pre(self):
        self._state_store.set('select_file_message',None)
        return "show"
    
    def show(self):
        message = self._state_store.get('select_file_message')
        if os.path.exists('llm_config.json'):
            with open('llm_config.json', 'r') as f:
                if len(json.loads(f.read())) > 0:
                    return render_template("integrations_template.htm", message=message)
        return redirect('/llms/')
    
    
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
    
    def post(self):
        message = None
        
        if 'file' not in request.files:
            message = "file missing"
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':        
            message = "file empty"
        extension = self.get_extension(file.filename) 
        if file and extension in {'xlsx', 'xls', 'json', 'csv'}:
            filename = secure_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            state_store = StateStore()
            state_store.set('integration_filename', filepath)
            return NextProcess("extract_columns")
        else:
            message = "wrong file type"
        
        self._state_store.set('select_file_message',message)
        self.show()   