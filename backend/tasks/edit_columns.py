from werkzeug.utils import secure_filename
from tasks.jhs_task import JHSTask
from flask import render_template, request
import os
import pandas as pd
from config import Config
from callback import StartEvent, NextProcess
import json
from state_store import StateStore

class EditColumns(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    def pre(self):
        return "show"
    
    def show(self):
        filename = self._state_store.get('integration_filename')
        df = self.read_file(filename)

        len_df = len(df)
        if len_df > 0:
            dict_df = df.iloc[:2].to_dict(orient='records')

        column_defintions = self._state_store.get('integration_column_definition')
        
        return render_template("integrations_dataset_template.htm", dataset=json.dumps({
            'filename':filename,
            'entries':len_df,
            'dump': dict_df,
            'struct': column_defintions
        }))
    
    
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
        
        names = request.form.getlist('name[]')
        types = request.form.getlist('type[]')
        columns = request.form.getlist('column[]')

        column_definitions = []

        for i in range(len(names)):
            column_definitions.append({'name':names[i], 'type':types[i], 'column':columns[i]})
        self._state_store.set('integration_column_definition', column_definitions)
        
        return NextProcess("select_filter_parameters")