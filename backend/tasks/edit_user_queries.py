from werkzeug.utils import secure_filename
from tasks.jhs_task import JHSTask
from flask import render_template, request
import os
import pandas as pd
from config import Config
from callback import StartEvent, NextProcess
import json
from state_store import StateStore

class EditUserQueries(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    def pre(self, trigger=None):
        return "show"
    
    def show(self):        
        user_queries = self._state_store.get('integration_user_queries')
        return render_template("integrations_edit_user_queries_template.htm", user_queries=json.dumps(user_queries))
    
    
    def post(self):
        if request.form.get('json'):            
            self._state_store.set('integration_user_queries', json.loads(request.form.get('json')))            
            return NextProcess("apply_filters_on_user_queries")
        else:
            return self.show()