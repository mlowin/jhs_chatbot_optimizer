from werkzeug.utils import secure_filename
from tasks.jhs_task import JHSTask
from flask import render_template, request
import os
import pandas as pd
from config import Config
from callback import StartEvent, NextProcess
import json
from state_store import StateStore

class EditPersona(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    def pre(self, trigger=None):
        return "show"
    
    def show(self):
        print("STATUS",self._state_store.get('integration_current_status'))
        
        persona_list = self._state_store.get('integration_persona_list')[0]
        return render_template("integrations_edit_persona_template.htm", persona_list=persona_list)
    
    
    def post(self):
        if request.form.get('json'):            
            self._state_store.set('integration_persona_list', json.loads(request.form.get('json')))            
            return NextProcess("select_user_query_prompt")
        else:
            return self.show()