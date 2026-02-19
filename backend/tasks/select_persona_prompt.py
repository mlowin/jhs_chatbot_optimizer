from werkzeug.utils import secure_filename
from tasks.jhs_task import JHSTask
from flask import render_template, request
import os
import pandas as pd
from config import Config
from callback import StartEvent, NextProcess
import json
from state_store import StateStore

class SelectPersonaPrompt(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    def pre(self, trigger_id=None):
        return "show"
    
    def show(self):        
        article_type = self._state_store.get('integration_filters_article_type')
        return render_template("integrations_persona_prompt_template.htm",article_type=article_type)
    
    def post(self):                
        prompt = request.form.get('prompt')
        self._state_store.set('integration_persona_prompt', prompt)
        persona_number = request.form.get('persona_number')
        self._state_store.set('integration_persona_number', persona_number)
        return NextProcess("create_persona")