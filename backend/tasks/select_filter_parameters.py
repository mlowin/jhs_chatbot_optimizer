from werkzeug.utils import secure_filename
from tasks.jhs_task import JHSTask
from flask import render_template, request
import os
import pandas as pd
from config import Config
from callback import StartEvent, NextProcess
import json
from state_store import StateStore

class SelectFilterParameters(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    def pre(self):
        return "show"
    
    def show(self):        
        return render_template("integrations_filter_form_template.htm")
    
    def post(self):
        article_type = request.form.get('type')
        article_examples = int(request.form.get('articles'))        
        
        self._state_store.set('integration_filters_article_type', article_type)
        self._state_store.set('integration_filters_article_examples', article_examples)
                
        return NextProcess("extract_filters")