from werkzeug.utils import secure_filename
from tasks.jhs_task import JHSTask
from flask import render_template, request
import os
import pandas as pd
from config import Config
from callback import StartEvent, NextProcess
import json
from state_store import StateStore

class EditFilters(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    def pre(self, trigger=None):
        return "show"
    
    def show(self):
        content_filters = self._state_store.get('integration_content_filters')
        return render_template("category_template.htm", category_config=content_filters, in_process=True)
    
    
    def post(self):
        message = None
        
        content_filters = request.form.get('json')
    
        self._state_store.set('integration_content_filters', json.loads(content_filters))
        
        return NextProcess("apply_filters_on_products")