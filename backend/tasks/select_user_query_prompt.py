from werkzeug.utils import secure_filename
from tasks.jhs_task import JHSTask
from flask import render_template, request
import os
import pandas as pd
from config import Config
from callback import StartEvent, NextProcess
import json
from state_store import StateStore

class SelectUserQueryPrompt(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    def pre(self):
        return "show"
    
    def show(self):        
        article_type = self._state_store.get('integration_filters_article_type')
        return render_template("ingetrations_user_query_prompt_template.htm",article_type=article_type)
    
    def post(self):                
        prompt = request.form.get('prompt')
        self._state_store.set('integration_user_query_prompt', prompt)
        user_query_number = request.form.get('user_query_number')
        self._state_store.set('integration_user_query_number_total', user_query_number)
        articles_number = request.form.get('articles_number')
        self._state_store.set('integration_user_query_articles_number', articles_number)
        # user_query_number_per_iteration = request.form.get('user_query_number_per_iteration')
        # self._state_store.set('integration_user_query_number_per_iteration', user_query_number_per_iteration)
        return NextProcess("create_user_queries")