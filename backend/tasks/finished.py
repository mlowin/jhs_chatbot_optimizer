from werkzeug.utils import secure_filename
from tasks.jhs_task import JHSTask
from flask import render_template, request
import os
import pandas as pd
from config import Config
from callback import StartEvent, NextProcess
import json
from state_store import StateStore

class Finished(JHSTask):

    def __init__(self):
        self._state_store = StateStore()

    def pre(self, trigger=None):
        return "show"
    
    def show(self):
        return render_template("finished_template.htm")
    
    
    def post(self):
        return render_template("finished_template.htm")
       