from flask import Flask

app = Flask(__name__)

# Import routes AFTER creating `app` to avoid circular import issues
from app import routes  
