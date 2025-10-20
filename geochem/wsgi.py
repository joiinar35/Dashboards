# wsgi.py - This file must exist in PythonAnywhere
import sys
import os

# Add your app directory to the Python path
path = '/home/jolinar35/geochem'  # Cambia 'tu_usuario' y 'mi_app' por tus valores
if path not in sys.path:
    sys.path.append(path)

# Import your app
from app import application