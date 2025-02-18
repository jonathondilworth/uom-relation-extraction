# solution from: https://stackoverflow.com/questions/48946639/how-to-create-and-open-a-jupyter-notebook-ipynb-file-directly-from-terminal
"""create-notebook.py
   Creates a minimal jupyter notebook (.ipynb)
   Usage: create-notebook <notebook>
"""
import sys
from jupyter_server import transutils as _
from jupyter_server.services.contents.filemanager import FileContentsManager as FCM

try:
    notebook_fname = sys.argv[1].strip('.ipynb')
except IndexError:
    print("Usage: create-notebook <notebook>")
    exit()

notebook_fname += '.ipynb'  # ensure .ipynb suffix is added
new_file = FCM()
new_file.new(path=notebook_fname)
