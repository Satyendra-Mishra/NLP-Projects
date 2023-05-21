import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

project_name = ""

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/engine.py",
    "config/config.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "notebooks/Bert_fine_tuning.ipynb"
]

# Create the above files
for filepath in list_of_files:
    filepath = Path(filepath)
    file_dir, file_name = os.path.split(filepath)

    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Creating directory {file_dir}")
    
    if (not filepath.exists()) or (os.path.getsize(str(filepath)) == 0):
        with open(str(filepath), mode="w"):
            pass
        logging.info(f"Creating empty file {filepath}")
    else:
        logging.info(f"{file_name} already exists")

