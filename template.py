import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(message)s')

# FileManager class manages the creation of files and folder
class FileManager:
    @staticmethod
    def create_directory(directory: Path) -> None:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Creating directory: {directory}")

    @staticmethod
    def create_empty_file(file_path: Path) -> None:
        if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
            with open(file_path, "w"):
                pass
            logging.info(f"Creating empty file: {file_path}")
        else:
            logging.info(f"{Path(file_path).name} already exists")

class ProjectInitializer:
    def __init__(self, project_name, files_to_initialize):
        self.project_name = project_name
        self.files_to_initialize = files_to_initialize

    def initialize_project(self):
        for file_path in self.files_to_initialize:
            file_path = Path(file_path)
            file_directory = file_path.parent

            if file_directory:
                FileManager.create_directory(file_directory)

            FileManager.create_empty_file(file_path)

if __name__ == "__main__":
    project_name = "cnnClassifier"
    
    files_to_initialize = [
        ".github/workflows/.gitkeep",
        f"src/{project_name}/__init__.py",
        f"src/{project_name}/components/__init__.py",
        f"src/{project_name}/utils/__init__.py",
        f"src/{project_name}/config/__init__.py",
        f"src/{project_name}/config/configuration.py",
        f"src/{project_name}/pipeline/__init__.py",
        f"src/{project_name}/entity/__init__.py",
        f"src/{project_name}/constants/__init__.py",
        "config/config.yaml",
        "dvc.yaml",
        "params.yaml",
        "requirements.txt",
        "setup.py",
        "research/trials.ipynb",
        "templates/index.html"
    ]
    
    project_initializer = ProjectInitializer(project_name, files_to_initialize)
    project_initializer.initialize_project()
