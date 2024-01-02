import os
import zipfile
import gdown
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig

def id_extractor(url:str)->str:
    return url.split("/")[-2]

def load_dataset(file_id:str, path:Path)->None:
    prefix = "https://drive.google.com/uc?/export=download&id="
    gdown.download(prefix+file_id, path)
    

class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config
        
    def download_file(self)->str:
        '''
        fetch data from url
        '''
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")
            
            load_dataset(id_extractor(url=dataset_url), path=zip_download_dir)
            logger.info(f"Data downloaded from {dataset_url} into file {zip_download_dir}")
            
        except Exception as e:
            raise e
        
    def extract_zip_file(self)->None:
        """
        zip_file_path:str
        Extracts the zip file into the data directory
        function return None 
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)