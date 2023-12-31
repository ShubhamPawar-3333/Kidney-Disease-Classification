## Kidney Disease Classification Using MLflow and DVC

### Workflows

1. Update config.yaml
2. Update params.yaml
3. Update the entity
4. Update the configuration manager in src config
5. Update the components
6. Update the pipeline
7. Update the main.py 
8. Update the dvc.yaml
9. app.py

gdrive_link = 

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/ShubhamPawar-3333/Kidney-Disease-Classification.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n KDCenv python -y
```

```bash
conda activate KDCenv
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```

MLFLOW_TRACKING_URI=https://dagshub.com/ShubhamPawar-3333/Kidney-Disease-Classification.mlflow \
MLFLOW_TRACKING_USERNAME=ShubhamPawar-3333 \
MLFLOW_TRACKING_PASSWORD=b56ab07fcce183640c3369cb1195faf02a2ef694 \
python script.py

Run this to export as env variables:
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/ShubhamPawar-3333/Kidney-Disease-Classification.mlflow
export MLFLOW_TRACKING_USERNAME=ShubhamPawar-3333
export MLFLOW_TRACKING_PASSWORD=b56ab07fcce183640c3369cb1195faf02a2ef694
```