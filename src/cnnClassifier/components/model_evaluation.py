import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
import os
from cnnClassifier.config.configuration import EvaluationConfig
from cnnClassifier.utils.common import save_json
import warnings
warnings.filterwarnings("ignore")

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path, compile=False)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        # Compile model for evaluation (needed after loading with compile=False)
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        """
        Log metrics to MLflow with robust experiment handling.
        
        Fix for: mlflow.exceptions.MissingConfigException: 'meta.yaml' does not exist
        Root Cause: MLflow's default experiment (id=0) doesn't exist in a fresh environment.
        Solution: Explicitly create/set an experiment by name before starting a run.
        """
        # 1. Set tracking URI with absolute path for cross-platform compatibility
        mlruns_path = Path(os.getcwd()) / "mlruns"
        mlflow.set_tracking_uri(f"file:///{mlruns_path.as_posix()}")
        
        # 2. Explicitly create or get experiment by name (avoids default experiment issue)
        experiment_name = "Kidney-Disease-Classification"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        # 3. Start run within the named experiment
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Log model artifact
            mlflow.keras.log_model(self.model, "model")
            
            print(f"\n✅ MLflow logging complete!")
            print(f"   Experiment: {experiment_name}")
            print(f"   Loss: {self.score[0]:.4f}, Accuracy: {self.score[1]:.4f}")
            print(f"   View runs: mlflow ui --backend-store-uri {mlruns_path}")
