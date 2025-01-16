from src.components.model_eval import ModelEvaluator
from src.entity.config_entity import ModelEvaluationConfig

def run_model_evaluation(config, model, X_test, y_test):
    model_evaluation_config = ModelEvaluationConfig(**config["model_evaluation"])
    model_evaluator = ModelEvaluator(model_evaluation_config)
    accuracy = model_evaluator.initiate_model_evaluation(model, X_test, y_test)
    return accuracy




# src/pipeline/model_eval_pipeline.py
from src/components.model_eval import ModelEvaluation
from src.entity.config_entity import ModelEvalConfig

class ModelEvalPipeline:
    def __init__(self, config: ModelEvalConfig):
        self.config = config
        self.model_eval = ModelEvaluation(self.config)

    def run(self):
        self.model_eval.evaluate_model()
        
