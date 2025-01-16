from src.components.model_eval import ModelEvaluator
from src.entity.config_entity import ModelEvaluationConfig

def run_model_evaluation(config, model, X_test, y_test):
    model_evaluation_config = ModelEvaluationConfig(**config["model_evaluation"])
    model_evaluator = ModelEvaluator(model_evaluation_config)
    accuracy = model_evaluator.initiate_model_evaluation(model, X_test, y_test)
    return accuracy
  
