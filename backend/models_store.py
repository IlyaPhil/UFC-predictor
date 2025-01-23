from typing import Dict, Any


models: Dict[str, Any] = {
                            "XGBoost": None,
                            "RNN": None
                          }

def get_models() -> Dict[str, Any]:
    global models
    return models