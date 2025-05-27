from modules.registry import get_current_model
from config import *
print(get_current_model(EXPERIMENT_NAME, "calinski_harabasz_score", "MINIMIZE"))