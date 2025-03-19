from pathlib import Path

BASE_DIR = Path(__file__).parent

DATA_DIR = BASE_DIR / "player_data"
TRAINING_DIR = BASE_DIR / "training_results"
MODELS_DIR = BASE_DIR / "models"
PREDICTIONS_DIR = BASE_DIR / "predictions"
DEBUG_DIR = BASE_DIR / "debug"

DATA_DIR.mkdir(exist_ok=True)
TRAINING_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)
DEBUG_DIR.mkdir(exist_ok=True)