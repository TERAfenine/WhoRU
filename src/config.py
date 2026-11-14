from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

IMAGE_SIZE = (300, 300)
BATCH_SIZE = 16
SEED = 42
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 5