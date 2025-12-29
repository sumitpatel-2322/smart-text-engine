from pathlib import Path 
BASE_DIR=Path(__file__).resolve().parent
MODEL_DIR=BASE_DIR/"models"
CLASSIFIER_MODEL_PATH = MODEL_DIR / "classifier" / "sentiment" / "best_model.pt"
TOKENIZER_DIR=MODEL_DIR/"classifier"/"tokenizer"
MAX_LEN=256
PRELOAD_MODELS = True
