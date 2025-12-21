from pathlib import Path 
BASE_DIR=Path(__file__).resolve().parent
MODEL_DIR=BASE_DIR/"models"
CLASSIFIER_MODEL_DIR=MODEL_DIR/"classifier"/"sentiment"
TOKENIZER_DIR=MODEL_DIR/"embedder"/"tokenizer"
MAX_LEN=96
PRELOAD_MODELS = True
