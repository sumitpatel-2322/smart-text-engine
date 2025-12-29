from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="torch"
)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import TOKENIZER_DIR, CLASSIFIER_MODEL_PATH, MAX_LEN
import torch.nn.functional as F 
THRESHOLD=0.30
# ------------------ GLOBALS ------------------
_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    0: "Negative",
    1: "Positive"
}
# ---------------------------------------------


def _load_model():
    global _tokenizer, _model

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    if _model is None:
        _model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2
        )
        _model.load_state_dict(
            torch.load(CLASSIFIER_MODEL_PATH, map_location=_device,weights_only=True)
        )
        _model.to(_device)
        _model.eval()


def predict_sentiment(text: str) -> dict:
    _load_model()

    inputs = _tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=1)[0]

    neg_prob=probs[0].item()
    pos_prob=probs[1].item()
    margin=abs(pos_prob-neg_prob)
    if margin<THRESHOLD:
        sentiment="Neutral"
        confidence=round(1-margin,3)
    else:
        if pos_prob>neg_prob:
            sentiment="Positive"
            confidence=round(pos_prob,3)
        else:
            sentiment="Negative"
            confidence=round(neg_prob,3)
    return {
        "sentiment":sentiment,
        "confidence":confidence,
        "probabilities":{
            "positive":round(pos_prob,3),
            "negative":round(neg_prob,3)
        }
    }


def preload_classifier():
    _load_model()


if __name__ == "__main__":
    tests = [
        "The movie was fine. Some parts worked, some didn’t.",
"It’s an average film with decent acting and okay pacing.",
"The story was simple and the performances were acceptable."
    ]

    for t in tests:
        print(t)
        print(predict_sentiment(t))
        print("-" * 50)