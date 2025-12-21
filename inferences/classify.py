import tensorflow as tf
from transformers import AutoTokenizer
from config import TOKENIZER_DIR,CLASSIFIER_MODEL_DIR,MAX_LEN
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class SentimentModel(tf.keras.Model):
    def __init__(self, transformer, num_classes=2):
        super().__init__()
        self.transformer = transformer
        self.classifier = tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )

    def call(self, inputs):
        outputs = self.transformer(inputs)
        token_embeddings = outputs.last_hidden_state

        mask = tf.cast(inputs["attention_mask"], tf.float32)
        mask = tf.expand_dims(mask, axis=-1)

        pooled = tf.reduce_sum(token_embeddings * mask, axis=1)
        pooled = pooled / tf.reduce_sum(mask, axis=1)

        return self.classifier(pooled)
#Loading the tokenizer
_tokenizer=None
_model=None

LABEL_MAP={
    0:"Negative",
    1:"Positive"
}

#Loader Function
def _load_model():
    global _tokenizer, _model
    
    if _tokenizer is None:
        _tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    if _model is None:
        _model=tf.keras.models.load_model(
            CLASSIFIER_MODEL_DIR,
            custom_objects={
                "SentimentModel":SentimentModel
            }
        )

#Prediction Function 
def predict_sentiment(text:str)->str:
    _load_model()
    tokens=_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="tf"
    )
    probs=_model(tokens)
    pred=tf.argmax(probs,axis=1).numpy()[0]
    return LABEL_MAP[pred]
def preload_classifier():
    _load_model()
if __name__=="__main__":
    text="The movie was slow and boring the plot was non existent."
    print(predict_sentiment(text))