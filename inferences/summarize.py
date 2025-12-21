#importing libraries 
import tensorflow as tf
import nltk
from transformers import TFAutoModel, AutoTokenizer
from nltk.tokenize import sent_tokenize
#importing custom variables from the custom config file
from config import TOKENIZER_DIR,MAX_LEN

_tokenizer = None
_embedder = None
_punkt_ready = False


def _ensure_punkt():
    global _punkt_ready
    if not _punkt_ready:
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt")
            nltk.download("punkt_tab")
        _punkt_ready = True

def _load_embedder():
    global _tokenizer, _embedder

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    if _embedder is None:
        _embedder = TFAutoModel.from_pretrained(
            "distilbert-base-uncased",
            use_safetensors=False
        )


def split_into_sentences(text: str):
    _ensure_punkt()
    return sent_tokenize(text)


def embed_sentences(sentences):
    _load_embedder()

    tokens = _tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="tf"
    )

    outputs = _embedder(tokens)
    token_embeddings = outputs.last_hidden_state

    mask = tf.cast(tokens["attention_mask"], tf.float32)
    mask = tf.expand_dims(mask, axis=-1)

    summed = tf.reduce_sum(token_embeddings * mask, axis=1)
    counts = tf.reduce_sum(mask, axis=1)

    return summed / counts


def compute_centroid(sentence_embeddings):
    return tf.reduce_mean(sentence_embeddings, axis=0)


def cosine_similarity(a, b):
    a = tf.nn.l2_normalize(a, axis=-1)
    b = tf.nn.l2_normalize(b, axis=-1)
    return tf.reduce_sum(a * b, axis=-1)


def score_sentences(sentence_embeddings):
    centroid = compute_centroid(sentence_embeddings)
    return cosine_similarity(sentence_embeddings, centroid)


def select_top_x(sentences, scores, x=3):
    ranked_indices = tf.argsort(scores, direction="DESCENDING")[:x]
    ranked_indices = tf.sort(ranked_indices)
    return [sentences[i] for i in ranked_indices.numpy()]

#For summarizing
def summarize_text(text: str, x: int = 3) -> str:
    sentences = split_into_sentences(text)

    if len(sentences) <= x:
        return text

    embeddings = embed_sentences(sentences)
    scores = score_sentences(embeddings)
    summary_sentences = select_top_x(sentences, scores, x)

    return " ".join(summary_sentences)
#preload function 
def preload_summarizer():
    _ensure_punkt()
    _load_embedder()