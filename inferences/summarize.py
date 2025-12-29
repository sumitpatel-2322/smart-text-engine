import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# ------------------ CONFIG ------------------
MODEL_NAME = "facebook/bart-large-cnn"
MAX_INPUT_TOKENS = 1024
OVERLAP_TOKENS = 50
# --------------------------------------------


_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model():
    global _tokenizer, _model

    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if _model is None:
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()

def _chunk_text(text: str):
    tokens = _tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + MAX_INPUT_TOKENS
        chunk_tokens = tokens[start:end]
        chunk_text = _tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - OVERLAP_TOKENS
    return chunks
def _summarize_chunk(
    text: str,
    max_summary_length: int = 150,
    min_summary_length: int = 40
) -> str:
    prompt = (
        "Summarize the following movie review focusing only on opinions, "
        "performances, story, pacing, and overall quality. "
        "Do not add ratings, age classifications, or factual metadata.\n\n"
    )
    inputs = _tokenizer(
        prompt+text,
        truncation=True,
        padding="max_length",
        max_length=MAX_INPUT_TOKENS,
        return_tensors="pt"
    )

    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        summary_ids = _model.generate(
            **inputs,
            max_length=max_summary_length,
            min_length=min_summary_length,
            num_beams=4,
            length_penalty=2.0,
            do_sample=False,
            early_stopping=True
        )
    return _tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

def clean_summary(text: str) -> str:
    banned_phrases = [
        "rated pg",
        "runtime",
        "box office",
        "budget",
        "director",
        "release date"
    ]

    sentences = text.split(".")
    clean_sentences = [
        s for s in sentences
        if not any(bp in s.lower() for bp in banned_phrases)
    ]

    return ".".join(clean_sentences).strip() + "."


MIN_SUMMARY_INPUT_LEN = 150
def summarize_text(text: str) -> str:
    """
    Summarize text of any length using hierarchical abstractive summarization.
    """
    if len(text.strip()) < MIN_SUMMARY_INPUT_LEN:
        return text.strip()
    _load_model()

    chunks = _chunk_text(text)

    # Short text → single pass
    if len(chunks) == 1:
        return _summarize_chunk(chunks[0])

    # Long text → chunk summaries
    chunk_summaries = []
    for chunk in chunks:
        chunk_summaries.append(_summarize_chunk(chunk))

    merged_summary = " ".join(chunk_summaries)

    # Second-pass compression
    final_summary = _summarize_chunk(
        merged_summary,
        max_summary_length=120,
        min_summary_length=60
    )
    final_summary = clean_summary(final_summary)
    return final_summary


def preload_summarizer():
    _load_model()


if __name__ == "__main__":
    test_text = """
    I went into this movie with fairly high expectations, mostly because of the cast and the amount of praise it had been receiving online. From the very first scene, it was clear that a lot of effort had gone into the visual presentation. The cinematography was impressive, with well-composed shots and a color palette that set the mood effectively. The background score also complemented the scenes nicely, never overpowering the dialogue but still adding emotional weight where needed.

The story itself starts off strong, introducing the main characters and their motivations in a way that feels natural rather than forced. The first act does a good job of building intrigue and making you care about what happens next. The lead actor delivers a convincing performance, showing a good range of emotions and making the character feel believable. Supporting characters also have their moments, especially one standout performance that adds depth to what could have been a very generic role.

However, as the movie progresses into the second act, the pacing begins to slow down noticeably. Certain scenes feel unnecessarily stretched, and there are moments where the narrative seems to lose focus. While some of these slower moments help in character development, others feel repetitive and could have been trimmed without affecting the overall story. This is where the movie tests the viewer’s patience, especially for those who prefer tighter storytelling.

The screenplay has its strengths, particularly in its dialogue. Many conversations feel authentic and grounded, avoiding overly dramatic or unrealistic exchanges. That said, there are also a few lines that feel cliché and predictable, which slightly detracts from the otherwise solid writing. The themes explored in the movie—such as ambition, regret, and personal growth—are handled with a decent level of maturity, even if they aren’t particularly groundbreaking.

One of the highlights of the film is its direction. The director clearly has a strong vision and isn’t afraid to let scenes breathe. There are several moments of silence that speak louder than words, allowing the audience to absorb the emotions of the characters. The use of close-up shots during key emotional moments is especially effective and adds intimacy to the storytelling.

As the film moves toward its climax, it regains some of the momentum it lost earlier. The conflicts introduced in the first half finally come to a head, and the stakes feel real. The emotional payoff, while not perfect, is satisfying enough to justify the buildup. The final act ties up most loose ends, though a few questions are left unanswered, which some viewers may find frustrating.

The ending itself is bittersweet and stays true to the tone of the movie. It avoids taking the easy route and instead opts for a conclusion that feels realistic, even if it’s not entirely uplifting. This choice may divide audiences, but it fits the story the film was trying to tell.

Overall, this movie is a well-made piece of cinema with strong performances, good direction, and high production values. While it does suffer from pacing issues and a few predictable moments, it still manages to leave a lasting impression. It’s not a perfect film, but it’s one that stays with you after the credits roll. I wouldn’t call it a masterpiece, but it’s definitely worth watching, especially if you appreciate character-driven stories and thoughtful filmmaking.
    """
    print(summarize_text(test_text))