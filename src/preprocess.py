import re
import string

def _default_stopwords():
    # Small Spanish stopword list (fallback)
    return {"de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "por", "un", "para", "con", "no", "una",
            "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre",
            "cuando", "muy", "sin", "sobre", "también", "me", "ya", "hay", "todos", "son", "dos", "también", "fue",
            "ha", "tener", "tengo", "tiene"}

# Try spaCy
try:
    import spacy
    try:
        nlp = spacy.load("es_core_news_sm")
    except Exception:
        nlp = None
except Exception:
    spacy = None
    nlp = None


def preprocess(text: str):
    """
    Preprocess Spanish text:
    - lowercase
    - remove punctuation
    - remove stopwords
    - lemmatize when spaCy is available
    Returns a cleaned string.
    """

    if not isinstance(text, str):
        text = str(text)

    # Lowercase
    text = text.lower()

    # Normalize decimals and remove punctuation
    text = text.replace(",", ".")
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # ---- spaCy branch ----
    if nlp is not None:
        doc = nlp(text)
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and token.lemma_.strip()
        ]
        return " ".join(tokens)

    # ---- Fallback: NLTK or simple tokenizer ----
    try:
        from nltk.corpus import stopwords as _nltk_stop
        from nltk.stem import SnowballStemmer
        import nltk

        # Ensure stopwords are downloaded
        try:
            _ = _nltk_stop.words("spanish")
        except Exception:
            nltk.download("stopwords")

        stop = set(_nltk_stop.words("spanish"))
        stemmer = SnowballStemmer("spanish")

    except Exception:
        stop = _default_stopwords()
        stemmer = None

    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    cleaned = []

    for t in tokens:
        if t in stop:
            continue
        if stemmer:
            t = stemmer.stem(t)
        cleaned.append(t)

    return " ".join(cleaned)


if __name__ == "__main__":
    print(preprocess("Tengo 38,5 de fiebre y me duele la cabeza desde ayer"))
