import re
import spacy

# Load spaCy model
try:
    nlp = spacy.load("es_core_news_sm")
except:
    nlp = None

def extract_entities(text: str) -> dict:
    """
    Extract entities from the text using regex and spaCy.
    Returns a dict with extracted slots.
    """
    entities = {
        "tipo_sintoma": None,
        "duracion": None,
        "temperatura": None,
        "gravedad_percibida": None,
        "zona_afectada": None
    }

    text_lower = text.lower()

    # Extract symptoms
    symptom_patterns = [
        r"dolor(?:es)? (?:de )?(\w+)",
        r"me duele (?:la|el|los|las)? ?(\w+)",
        r"fiebre",
        r"tos",
        r"náuseas",
        r"mareo",
        r"vómitos?",
        r"diarrea",
        r"constipación",
        r"insomnio",
        r"cansancio",
        r"fatiga"
    ]
    for pattern in symptom_patterns:
        match = re.search(pattern, text_lower)
        if match:
            if "dolor" in pattern and "de" not in pattern:
                entities["tipo_sintoma"] = f"dolor de {match.group(1)}"
            elif "duele" in pattern:
                entities["tipo_sintoma"] = f"dolor de {match.group(1)}"
            else:
                entities["tipo_sintoma"] = match.group(0).strip()
            break

    # Extract duration
    duration_patterns = [
        r"desde (ayer|hace (\d+) (días?|semanas?|meses?|horas?))",
        r"(\d+) (días?|semanas?|meses?|horas?)",
        r"una semana",
        r"dos semanas",
        r"un mes",
        r"desde (esta mañana|ayer|anoche|hace una hora)",
        r"desde hace (una hora|dos horas|un día|dos días)"
    ]
    for pattern in duration_patterns:
        match = re.search(pattern, text_lower)
        if match:
            entities["duracion"] = match.group(0).strip()
            break

    # Extract temperature
    temp_match = re.search(r"(\d+(?:[.,]\d+)?) ?(?:grados?|°(?:c|C)?)", text_lower)
    if temp_match:
        entities["temperatura"] = float(temp_match.group(1).replace(",", "."))

    # Extract gravity
    gravity_keywords = {
        "leve": "leve",
        "moderado": "moderado",
        "moderada": "moderado",
        "grave": "grave",
        "intenso": "grave",
        "fuerte": "grave",
        "mucho": "grave",
        "poco": "leve"
    }
    for word, level in gravity_keywords.items():
        if word in text_lower:
            entities["gravedad_percibida"] = level
            break

    # Extract affected area using spaCy if available
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["LOC", "MISC"]:
                entities["zona_afectada"] = ent.text.lower()
        # Also check for body parts
        body_parts = ["cabeza", "pecho", "abdomen", "pierna", "brazo", "espalda", "cuello", "estómago"]
        for token in doc:
            if token.text.lower() in body_parts:
                entities["zona_afectada"] = token.text.lower()
                break

    return entities