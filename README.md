# Sallexa

## 📝 Descripción de la actividad

Un centro de atención sanitaria recibe mensajes escritos por pacientes a través de WhatsApp. Tu objetivo es desarrollar un **sistema automático capaz de clasificar estos mensajes** según su naturaleza.

### Categorías a detectar:

- **Síntomas**
- **Administrativo** (citas, bajas, recetas…)
- **Ruido / irrelevante**
- **Urgencia potencial**

### Ejemplos:

- "Tengo 39 de fiebre y me cuesta respirar" → Urgencia
- "Necesito renovar la receta" → Administrativo
- "A qué hora cerráis hoy?" → Administrativo
- "Me duele la cabeza desde ayer" → Síntomas
- "Bombardiro crocodilo" → Ruido

## 🧩 Tareas

### **1. Preprocesamiento del texto**

Implementa una función que:

- pase el texto a minúsculas,
- elimine stopwords,
- elimine signos de puntuación,
- lematice,
- devuelva un texto limpio.

Puedes usar **spaCy** o **NLTK**.

```python
import spacy
nlp = spacy.load("es_core_news_sm")

def preprocess(text):
    [...]
    return " ".join(tokens)

print(preprocess("Tengo 38,5 de fiebre y me duele la cabeza desde ayer"))

```

### **2. Construcción del dataset**

- Crea al menos **40 mensajes** etiquetados.
- Deben aparecer todas las categorías.
- Guarda el dataset en CSV.

```python
data = [
    ("Tengo 38,5 de fiebre y me cuesta respirar", "urgencia"),
    ("Necesito renovar la receta", "administrativo"),
    ...
]


import pandas as pd
df = pd.DataFrame(data, columns=["message", "label"])
df.to_csv("dataset.csv", index=False)

```

### **3. Vectorización**

Elige uno:

- **Bag-of-Words** con `CountVectorizer`
- **TF-IDF**
- **Embeddings** de spaCy

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([preprocess(msg) for msg, label in data])
y = [label for msg, label in data]
```

### **4. Entrenamiento del modelo**

- Usa un clasificador simple:
  - Regresión logística
  - Naïve Bayes
  - SVM lineal
- Genera métricas de calidad
  - accuracy
  - precision, recall, F1
  - matriz de confusión

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# regresión logística example
clf = LogisticRegression()
# Entrenamiento
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# Métricas
print(classification_report(y_test, y_pred))
```

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# naive bayes example
clf = MultinomialNB()
# Entrenamiento
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# Métricas
print(classification_report(y_test, y_pred))
```

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM lineal example
clf = LinearSVC()
# Entrenamiento
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# Métricas
print(classification_report(y_test, y_pred))
```

> ⚠️ Nota sobre el desbalanceo de clases
>
> Es probable que el dataset tenga más ejemplos de algunas categorías que de otras (por ejemplo, mensajes administrativos). Esto puede provocar que el modelo se sesgue hacia la clase mayoritaria.
>
> Para mitigarlo, puedes:
>
> - Opción 1: habilitar el modo balanceado del clasificador:
>
> ```python
> clf = LogisticRegression(class_weight='balanced')
> clf = LinearSVC(class_weight='balanced')
> ```
>
> - Opción 2: revisar la distribución de clases antes de entrenar:
>
> ```python
> import collections
> print(collections.Counter(y))
> ```
>
> No es obligatorio usarlo, pero sí recomendable evaluarlo.

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.show()
```

### **5. Despliegue básico**

- Guarda `sallexa_model.pkl`.

```python
import joblib
joblib.dump(clf, 'sallexa_model.pkl')
```

### **6. Úsalo en producción**

Crea otro proyecto/script donde cargues el modelo y hagas predicciones sobre mensajes nuevos.

```python
import joblib
clf = joblib.load('sallexa_model.pkl')

def classify_message(message):
    clean = preprocess(message)
    tokens = preprocess([clean])
    X_new = vectorizer.transform([' '.join(tokens)])
    prediction = clf.predict(X_new)
    return prediction[0]

print(classify_message("Tengo dolor fuerte en el pecho"))
```

### **7. Prueba tu sistema**

- Prueba con al menos **5 mensajes nuevos**.

```python
test_messages = [
    "Me siento mareado y con náuseas",
    "Quiero pedir una cita con el médico",
    "¿Cuál es el horario de atención?",
    "Tengo un dolor intenso en el abdomen",
    "asdfghjkl qwertyuiop"
]
for msg in test_messages:
    print(f"Mensaje: {msg} → Clasificación: {classify_message(msg)}")
```

### **8. Créale una interfaz**

Puedes usar **Gradio** o **FastAPI + HTML** para crear una interfaz sencilla donde un usuario pueda escribir un mensaje y ver la clasificación.

#### **Opción 1 — Gradio**

```python
import gradio as gr

def bot(msg):
    return classify_message(msg)

gr.ChatInterface(bot).launch()
```

#### **Opción 2 — FastAPI + HTML**

```python
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/classify", response_class=HTMLResponse)
async def classify(request: Request):
    form = await request.form()
    message = form.get("message")
    classification = classify_message(message)
    return templates.TemplateResponse("index.html", {"request": request, "classification": classification})
```

### **9. Crea una api**

Crear un endpoint de predicción con FastAPI:

```python
GET /predict?text=...
→ {"label": "urgencia"}
```

### **10. Reflexión ética (máx. 10 líneas)**

Incluye:

- ¿Es seguro usar este modelo en un entorno sanitario real?
- ¿Qué riesgos tiene?
- ¿Qué mejoras serían necesarias?

## Entrega (obligatoria)

- `notebook.ipynb` o `src/*.py` con todo el desarrollo.
- `dataset.csv`
- `sallexa_model.pkl`
- `README.md` explicando tu solución.
