"""
FastAPI app para Sallexa: API de clasificación de mensajes.
Endpoints:
  - GET  /predict?text=... → JSON {"label": "urgencia"}
  - POST /classify        → HTML con formulario y resultado
  - GET  /                → Formulario HTML
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import joblib
from src.preprocess import preprocess

app = FastAPI(title="Sallexa", description="Clasificador de mensajes sanitarios")

# Cargar modelo y vectorizador
ROOT = os.path.dirname(os.path.dirname(__file__))  # Parent dir (raíz del proyecto)
MODEL_PATH = os.path.join(ROOT, "sallexa_model.pkl")
VEC_PATH = os.path.join(ROOT, "vectorizer.pkl")

try:
    clf = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    print(f"✓ Modelo cargado desde {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error cargando modelo: {e}")
    clf = None
    vectorizer = None

# Templates
template_dir = os.path.join(ROOT, "templates")
if not os.path.exists(template_dir):
    os.makedirs(template_dir)
templates = Jinja2Templates(directory=template_dir)

def classify_message(message: str) -> str:
    """Clasifica un mensaje usando el modelo guardado."""
    # Regla simple por palabras clave para casos administrativos
    admin_keywords = [
        "horario", "horarios", "cita", "citas", "turno", "turnos",
        "receta", "recetas", "renovar", "agenda", "atencion", "atención",
        "baja", "bajas", "factura", "facturación", "pago"
    ]
    lower = message.lower()
    for kw in admin_keywords:
        if kw in lower:
            return "administrativo"

    if clf is None or vectorizer is None:
        return "error"

    clean = preprocess(message)
    X_new = vectorizer.transform([clean])
    prediction = clf.predict(X_new)
    return prediction[0]

# ENDPOINT 1: GET /predict?text=...
@app.get("/predict", response_class=JSONResponse)
async def predict(text: str):
    """
    Endpoint JSON para predicción rápida.
    Uso: GET /predict?text=Me%20duele%20la%20cabeza
    Respuesta: {"label": "síntomas"}
    """
    if not text or not text.strip():
        return JSONResponse({"error": "Parámetro 'text' vacío"}, status_code=400)
    
    try:
        label = classify_message(text)
        return JSONResponse({"label": label, "message": text})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ENDPOINT 2: GET / - Mostrar formulario
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    """Muestra el formulario HTML para clasificar mensajes."""
    return templates.TemplateResponse("index.html", {"request": request})

# ENDPOINT 3: POST /classify - Procesar formulario
@app.post("/classify", response_class=HTMLResponse)
async def classify(request: Request):
    """
    Procesa el formulario POST con un mensaje y devuelve la clasificación.
    """
    try:
        form = await request.form()
        message = form.get("message", "").strip()
        
        if not message:
            return f"<div data-classification='error'>Error: Mensaje vacío</div>"
        
        classification = classify_message(message)
        return f"<div data-classification='{classification}'></div>"
    except Exception as e:
        return f"<div data-classification='error'>Error: {str(e)}</div>"

# ENDPOINT 4: GET /docs - Documentación automática (Swagger UI)
# (FastAPI lo genera automáticamente)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
