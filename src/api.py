"""
FastAPI app para Sallexa v2.0: Asistente Médico Conversacional.
Endpoints:
  - GET  / → Página de chat
  - POST /chat → Procesa mensaje del usuario y devuelve respuesta del bot
  - GET  /docs → Documentación automática
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from src.dialogue import SistemaExperto

app = FastAPI(title="Sallexa v2.0", description="Asistente Médico Conversacional")

# Instancia global del sistema experto (para demo single-user)
sistema = SistemaExperto()

# Templates
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
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

# ENDPOINT 1: GET /predict?text=... (legacy)
@app.get("/predict", response_class=JSONResponse)
async def predict(text: str):
    """
    Endpoint JSON para predicción rápida (legacy).
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

# ENDPOINT 2: GET / - Mostrar chat
@app.get("/", response_class=HTMLResponse)
async def read_chat(request: Request):
    """Muestra la interfaz de chat."""
    return templates.TemplateResponse("index.html", {"request": request})

# ENDPOINT 3: POST /chat - Procesar mensaje del usuario
@app.post("/chat", response_class=JSONResponse)
async def chat(request: Request):
    """
    Procesa un mensaje del usuario y devuelve la respuesta del bot.
    """
    try:
        form = await request.form()
        message = form.get("message", "").strip()
        
        if not message:
            return JSONResponse({"error": "Mensaje vacío"}, status_code=400)
        
        # Procesar con el sistema experto
        respuesta = sistema.procesar_mensaje(message)
        
        return JSONResponse({
            "respuesta": respuesta,
            "estado": sistema.contexto_paciente["estado_actual"].name,
            "slots": sistema.contexto_paciente["slots"]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ENDPOINT 4: POST /classify (legacy)
@app.post("/classify", response_class=HTMLResponse)
async def classify(request: Request):
    """
    Endpoint legacy para compatibilidad.
    """
    try:
        form = await request.form()
        message = form.get("message", "").strip()
        
        if not message:
            return f"<div data-classification='error'>Error: Mensaje vacío</div>"
        
        # Usar el sistema nuevo
        respuesta = sistema.procesar_mensaje(message)
        return f"<div data-respuesta='{respuesta}'></div>"
    except Exception as e:
        return f"<div data-respuesta='Error: {str(e)}'></div>"

# ENDPOINT 5: POST /reset - Resetear conversación
@app.post("/reset", response_class=JSONResponse)
async def reset():
    """Resetea el contexto de la conversación."""
    sistema.reset_contexto()
    return JSONResponse({"message": "Conversación reseteada"})

# ENDPOINT 4: GET /docs - Documentación automática (Swagger UI)
# (FastAPI lo genera automáticamente)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
