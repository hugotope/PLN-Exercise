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
import uuid
import json
from datetime import datetime
from src.dialogue import SistemaExperto

app = FastAPI(title="Sallexa v2.0", description="Asistente Médico Conversacional")

# Instancia global de sesiones
sessions = {}

# Archivo de logs
LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "conversations.log")

# Templates
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
if not os.path.exists(template_dir):
    os.makedirs(template_dir)
templates = Jinja2Templates(directory=template_dir)

def get_or_create_session(session_id: str) -> SistemaExperto:
    if session_id not in sessions:
        sessions[session_id] = SistemaExperto()
    return sessions[session_id]

def log_conversation(session_id: str, user_msg: str, bot_response: str):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "user_message": user_msg,
        "bot_response": bot_response
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

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
    Maneja sesiones por cookie.
    """
    try:
        # Obtener session_id de cookie, o generar nueva
        session_id = request.cookies.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())
        
        form = await request.form()
        message = form.get("message", "").strip()
        
        if not message:
            return JSONResponse({"error": "Mensaje vacío"}, status_code=400)
        
        # Obtener o crear sesión
        sistema = get_or_create_session(session_id)
        
        # Procesar mensaje
        respuesta = sistema.procesar_mensaje(message)
        
        # Loggear conversación
        log_conversation(session_id, message, respuesta)
        
        # Preparar respuesta
        response_data = {
            "respuesta": respuesta,
            "estado": sistema.contexto_paciente["estado_actual"].name,
            "slots": sistema.contexto_paciente["slots"]
        }
        
        # Crear respuesta con cookie
        response = JSONResponse(response_data)
        response.set_cookie(key="session_id", value=session_id, httponly=True)
        
        return response
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
async def reset(request: Request):
    """Resetea el contexto de la conversación."""
    session_id = request.cookies.get("session_id")
    if session_id and session_id in sessions:
        del sessions[session_id]
    return JSONResponse({"message": "Conversación reseteada"})

# ENDPOINT 4: GET /docs - Documentación automática (Swagger UI)
# (FastAPI lo genera automáticamente)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
