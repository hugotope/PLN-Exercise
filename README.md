# Sallexa v2.0 — Asistente Médico Conversacional con Memoria y Lógica

Este repositorio contiene una evolución de Sallexa, ahora un asistente conversacional capaz de mantener contexto en diálogos, extraer entidades de síntomas, duración, temperatura, etc., y razonar con reglas IF-THEN para proporcionar recomendaciones o activar protocolos de urgencia. Está pensado como ejercicio académico/prototipo, no como sistema clínico en producción.

**Nuevas funcionalidades en v2.0:**

- **Máquina de Estados Finitos (FSM):** Gestiona el flujo de conversación (IDLE → RECABANDO_DATOS → URGENCIA/RECOMENDACIONES → FINALIZAR).
- **Extracción de Entidades:** Usa regex y spaCy para identificar síntomas, duración, temperatura, gravedad y zona afectada.
- **Sistema Experto:** Motor de inferencia con reglas para decidir urgencias (ej. fiebre ≥39°C → URGENCIA_ALTA) o recomendaciones.
- **Interfaz Conversacional:** Chat web que mantiene contexto por sesión.
- **Consideraciones Éticas:** Disclaimer legal, manejo de errores, y reflexión sobre privacidad, sesgos y responsabilidad.

**Contenido del repositorio**

- `dataset.csv` — CSV con los ejemplos usados para entrenar/evaluar (v1.0).
- `src/` — Código fuente:
	- `src/preprocess.py` — preprocesado de texto (spaCy si está disponible; fallback con NLTK o heurísticas).
	- `src/train.py` — script para entrenar modelos (v1.0).
	- `src/predict.py` — script de uso local para probar el clasificador (v1.0).
	- `src/api.py` — API web con FastAPI (endpoints `/chat`, `/`, etc. para v2.0).
	- `src/entities.py` — Extracción de entidades con NLP.
	- `src/dialogue.py` — Sistema experto con FSM y reglas de inferencia.
- `sallexa_model.pkl`, `vectorizer.pkl` — modelo y vectorizador guardados (v1.0).
- `templates/index.html` — Interfaz de chat actualizada.
- `train_report.txt`, `confusion_matrix.csv`, `confusion_matrix.png` — artefactos de evaluación (v1.0).

Cómo funciona (resumen técnico)

- **v1.0 (Clasificación de mensajes sueltos):** Preprocesado, vectorización TF-IDF, modelo ML (LogisticRegression) para clasificar en 4 categorías.
- **v2.0 (Asistente conversacional):**
  - **Estado del diálogo:** Controla el flujo basado en intención detectada.
  - **Extracción de slots:** Actualiza dinámicamente un diccionario de contexto con entidades extraídas.
  - **Razonamiento:** Reglas IF-THEN para decisiones (urgencias, recomendaciones).
  - **Respuestas:** Generadas según estado y contexto, con protocolos de emergencia.

Instalación y ejecución

1. Crear un entorno virtual (recomendado):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

3. Instalar modelo de spaCy (para extracción de entidades):

```powershell
python -m spacy download es_core_news_sm
```

4. Ejecutar la API web con Uvicorn:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Luego abre `http://127.0.0.1:8000/` para el chat conversacional. La documentación automática está en `http://127.0.0.1:8000/docs`.

Ejemplos de diálogos

- **Fiebre alta:** Usuario: "Tengo fiebre" → Bot: "¿Cuál es tu temperatura?" → Usuario: "39 grados" → Bot: "URGENCIA_ALTA. Llama al 112."
- **Dolor de pecho:** Usuario: "Me duele el pecho" → Bot: "¿Desde cuándo?" → Usuario: "Desde esta mañana" → Bot: "URGENCIA_INFARTO. Llama al 112."
- **Tos prolongada:** Usuario: "Tengo tos" → Bot: "¿Desde cuándo?" → Usuario: "Una semana" → Bot: "CITA_PREVIA con tu médico."

Reflexión ética (máx. 10 líneas)

Este sistema es un prototipo educativo y no debe usarse en entornos clínicos reales. Incluye disclaimer legal al iniciar conversaciones y maneja incertidumbre derivando a humanos si la confianza es baja (<60%). Privacidad: Los datos se almacenan en memoria volátil por sesión, sin persistencia (GDPR compliant en demo). Sesgos: El modelo puede no entender expresiones culturales variadas o de edades extremas. Responsabilidad: Cualquier recomendación errónea recae en el usuario final; el sistema advierte que no sustituye consejo médico profesional. Se mitiga con reglas conservadoras y fallback humano.

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Luego abre `http://127.0.0.1:8000/` para el formulario web o `http://127.0.0.1:8000/predict?text=Tu+mensaje` para la API JSON. La documentación automática está en `http://127.0.0.1:8000/docs`.

Notas prácticas

- Si no tienes `spaCy` y su modelo `es_core_news_sm`, la extracción de entidades será limitada. Para instalar:

```powershell
pip install spacy
python -m spacy download es_core_news_sm
```

- El sistema mantiene contexto en memoria por sesión; resetea al finalizar conversación.

Precisión y evaluación (v1.0)

El entrenamiento guarda un `train_report.txt` con la siguiente información (ejemplo generado en este repositorio):

```
best_model: LogisticRegression
best_f1_macro: 0.9990016895472014
classes_distribution: Counter({'administrativo': 2567, 'urgencia': 2530, 'ruido': 2463, 'síntomas': 2459})
```

Interpretación y limitaciones de las métricas:

- El F1 macro reportado (~0.999) indica un resultado aparentemente excelente en la partición de test usada por el script. Sin embargo, esas métricas pueden estar sesgadas por:
	- fugas de información (feature leakage) o preprocesado compartido entre train/test;
	- un dataset que no refleja el tráfico real (diferencias en lenguaje, registros y errores humanos);
	- evaluación en una sola partición en lugar de validación cruzada.

- Recomendaciones para evaluar más sólidamente: aumentar el tamaño y la diversidad del dataset, usar validación cruzada estratificada, revisar la separación train/test para evitar fugas, y calcular curvas ROC/PR y calibración de probabilidades.

Mejoras sugeridas

- Recolectar y etiquetar más datos reales y variados (diferentes pacientes, registros, dialectos).
- Añadir detección de incertidumbre (p. ej. umbrales sobre probabilidades) y rutas de escalado a revisión humana.
- Implementar pipeline de pruebas automáticas y auditoría de rendimiento por clase.