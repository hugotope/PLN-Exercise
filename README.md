# Sallexa — Clasificador rápido de mensajes sanitarios

Este repositorio contiene una solución sencilla para clasificar mensajes relacionados con salud en cuatro etiquetas (por ejemplo: `urgencia`, `síntomas`, `administrativo`, `ruido`). Está pensado como ejercicio académico/prototipo, no como sistema clínico en producción.

**Contenido del repositorio**

- `dataset.csv` — CSV con los ejemplos usados para entrenar/evaluar.
- `src/` — Código fuente:
	- `src/preprocess.py` — preprocesado de texto (spaCy si está disponible; fallback con NLTK o heurísticas).
	- `src/train.py` — script para entrenar modelos, seleccionar el mejor, guardar `sallexa_model.pkl` y `vectorizer.pkl`, y generar `train_report.txt` y `confusion_matrix.csv`.
	- `src/predict.py` — script de uso local para probar el clasificador con ejemplos.
	- `src/api.py` — API web con FastAPI (endpoints `/predict`, `/`, `/classify`).
- `sallexa_model.pkl`, `vectorizer.pkl` — modelo y vectorizador guardados (si fueron generados).
- `train_report.txt`, `confusion_matrix.csv`, `confusion_matrix.png` — artefactos de evaluación.

Cómo funciona (resumen técnico)

- Preprocesado: `src/preprocess.py` normaliza texto, elimina puntuación y stopwords, y realiza lematización si `spaCy` con `es_core_news_sm` está disponible. Si no, intenta usar `nltk` (stopwords + SnowballStemmer) o un fallback simple.
- Vectorización: `src/train.py` usa `TfidfVectorizer` (unigramas y bigramas, `min_df=2`, `max_df=0.9`).
- Modelos evaluados: `MultinomialNB`, `LogisticRegression`, `LinearSVC`. El script entrena sobre una partición de entrenamiento y elige el mejor basado en F1 macro sobre el conjunto de test.
- Salidas: se guardan el modelo (`sallexa_model.pkl`) y el vectorizador (`vectorizer.pkl`), además de reportes (`train_report.txt`, matriz de confusión) y figuras.

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

3. Entrenar y guardar el modelo (genera `sallexa_model.pkl` y `vectorizer.pkl`):

```powershell
python src/train.py
```

4. Probar predicciones localmente (script de ejemplo):

```powershell
python src/predict.py
```

5. Ejecutar la API web con Uvicorn (si quieres probar la interfaz web):

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Luego abre `http://127.0.0.1:8000/` para el formulario web o `http://127.0.0.1:8000/predict?text=Tu+mensaje` para la API JSON. La documentación automática está en `http://127.0.0.1:8000/docs`.

Notas prácticas

- Si no tienes `spaCy` y su modelo `es_core_news_sm`, el preprocesado caerá al fallback de `nltk` o a una lista reducida de stopwords. Para instalar spaCy (opcional):

```powershell
pip install spacy
python -m spacy download es_core_news_sm
```

- `src/api.py` incluye una regla simple basada en palabras clave para detectar mensajes administrativos (p. ej. `horario`, `cita`, `receta`) antes de pasar el texto al modelo.

Precisión y evaluación

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

Reflexión ética (máx. 10 líneas)

Este modelo es un prototipo y no debe usarse como único criterio en decisiones clínicas. El dataset y la evaluación actuales no garantizan ausencia de falsos negativos en situaciones críticas —lo que podría poner en riesgo a pacientes— ni eliminan sesgos por idioma o registro. Antes de cualquier despliegue en entorno sanitario se requieren: validación clínica independiente, ampliación y diversificación del dataset, monitoreo continuo y un mecanismo de fallback humano para casos ambiguos o potencialmente graves.