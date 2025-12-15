#!/usr/bin/env python
"""
Script para probar el clasificador de mensajes Sallexa.
Clasifica al menos 5 mensajes nuevos como pide el README.
"""

import joblib
from preprocess import preprocess

# Cargar modelo y vectorizador guardados
clf = joblib.load('../sallexa_model.pkl')
vectorizer = joblib.load('../vectorizer.pkl')

def classify_message(message: str):
    """Clasifica un mensaje usando el modelo guardado."""
    # Regla rápida: si contiene palabras clave administrativas, devolver "administrativo"
    admin_keywords = [
        "horario", "horarios", "cita", "citas", "turno", "turnos",
        "receta", "recetas", "renovar", "agenda", "atencion", "atención",
        "baja", "bajas", "factura", "facturación", "pago"
    ]
    lower = message.lower()
    for kw in admin_keywords:
        if kw in lower:
            return "administrativo"

    clean = preprocess(message)
    X_new = vectorizer.transform([clean])
    prediction = clf.predict(X_new)
    return prediction[0]

# Mensajes de prueba (al menos 5 como pide el README)
test_messages = [
    "Me siento mareado y con náuseas",
    "Quiero pedir una cita con el médico",
    "¿Cuál es el horario de atención?",
    "Tengo un dolor intenso en el abdomen",
    "asdfghjkl qwertyuiop"
]

if __name__ == '__main__':
    print("=" * 75)
    print("Pruebas del Sistema Sallexa — Clasificación de Mensajes")
    print("=" * 75)
    
    for i, msg in enumerate(test_messages, 1):
        resultado = classify_message(msg)
        print(f"\n{i}. Mensaje: {msg}")
        print(f"   → Clasificación: {resultado}")
    
    print("\n" + "=" * 75)
    print(f"Total de mensajes probados: {len(test_messages)}")
    print("=" * 75)
