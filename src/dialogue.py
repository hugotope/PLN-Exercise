from enum import Enum
from typing import Dict, Any
from src.entities import extract_entities

class Estado(Enum):
    IDLE = 0
    RECABANDO_DATOS = 1
    URGENCIA = 2
    RECOMENDACIONES = 3
    FINALIZAR = 4

class SistemaExperto:
    def __init__(self):
        self.contexto_paciente: Dict[str, Any] = {
            "estado_actual": Estado.IDLE,
            "intencion_actual": None,
            "slots": {
                "tipo_sintoma": None,
                "duracion": None,
                "temperatura": None,
                "gravedad_percibida": None,
                "zona_afectada": None
            }
        }

    def reset_contexto(self):
        self.contexto_paciente = {
            "estado_actual": Estado.IDLE,
            "intencion_actual": None,
            "slots": {
                "tipo_sintoma": None,
                "duracion": None,
                "temperatura": None,
                "gravedad_percibida": None,
                "zona_afectada": None
            }
        }

    def clasificar_intencion(self, mensaje: str) -> str:
        """Clasifica la intención del mensaje usando reglas simples."""
        msg_lower = mensaje.lower()

        # Palabras clave para urgencia
        urgencia_keywords = [
            "emergencia", "urgente", "grave", "muerte", "morir", "muriendo", "infarto", "accidente",
            "sangre", "desmayo", "convulsiones", "dificultad para respirar"
        ]
        if any(kw in msg_lower for kw in urgencia_keywords):
            return "urgencia"

        # Palabras clave para síntomas
        sintoma_keywords = [
            "dolor", "duele", "duelen", "fiebre", "tos", "náuseas", "mareo", "vómito", "diarrea",
            "cansancio", "fatiga", "insomnio", "dolor de cabeza", "dolor de estómago"
        ]
        if any(kw in msg_lower for kw in sintoma_keywords):
            return "síntomas"

        # Palabras clave para administrativo
        admin_keywords = [
            "cita", "horario", "turno", "receta", "factura", "pago", "renovar"
        ]
        if any(kw in msg_lower for kw in admin_keywords):
            return "administrativo"

        # Saludos
        saludo_keywords = ["hola", "buenas", "buenos días", "buenas tardes", "buenas noches", "hey", "hi"]
        if any(kw in msg_lower for kw in saludo_keywords):
            return "saludo"

        return "ruido"

    def actualizar_slots(self, mensaje: str):
        """Actualiza los slots del contexto con entidades extraídas."""
        entidades = extract_entities(mensaje)
        for key, value in entidades.items():
            if value is not None:
                self.contexto_paciente["slots"][key] = value

    def razonar(self) -> str:
        """Motor de inferencia basado en reglas IF-THEN."""
        slots = self.contexto_paciente["slots"]

        # Reglas de urgencia
        if slots["temperatura"] and slots["temperatura"] >= 39:
            return "URGENCIA_ALTA"
        if slots["zona_afectada"] == "pecho" and "dolor" in (slots["tipo_sintoma"] or ""):
            return "URGENCIA_INFARTO"
        if slots["tipo_sintoma"] and "dificultad para respirar" in slots["tipo_sintoma"]:
            return "URGENCIA_RESPIRATORIA"

        # Reglas para recomendaciones
        if slots["tipo_sintoma"] and "tos" in slots["tipo_sintoma"]:
            if slots["duracion"] and ("semana" in slots["duracion"] or "7" in slots["duracion"]):
                return "CITA_PREVIA"
            else:
                return "RECOMENDACION_DESCANSO"

        if slots["tipo_sintoma"] and "fiebre" in slots["tipo_sintoma"]:
            return "RECOMENDACION_MEDICAMENTOS"

        return "CONSULTA_GENERAL"

    def procesar_mensaje(self, mensaje: str) -> str:
        """Procesa el mensaje según el estado actual y devuelve respuesta."""
        estado = self.contexto_paciente["estado_actual"]

        if estado == Estado.IDLE:
            intencion = self.clasificar_intencion(mensaje)
            self.contexto_paciente["intencion_actual"] = intencion

            if intencion == "urgencia":
                self.contexto_paciente["estado_actual"] = Estado.URGENCIA
                return "¡Esto parece una URGENCIA! Por favor, llama inmediatamente al 112. ¿Has llamado ya?"
            elif intencion == "síntomas":
                self.contexto_paciente["estado_actual"] = Estado.RECABANDO_DATOS
                self.actualizar_slots(mensaje)
                return "¿Desde cuándo tienes este síntoma? ¿Puedes darme más detalles?"
            elif intencion == "administrativo":
                return "Para asuntos administrativos, por favor contacta con recepción al 123-456-789."
            elif intencion == "saludo":
                return "¡Hola! Soy Sallexa, tu asistente médico. ¿En qué puedo ayudarte hoy?"
            else:
                return "Hola, ¿en qué puedo ayudarte? Cuéntame tus síntomas o preguntas."

        elif estado == Estado.RECABANDO_DATOS:
            self.actualizar_slots(mensaje)
            # Verificar si tenemos suficientes datos
            slots = self.contexto_paciente["slots"]
            if slots["tipo_sintoma"]:
                if "fiebre" in slots["tipo_sintoma"] and not slots["temperatura"]:
                    return "¿Cuál es tu temperatura?"
                elif slots["duracion"] or slots["temperatura"]:
                    decision = self.razonar()
                    if "URGENCIA" in decision:
                        self.contexto_paciente["estado_actual"] = Estado.URGENCIA
                        return f"Basado en tus síntomas, esto es {decision}. Por favor, llama al 112 inmediatamente."
                    else:
                        self.contexto_paciente["estado_actual"] = Estado.RECOMENDACIONES
                        return f"Entiendo. {self.generar_recomendacion(decision)}"
            return "¿Puedes darme más información? Por ejemplo, ¿cuánto tiempo hace que lo tienes?"

        elif estado == Estado.URGENCIA:
            if "sí" in mensaje.lower() or "llamé" in mensaje.lower():
                self.contexto_paciente["estado_actual"] = Estado.FINALIZAR
                return "Bien, has llamado al 112. Quédate tranquilo, la ayuda viene en camino. ¿Necesitas algo más?"
            else:
                return "Por favor, llama al 112 AHORA. Es muy importante. ¿Has llamado?"

        elif estado == Estado.RECOMENDACIONES:
            intencion = self.clasificar_intencion(mensaje)
            if intencion == "síntomas" or intencion == "urgencia":
                # Nuevo síntoma, resetear conversación
                self.reset_contexto()
                return self.procesar_mensaje(mensaje)  # Procesar como nuevo
            elif "gracias" in mensaje.lower() or "ok" in mensaje.lower() or "nada" in mensaje.lower():
                self.contexto_paciente["estado_actual"] = Estado.FINALIZAR
                return "De nada. Si tus síntomas empeoran, no dudes en consultar de nuevo. ¡Cuídate!"
            else:
                return "¿Hay algo más en lo que pueda ayudarte?"

        elif estado == Estado.FINALIZAR:
            self.reset_contexto()
            return "Conversación finalizada. Si tienes más consultas, estoy aquí."

        return "Lo siento, no entiendo. ¿Puedes repetir?"

    def generar_recomendacion(self, decision: str) -> str:
        """Genera una recomendación basada en la decisión."""
        recomendaciones = {
            "CITA_PREVIA": "Te recomiendo pedir una cita previa con tu médico de cabecera lo antes posible.",
            "RECOMENDACION_DESCANSO": "Descansa, bebe líquidos y si no mejora en 48 horas, consulta con un médico.",
            "RECOMENDACION_MEDICAMENTOS": "Puedes tomar paracetamol si no tienes contraindicaciones. Si la fiebre persiste, consulta.",
            "CONSULTA_GENERAL": "Te sugiero consultar con un profesional de la salud para una evaluación adecuada."
        }
        return recomendaciones.get(decision, "Consulta con un médico.")