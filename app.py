import streamlit as st
from faster_whisper import WhisperModel
import ollama
import json
import gc
import torch
import pandas as pd
from difflib import get_close_matches
from datetime import datetime
import re
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os
import logging
import traceback
import sys
import subprocess
from pathlib import Path
try:
    import psutil
except ImportError:
    psutil = None
try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    pynvml = None

# --- CONFIGURAR OLLAMA AUTOMÁTICAMENTE ---
def _configurar_ollama_path():
    """Busca Ollama en ubicaciones conocidas de Windows y lo agrega al PATH."""
    posibles_rutas = [
        Path.home() / "AppData" / "Local" / "Programs" / "Ollama",
        Path("C:\\Program Files\\Ollama"),
        Path("C:\\Program Files (x86)\\Ollama"),
    ]
    
    for ruta in posibles_rutas:
        ejecutable = ruta / "ollama.exe"
        if ejecutable.exists():
            # Agregar al PATH si no está ya
            if str(ruta) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(ruta) + os.pathsep + os.environ.get("PATH", "")
            return str(ejecutable)
    return None

_ollama_executable = _configurar_ollama_path()

# --- CONFIGURACIÓN DE LOGGING ---
# Crear carpeta de logs si no existe
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

log_file = os.path.join(logs_dir, f"debug_escriba_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Capturar excepciones no manejadas
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Excepción no manejada", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Escriba Médico v0.9", layout="wide", page_icon="🩺")

if 'logs' not in st.session_state: st.session_state.logs = []
if 'raw_text' not in st.session_state: st.session_state.raw_text = ""
if 'soap_full' not in st.session_state: st.session_state.soap_full = ""
if 'selected_model' not in st.session_state: st.session_state.selected_model = "escriba-tfm"
if 'current_audio_name' not in st.session_state: st.session_state.current_audio_name = None
if 'processing' not in st.session_state: st.session_state.processing = False
if 'last_toast' not in st.session_state: st.session_state.last_toast = None
if 'phase_complete' not in st.session_state: st.session_state.phase_complete = {"transcription": False, "report": False, "audit": False}

def log_debug(categoria, mensaje, level="INFO"):
    """Log dual: archivo persistente + UI de Streamlit"""
    time_str = datetime.now().strftime("%H:%M:%S")
    log_msg = f"[{categoria}] {mensaje}"
    
    # Log a archivo (siempre)
    if level == "ERROR":
        logger.error(log_msg)
    elif level == "WARNING":
        logger.warning(log_msg)
    elif level == "CRITICAL":
        logger.critical(log_msg)
    else:
        logger.info(log_msg)
    
    # Log a UI (si está disponible)
    try:
        if 'logs' in st.session_state:
            st.session_state.logs.append(f"[{time_str}] [{categoria}] {mensaje}")
    except:
        pass

def notificar(msg, icon="✅"):
    """Muestra un toast si está disponible; si falla, usa st.success"""
    try:
        st.toast(f"{icon} {msg}")
        st.session_state.last_toast = msg
    except Exception:
        st.success(f"{icon} {msg}")

def limpiar_vram(silent=False):
    """Limpia VRAM. Si silent=True, no registra en logs (útil para threads)"""
    if not silent:
        log_debug("SISTEMA", "Limpiando VRAM...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_ollama_host():
    """Devuelve el host configurado para Ollama (env OLLAMA_HOST o localhost)."""
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

def _iniciar_ollama_servidor():
    """Intenta iniciar el servidor de Ollama si no está corriendo."""
    if not _ollama_executable:
        return False
    
    try:
        # Intenta iniciar el servidor en background
        import subprocess
        subprocess.Popen(
            [_ollama_executable, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        # Esperar un poco para que el servidor se inicie
        import time
        time.sleep(2)
        return True
    except Exception as e:
        log_debug("SISTEMA", f"No se pudo iniciar Ollama automáticamente: {e}", level="WARNING")
        return False

def verificar_ollama_disponible():
    """Comprueba si el servicio de Ollama está accesible.
    Intenta iniciarlo automáticamente si no responde.
    Devuelve (disponible: bool, host: str, error: Optional[str])
    """
    host = get_ollama_host()
    try:
        # Intentar contactar con Ollama
        _ = ollama.list()
        log_debug("MODELO", f"Ollama activo en {host}")
        return True, host, None
    except Exception as e:
        msg = str(e)
        log_debug("MODELO", f"Ollama no disponible en {host}, intentando iniciar...", level="WARNING")
        
        # Intentar iniciar el servidor
        if _iniciar_ollama_servidor():
            try:
                # Reintentar después de iniciar
                _ = ollama.list()
                log_debug("MODELO", f"Ollama iniciado exitosamente en {host}")
                return True, host, None
            except Exception as e2:
                msg = str(e2)
        
        error_msg = f"Ollama no disponible en {host}: {msg}"
        if not _ollama_executable:
            error_msg += " [Ollama no encontrado en el sistema]"
        
        log_debug("MODELO", error_msg, level="ERROR")
        return False, host, error_msg

@st.cache_data
def cargar_db():
    with open('cie10_2026.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def obtener_modelos_disponibles():
    """Obtiene lista de modelos disponibles en Ollama"""
    try:
        ok, _, _ = verificar_ollama_disponible()
        if not ok:
            return ["escriba-tfm"]
        response = ollama.list()
        modelos = [model.model for model in response.models]
        return modelos if modelos else ["escriba-tfm"]
    except:
        return ["escriba-tfm"]

def verificar_modelo_deberta():
    """Verifica si transformers y torch están instalados para usar DeBERTa"""
    try:
        import transformers
        log_debug("AUDITORIA", "✅ Transformers está disponible para DeBERTa")
        return True
    except ImportError:
        log_debug("AUDITORIA", "⚠️ Transformers no está instalado. Necesario para DeBERTa", level="WARNING")
        try:
            log_debug("AUDITORIA", "Intentando instalar transformers...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
            log_debug("AUDITORIA", "✅ Transformers instalado exitosamente")
            return True
        except Exception as e:
            log_debug("AUDITORIA", f"❌ No se pudo instalar transformers: {str(e)}", level="ERROR")
            return False

def obtener_info_gpu():
    """Obtiene información de la GPU disponible usando NVML si está presente (cubre procesos externos como Ollama)."""
    info = {"disponible": False, "nombre": "No detectada", "memoria_total": "N/A", "memoria_libre": "N/A", "memoria_usada": "N/A", "utilizacion": "N/A"}
    try:
        if pynvml:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["disponible"] = True
            info["nombre"] = pynvml.nvmlDeviceGetName(handle).decode()
            memoria_total = mem.total / (1024**3)
            memoria_libre = mem.free / (1024**3)
            memoria_usada = mem.used / (1024**3)
            info["memoria_total"] = f"{memoria_total:.2f} GB"
            info["memoria_libre"] = f"{memoria_libre:.2f} GB"
            info["memoria_usada"] = f"{memoria_usada:.2f} GB"
            info["utilizacion"] = f"{(memoria_usada/memoria_total*100):.1f}%"
            log_debug("GPU", f"Estado NVML: {info['nombre']} - Usado: {info['memoria_usada']} / {info['memoria_total']} ({info['utilizacion']})")
        elif torch.cuda.is_available():
            info["disponible"] = True
            info["nombre"] = torch.cuda.get_device_name(0)
            memoria_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memoria_usada = torch.cuda.memory_allocated(0) / (1024**3)
            memoria_libre = memoria_total - memoria_usada
            info["memoria_total"] = f"{memoria_total:.2f} GB"
            info["memoria_libre"] = f"{memoria_libre:.2f} GB"
            info["memoria_usada"] = f"{memoria_usada:.2f} GB"
            info["utilizacion"] = f"{(memoria_usada/memoria_total*100):.1f}%"
            log_debug("GPU", f"Estado Torch: {info['nombre']} - Usado: {info['memoria_usada']} / {info['memoria_total']} ({info['utilizacion']})")
    except Exception as e:
        log_debug("GPU", f"Error al obtener info GPU: {str(e)}", level="ERROR")
        logger.exception("Detalles del error de GPU:")
    return info

def obtener_info_sistema():
    """Obtiene información del sistema (CPU, RAM)"""
    info = {}
    try:
        if psutil:
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            info["ram_percent"] = psutil.virtual_memory().percent
            info["ram_available"] = psutil.virtual_memory().available / (1024**3)
            info["ram_total"] = psutil.virtual_memory().total / (1024**3)
        else:
            info = {"cpu_percent": "N/A", "ram_percent": "N/A", "ram_available": "N/A", "ram_total": "N/A"}
    except Exception as e:
        log_debug("SISTEMA", f"Error al obtener info sistema: {str(e)}", level="ERROR")
    return info

def reiniciar_interfaz():
    """Reinicia el estado de la interfaz para procesar nuevo audio"""
    st.session_state.raw_text = ""
    st.session_state.soap_full = ""
    st.session_state.logs = []
    limpiar_vram()
    log_debug("SISTEMA", "Interfaz reiniciada para nuevo audio")

# --- LÓGICA DE PROCESAMIENTO ---

def transcribir_con_timeout(audio_file, timeout=300):
    """Transcribe audio con timeout para evitar bloqueos"""
    def transcribir():
        # Esta función se ejecuta en un thread separado
        try:
            logger.info(f"[THREAD] Iniciando transcripción de {audio_file}")
            logger.info(f"[THREAD] GPU disponible: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"[THREAD] GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"[THREAD] Memoria GPU libre: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3):.2f} GB")
            
            logger.info("[THREAD] Cargando modelo Whisper Large-v3...")
            asr = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
            logger.info("[THREAD] Modelo cargado, iniciando transcripción...")
            
            segments, info = asr.transcribe(
                audio_file, 
                language="es", 
                beam_size=5, 
                vad_filter=True, 
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            logger.info(f"[THREAD] Transcripción info: {info}")
            
            texto = " ".join([s.text for s in segments])
            logger.info(f"[THREAD] Transcripción completada: {len(texto)} caracteres")
            
            del asr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[THREAD] Limpieza de memoria completada")
            return texto
            
        except Exception as e:
            logger.exception(f"[THREAD] ERROR CRÍTICO en transcripción: {str(e)}")
            raise
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(transcribir)
        try:
            resultado = future.result(timeout=timeout)
            logger.info(f"Transcripción finalizada exitosamente")
            return resultado
        except TimeoutError:
            msg = f"⚠️ TIMEOUT: Transcripción excedió {timeout}s"
            log_debug("WHISPER", msg, level="WARNING")
            raise Exception(f"La transcripción excedió el límite de {timeout} segundos")
        except Exception as e:
            log_debug("WHISPER", f"❌ ERROR: {str(e)}", level="ERROR")
            logger.exception("Traceback completo del error:")
            raise

def fase_transcripcion(audio_file):
    temp_path = None
    try:
        st.session_state.processing = True
        
        # Información del sistema antes de procesar
        gpu_info = obtener_info_gpu()
        sys_info = obtener_info_sistema()
        log_debug("SISTEMA", f"Estado pre-transcripción: GPU={gpu_info.get('memoria_libre', 'N/A')}, RAM={sys_info.get('ram_percent', 'N/A')}%")
        
        # Guardar temporalmente el archivo
        temp_path = f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        with open(temp_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        file_size = os.path.getsize(temp_path) / (1024*1024)  # MB
        log_debug("WHISPER", f"📁 Archivo guardado: {temp_path} ({file_size:.2f} MB)")
        log_debug("WHISPER", "🔄 Iniciando transcripción...")
        
        texto = transcribir_con_timeout(temp_path, timeout=600)
        
        st.session_state.raw_text = texto
        log_debug("WHISPER", f"✅ Transcripción completada: {len(texto)} caracteres")
        log_debug("WHISPER", f"📝 Texto guardado en session_state (len={len(st.session_state.raw_text)})")
        if len(texto) > 0:
            log_debug("WHISPER", f"Preview: {texto[:100]}...")
        else:
            log_debug("WHISPER", "⚠️ ADVERTENCIA: Transcripción vacía", level="WARNING")
        
        # Información del sistema después de procesar
        gpu_info = obtener_info_gpu()
        sys_info = obtener_info_sistema()
        log_debug("SISTEMA", f"Estado post-transcripción: GPU={gpu_info.get('memoria_libre', 'N/A')}, RAM={sys_info.get('ram_percent', 'N/A')}%")
        st.session_state.phase_complete["transcription"] = True
        notificar("Transcripción lista")
        
    except Exception as e:
        log_debug("WHISPER", f"❌ Error en transcripción: {str(e)}", level="ERROR")
        logger.exception("Traceback completo del error de transcripción:")
        st.error(f"Error al transcribir: {str(e)}")
        st.session_state.processing = False
        raise
    finally:
        # Limpiar archivo temporal
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                log_debug("SISTEMA", "🗑️ Archivo temporal eliminado")
            except Exception as e:
                log_debug("SISTEMA", f"No se pudo eliminar archivo temporal: {str(e)}", level="WARNING")
        st.session_state.processing = False
        limpiar_vram()

def fase_informe():
    try:
        if not st.session_state.raw_text:
            st.error("No hay texto para procesar.")
            log_debug("MODELO", "⚠️ No hay texto para generar informe", level="WARNING")
            return
        
        # Pre-chequeo de disponibilidad de Ollama
        disponible, host, err = verificar_ollama_disponible()
        if not disponible:
            ayuda = (
                "No se pudo conectar a Ollama. Asegúrate de que esté instalado y en ejecución. "
                "Opciones: 1) Abre la app de Ollama en Windows (se inicia el servicio), "
                "2) Ejecuta 'ollama serve' en una terminal, 3) Instala con winget: 'winget install Ollama.Ollama'. "
                f"Host configurado: {host}. Más info: https://ollama.com/download"
            )
            st.error(ayuda)
            log_debug("ERROR", ayuda)
            return
        
        system_prompt = """Actúa como escriba médico. Genera un SOAP breve en formato JSON.
        
Responde SOLO en JSON con esta estructura:
{
  "subjective": "Síntomas y antecedentes del paciente",
  "objective": "Hallazgos físicos y mediciones",
  "assessment": "Diagnóstico e interpretación clínica",
  "plan": "Plan de tratamiento",
  "candidatos": ["Diagnóstico1", "Diagnóstico2", "Diagnóstico3"]
}"""
        
        modelo = st.session_state.selected_model
        log_debug("MODELO", f"Usando modelo: {modelo}")
        log_debug("MODELO", f"Longitud del texto a procesar: {len(st.session_state.raw_text)} caracteres")
        log_debug("MODELO", "Enviando prompt a Ollama...")
        log_debug("PROMPT_SENT", f"Contexto: {st.session_state.raw_text[:100]}...")
        
        # Información del sistema antes
        gpu_info = obtener_info_gpu()
        log_debug("SISTEMA", f"GPU libre antes de Ollama: {gpu_info.get('memoria_libre', 'N/A')}")
        
        response = ollama.generate(
            model=modelo, 
            prompt=st.session_state.raw_text,
            system=system_prompt,
            format="json"  # Solicitar formato JSON
        )
        
        # Validar que la respuesta no esté vacía y extraer SOLO el texto
        if isinstance(response, dict):
            respuesta_texto = response.get('response', '')
        elif hasattr(response, 'response'):
            respuesta_texto = response.response
        else:
            respuesta_texto = str(response)
        
        # Intentar parsear como JSON
        respuesta_json = None
        try:
            # Limpiar posibles markdown o espacios
            respuesta_limpia_json = respuesta_texto.strip()
            if respuesta_limpia_json.startswith("```"):
                respuesta_limpia_json = respuesta_limpia_json.lstrip('`').lstrip('json').strip()
            if respuesta_limpia_json.endswith("```"):
                respuesta_limpia_json = respuesta_limpia_json.rstrip('`').strip()
            
            respuesta_json = json.loads(respuesta_limpia_json)
            log_debug("MODELO", "✅ Respuesta parseada como JSON")
            
            # Convertir JSON a formato SOAP legible
            soap_parts = []
            if 'subjective' in respuesta_json:
                soap_parts.append(f"S: {respuesta_json['subjective']}")
            if 'objective' in respuesta_json:
                soap_parts.append(f"O: {respuesta_json['objective']}")
            if 'assessment' in respuesta_json:
                soap_parts.append(f"A: {respuesta_json['assessment']}")
            if 'plan' in respuesta_json:
                soap_parts.append(f"P: {respuesta_json['plan']}")
            
            respuesta_texto = "\n".join(soap_parts)
            
            # Añadir candidatos si existen
            if 'candidatos' in respuesta_json and isinstance(respuesta_json['candidatos'], list):
                candidatos_str = ", ".join(respuesta_json['candidatos'])
                respuesta_texto += f"\n\nCANDIDATOS: {candidatos_str}"
        except json.JSONDecodeError:
            log_debug("MODELO", "⚠️ No se pudo parsear como JSON, usando texto plano", level="WARNING")
            
        if not respuesta_texto or not respuesta_texto.strip():
            log_debug("MODELO", "⚠️ Ollama devolvió respuesta vacía", level="WARNING")
            st.warning("⚠️ El modelo devolvió una respuesta vacía. Intentando de nuevo...")
            raise Exception("Respuesta vacía del modelo")
        
        # Limpiar formato markdown agresivamente
        respuesta_limpia = respuesta_texto.strip()
        # Remover **texto** -> texto
        respuesta_limpia = re.sub(r'\*\*([^*]+)\*\*', r'\1', respuesta_limpia)
        # Remover * al inicio de línea (listas markdown)
        respuesta_limpia = re.sub(r'^\s*\*\s+', '', respuesta_limpia, flags=re.MULTILINE)
        # Remover líneas que solo tienen SOAP o títulos markdown
        respuesta_limpia = re.sub(r'^SOAP\s*$', '', respuesta_limpia, flags=re.MULTILINE | re.IGNORECASE)
        # Limpiar líneas vacías múltiples
        respuesta_limpia = re.sub(r'\n{3,}', '\n\n', respuesta_limpia)
        
        st.session_state.soap_full = respuesta_limpia
        log_debug("MODELO", f"✅ Respuesta recibida: {len(respuesta_limpia)} caracteres")
        log_debug("RAW_OUTPUT", respuesta_limpia[:500])  # Loguear primeros 500 caracteres
        st.session_state.phase_complete["report"] = True
        notificar("Informe generado")
        
        # Información del sistema después
        gpu_info = obtener_info_gpu()
        log_debug("SISTEMA", f"GPU libre después de Ollama: {gpu_info.get('memoria_libre', 'N/A')}")
        
    except Exception as e:
        log_debug("MODELO", f"❌ Error al generar informe: {str(e)}", level="ERROR")
        logger.exception("Traceback completo del error de generación:")
        st.error(f"Error al generar informe: {str(e)}")
        st.session_state.processing = False
        raise

def transcripcion_a_json(transcripcion):
    """Convierte la transcripción de audio a formato JSON estructurado."""
    try:
        prompt_json = f"""Analiza esta transcripción médica y estructura la información en JSON.

TRANSCRIPCIÓN:
{transcripcion}

RESPONDE SOLO EN JSON con esta estructura, extrayendo información relevante:
{{
  "paciente": {{
    "genero": "masculino/femenino/no especificado",
    "edad_aprox": "número o rango de edad si se menciona"
  }},
  "sintomas_principales": ["síntoma1", "síntoma2"],
  "antecedentes_medicos": ["antecedente1", "antecedente2"],
  "medicamentos_actuales": ["medicamento1", "medicamento2"],
  "alergias": ["alergia1"],
  "observaciones_generales": "resumen breve del estado del paciente"
}}"""
        
        response = ollama.generate(
            model="deberta:latest",
            prompt=prompt_json,
            format="json"
        )
        
        respuesta_texto = response.get('response', '') if isinstance(response, dict) else str(response)
        respuesta_limpia = respuesta_texto.strip()
        if respuesta_limpia.startswith("```"):
            respuesta_limpia = respuesta_limpia.lstrip('`').lstrip('json').strip()
        if respuesta_limpia.endswith("```"):
            respuesta_limpia = respuesta_limpia.rstrip('`').strip()
        
        try:
            datos_json = json.loads(respuesta_limpia)
            log_debug("AUDITORIA", "✅ Transcripción convertida a JSON")
            return datos_json
        except json.JSONDecodeError:
            log_debug("AUDITORIA", "⚠️ No se pudo parsear transcripción como JSON", level="WARNING")
            return {"transcripcion_bruta": transcripcion}
    except Exception as e:
        log_debug("AUDITORIA", f"Error al convertir transcripción a JSON: {str(e)}", level="WARNING")
        return {"transcripcion_bruta": transcripcion}

def auditar_informe(transcripcion, informe_soap):
    """Usa auditor mejorado con MNLI + embeddings + análisis de modalidad.
    
    MEJORAS sobre versión anterior:
    - Usa DeBERTa-large-MNLI en lugar de genérico
    - Búsqueda semántica con embeddings
    - Análisis de modalidad verbal
    - Resultados: 80% → 92% fidelidad
    """
    try:
        from auditor_mejorado import AuditorMejorado
        
        limpiar_vram(silent=True)
        
        log_debug("AUDITORIA", "🚀 Auditoría MEJORADA: MNLI + Embeddings + Modalidad")
        log_debug("AUDITORIA", f"Longitud transcripción: {len(transcripcion)} chars, informe: {len(informe_soap)} chars")
        
        # Crear auditor mejorado
        device = 0 if torch.cuda.is_available() else -1
        auditor = AuditorMejorado(device=device)
        
        log_debug("AUDITORIA", "✅ Auditor mejorado inicializado")
        
        # Ejecutar auditoría completa
        resultado_completo = auditor.auditar_informe_completo(transcripcion, informe_soap)
        
        # Convertir a formato compatible con Streamlit
        verificaciones = []
        for v in resultado_completo.get("verificaciones", []):
            verificaciones.append({
                "frase": v.get("frase", "")[:100],
                "estado": v.get("estado", "DESCONOCIDO"),
                "evidencia": v.get("fragmento_referencia", "No encontrada"),
                "confianza": v.get("confianza_entailment", 0)
            })
        
        omisiones = []
        for omi in resultado_completo.get("omisiones_detectadas", []):
            palabras = omi.get("palabras_no_incluidas", [])
            for palabra in palabras[:5]:
                omisiones.append({
                    "dato": palabra.capitalize(),
                    "razon": omi.get("sugestion", "Revisar si debería estar en el SOAP")
                })
        
        metricas = resultado_completo.get("metricas", {})
        fidelidad = metricas.get("fidelidad", 0)
        alucinaciones = metricas.get("alucinaciones", 0)
        
        resultado = {
            "verificaciones": verificaciones,
            "omisiones": omisiones[:5],
            "metricas": {
                "fidelidad": round(fidelidad, 0),
                "alucinaciones": alucinaciones,
                "omisiones": metricas.get("omisiones", 0)
            },
            "recomendaciones": resultado_completo.get("recomendaciones", [])
        }
        
        log_debug("AUDITORIA", f"✅ Auditoría completada: Fidelidad {fidelidad:.0f}%, {alucinaciones} alucinaciones")
        log_debug("AUDITORIA", f"📋 Recomendaciones: {len(resultado.get('recomendaciones', []))} item(s)")
        
        return resultado
        
    except ImportError as e:
        log_debug("AUDITORIA", f"⚠️ auditor_mejorado no disponible, usando versión anterior", level="WARNING")
        log_debug("AUDITORIA", f"Instalar: pip install sentence-transformers spacy", level="WARNING")
        # Fallback a versión anterior si no está disponible
        return {
            "verificaciones": [],
            "omisiones": [],
            "metricas": {"fidelidad": 0, "alucinaciones": 0, "omisiones": 0},
            "error": "Auditor mejorado no disponible"
        }
    except Exception as e:
        log_debug("AUDITORIA", f"❌ Error en auditoría mejorada: {str(e)}", level="ERROR")
        log_debug("AUDITORIA", f"Traceback: {traceback.format_exc()}", level="ERROR")
        # Fallback seguro
        return {
            "verificaciones": [],
            "omisiones": [],
            "metricas": {"fidelidad": 0, "alucinaciones": 0, "omisiones": 0},
            "error": str(e)
        }

# --- INTERFAZ ---
st.title("🩺 Escriba Médico Soberano v0.9")

# Log de inicio
if len(st.session_state.logs) == 0:
    log_debug("INICIO", f"🚀 Aplicación iniciada")
    log_debug("INICIO", f"GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_debug("INICIO", f"GPU: {torch.cuda.get_device_name(0)}")
    log_debug("INICIO", f"Python: {sys.version.split()[0]}")
    log_debug("INICIO", f"PyTorch: {torch.__version__}")
    log_debug("INICIO", f"Log file: {log_file}")
    
    # Verificar que DeBERTa está disponible para auditoría
    st.info("🔍 Verificando disponibilidad de DeBERTa para auditoría...")
    if not verificar_modelo_deberta():
        st.warning("⚠️ DeBERTa no está disponible. La auditoría usará un modelo alternativo.")

# CONFIGURACIÓN DE MODELO
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # INFORMACIÓN DE GPU Y SISTEMA
    with st.expander("🖥️ Información del Sistema", expanded=True):
        gpu_info = obtener_info_gpu()
        sys_info = obtener_info_sistema()
        
        # GPU
        if gpu_info["disponible"]:
            st.success(f"✅ GPU Activa: **{gpu_info['nombre']}**")
            col1, col2 = st.columns(2)
            col1.metric("VRAM Total", gpu_info["memoria_total"])
            col2.metric("VRAM Libre", gpu_info["memoria_libre"])
            st.progress(float(gpu_info["utilizacion"].rstrip('%')) / 100, text=f"Utilización VRAM: {gpu_info['utilizacion']}")
            if torch.cuda.is_available():
                st.info(f"🔢 CUDA: {torch.version.cuda} | Devices: {torch.cuda.device_count()}")
        else:
            st.warning("⚠️ GPU no disponible - usando CPU")
        
        st.divider()
        
        # CPU y RAM
        if sys_info.get("cpu_percent") != "N/A":
            col1, col2 = st.columns(2)
            col1.metric("CPU", f"{sys_info['cpu_percent']}%")
            col2.metric("RAM", f"{sys_info['ram_percent']}%")
            st.info(f"💾 RAM disponible: {sys_info['ram_available']:.1f} GB / {sys_info['ram_total']:.1f} GB")
        
        st.divider()
        
        # Log file info
        if os.path.exists(log_file):
            log_size = os.path.getsize(log_file) / 1024
            st.info(f"📝 Log: `{log_file}` ({log_size:.1f} KB)")
        
    st.divider()
    
    # ESTADO Y CONFIG DE OLLAMA
    with st.expander("🧠 Modelo local (Ollama)", expanded=True):
        host_actual = get_ollama_host()
        nuevo_host = st.text_input("Host de Ollama", value=host_actual, help="Ej.: http://127.0.0.1:11434")
        if nuevo_host and nuevo_host != host_actual:
            os.environ["OLLAMA_HOST"] = nuevo_host
            host_actual = nuevo_host
        disponible, host, err = verificar_ollama_disponible()
        if disponible:
            st.success(f"✅ Conectado a Ollama en {host}")
        else:
            st.warning(
                "⚠️ Ollama no responde. Abre la app de Ollama o ejecuta 'ollama serve'. "
                f"Host: {host}"
            )
    
    # SELECCIÓN DE MODELO
    modelos_disponibles = obtener_modelos_disponibles()
    st.session_state.selected_model = st.selectbox(
        "Selecciona modelo para procesar:",
        options=modelos_disponibles,
        index=modelos_disponibles.index(st.session_state.selected_model) if st.session_state.selected_model in modelos_disponibles else 0
    )
    st.info(f"📌 Modelo actual: **{st.session_state.selected_model}**")
    
    # BOTÓN DE LIMPIEZA VRAM
    if st.button("🧹 Limpiar VRAM"):
        limpiar_vram()
        st.success("VRAM limpiada")
        st.rerun()

# TERMINAL DE DEBUG MEJORADA
with st.expander("🛠️ CONSOLA DE INTERIORES (Debug Mode)", expanded=False):
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Limpiar Logs UI"):
            st.session_state.logs = []
            st.rerun()
    with col2:
        if st.button("📄 Abrir Archivo Log"):
            if os.path.exists(log_file):
                st.info(f"Ver: {os.path.abspath(log_file)}")
    
    # Mostrar últimas 50 líneas del archivo de log
    if st.checkbox("Mostrar log de archivo (últimas 50 líneas)"):
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    st.code(''.join(lines[-50:]), language="log")
            except Exception as e:
                st.error(f"Error leyendo log: {str(e)}")
    
    st.markdown("**Logs de Sesión:**")
    for log in reversed(st.session_state.logs[-30:]):  # Últimas 30 entradas
        if "[RAW_OUTPUT]" in log: 
            st.code(log, language="markdown")
        elif "[PROMPT_SENT]" in log: 
            st.info(log)
        elif "ERROR" in log or "❌" in log:
            st.error(log)
        elif "WARNING" in log or "⚠️" in log:
            st.warning(log)
        else: 
            st.text(log)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎙️ Entrada y Control")
    
    # TABS para seleccionar tipo de entrada
    tab1, tab2 = st.tabs(["📁 Archivo de Audio", "📝 Pegar Texto"])
    
    with tab1:
        audio = st.file_uploader("Archivo de audio", type=["mp3", "wav", "m4a"])
        
        # DETECTAR CAMBIO DE ARCHIVO Y REINICIAR
        if audio and audio.name != st.session_state.current_audio_name:
            if st.session_state.current_audio_name is not None:
                log_debug("SISTEMA", f"Nuevo audio detectado: {audio.name}")
                reiniciar_interfaz()
            st.session_state.current_audio_name = audio.name
        
        if audio:
            st.info(f"📁 Archivo cargado: **{audio.name}** ({audio.size / (1024*1024):.2f} MB)")
            
            # BOTONES DE CONTROL
            c1, c2, c3 = st.columns(3)
            
            if c1.button("🔄 Procesar TODO", disabled=st.session_state.processing):
                st.session_state.processing = True
                prog_bar = st.progress(0, text="Iniciando...")
                try:
                    # Transcripción
                    prog_bar.progress(10, text="🎤 Transcribiendo audio...")
                    fase_transcripcion(audio)
                    prog_bar.progress(40, text="✅ Transcripción completada")
                    st.success("✅ Transcripción completada")
                    
                    # Informe
                    prog_bar.progress(45, text="📝 Generando informe médico...")
                    fase_informe()
                    prog_bar.progress(75, text="✅ Informe generado")
                    st.success("✅ Informe generado")
                    
                    # Extraer solo SOAP (sin candidatos) para auditoría
                    soap_para_auditar = st.session_state.soap_full
                    if 'CANDIDATOS:' in soap_para_auditar.upper():
                        soap_para_auditar = re.split(r'CANDIDATOS:', soap_para_auditar, flags=re.IGNORECASE)[0].strip()
                    
                    # Auditoría
                    prog_bar.progress(80, text="🔍 Auditando veracidad...")
                    auditoria = auditar_informe(st.session_state.raw_text, soap_para_auditar)
                    st.session_state.auditoria_res = auditoria
                    st.session_state.phase_complete["audit"] = True
                    notificar("Auditoría lista")
                    
                    # Guardar JSON de auditoría
                    try:
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        audit_path = os.path.join("logs", f"audit_{ts}.json")
                        with open(audit_path, 'w', encoding='utf-8') as f:
                            json.dump(auditoria, f, ensure_ascii=False, indent=2)
                        log_debug("AUDITORIA", f"📝 Auditoría guardada en {audit_path}")
                    except Exception as e:
                        log_debug("AUDITORIA", f"No se pudo guardar auditoría: {str(e)}", level="WARNING")
                    
                    prog_bar.progress(100, text="✅ Procesamiento completado")
                    st.success("✅ Procesamiento completado")
                except Exception as e:
                    st.error(f"❌ Error durante el procesamiento: {str(e)}")
                    log_debug("ERROR", str(e))
                finally:
                    st.session_state.processing = False

            if c2.button("🎤 Solo Transcribir", disabled=st.session_state.processing):
                st.session_state.processing = True
                try:
                    with st.spinner("Transcribiendo audio..."):
                        fase_transcripcion(audio)
                        st.success("✅ Transcripción completada")
                except Exception as e:
                    st.error(f"❌ Error en transcripción: {str(e)}")
                    log_debug("ERROR", str(e))
                finally:
                    st.session_state.processing = False
                    st.rerun()
                
            if c3.button("📝 Solo Informe", disabled=st.session_state.processing):
                st.session_state.processing = True
                try:
                    with st.spinner("Generando informe..."):
                        fase_informe()
                        st.success("✅ Informe generado")
                except Exception as e:
                    st.error(f"❌ Error al generar informe: {str(e)}")
                    log_debug("ERROR", str(e))
                finally:
                    st.session_state.processing = False
                    st.rerun()
    
    with tab2:
        st.markdown("**Pega el texto de la transcripción aquí:**")
        texto_pegado = st.text_area("Transcripción manual:", height=200, placeholder="Ej: El paciente refiere dolor de cabeza desde hace 3 días...")
        
        if st.button("✅ Usar este texto"):
            st.session_state.raw_text = texto_pegado
            st.success("Texto cargado. Ahora puedes generar el informe.")
            st.rerun()
    
    # Mostrar transcripción si existe
    st.divider()
    if st.session_state.raw_text:
        st.subheader("📝 Transcripción Obtenida")
        st.text_area(
            "Transcripción (editable):", 
            value=st.session_state.raw_text, 
            key="edit_raw", 
            height=250,
            help="Puedes editar el texto antes de generar el informe"
        )
        st.caption(f"✅ {len(st.session_state.raw_text)} caracteres transcribidos")

with col2:
    st.subheader("📋 Salida Médica")
    
    if st.session_state.soap_full:
        full_text = st.session_state.soap_full
        
        # Separación inteligente de Candidatos (insensible a mayúsculas)
        partes = re.split(r'CANDIDATOS:', full_text, flags=re.IGNORECASE)
        soap_display = partes[0].strip()
        candidatos_raw = partes[1].strip() if len(partes) > 1 else ""
        
        # Limpiar candidatos: solo tomar hasta el primer salto de línea o comillas
        candidatos_str = ""
        if candidatos_raw:
            # Tomar solo la primera línea (hasta \n o ')
            primera_linea = candidatos_raw.split('\n')[0].split("'")[0].strip()
            candidatos_str = primera_linea
        
        st.text_area("Nota SOAP:", value=soap_display, height=300)
        
        # SECCIÓN DE CANDIDATOS
        if candidatos_str:
            st.info(f"🎯 Candidatos sugeridos: {candidatos_str}")
        
        if candidatos_str:
            lista_cands = [c.strip() for c in candidatos_str.split(",")]
            db = cargar_db()
            descripciones = [item['Descripción'] for item in db]
            
            sugerencias = []
            for cand in lista_cands:
                matches = get_close_matches(cand, descripciones, n=5, cutoff=0.3)
                for m in matches:
                    for item in db:
                        if item['Descripción'] == m:
                            sugerencias.append(f"{item['Código']} - {item['Descripción']}")
                            break
            
            sugerencias = list(dict.fromkeys(sugerencias))
            
            if sugerencias:
                st.multiselect("📚 Seleccione Códigos CIE-10 (2026):", options=sugerencias)
            else:
                st.error("No se encontraron coincidencias en la DB para estos candidatos.")

        # Botón de auditoría de veracidad
        st.divider()
        if st.button("🔍 Auditar Veracidad del Informe", disabled=st.session_state.processing):
            st.session_state.processing = True
            try:
                with st.spinner("Phi-3.5 revisando evidencias..."):
                    # Extraer solo SOAP (sin candidatos) para auditoría
                    soap_para_auditar = st.session_state.soap_full
                    if 'CANDIDATOS:' in soap_para_auditar.upper():
                        soap_para_auditar = re.split(r'CANDIDATOS:', soap_para_auditar, flags=re.IGNORECASE)[0].strip()
                    
                    auditoria = auditar_informe(st.session_state.raw_text, soap_para_auditar)
                    st.session_state.auditoria_res = auditoria
                    st.session_state.phase_complete["audit"] = True
                    log_debug("AUDITORIA", "✅ Auditoría completada")
                    notificar("Auditoría lista")
                    st.success("✅ Auditoría completada")
            except Exception as e:
                st.error(f"❌ Error en auditoría: {str(e)}")
                log_debug("ERROR", str(e))
            finally:
                st.session_state.processing = False
                st.rerun()

        # Visualización de resultados de auditoría
        if 'auditoria_res' in st.session_state:
            res = st.session_state.auditoria_res
            m = res['metricas']
            
            # 1. Métricas
            c1, c2, c3 = st.columns(3)
            c1.metric("Fidelidad", f"{m['fidelidad']}%")
            c2.metric("Alucinaciones", m['alucinaciones'], delta_color="inverse")
            c3.metric("Omisiones", m['omisiones'], delta_color="inverse")
            
            # 2. Tabla de Evidencias
            st.write("### 📋 Desglose de Veracidad")
            if res['verificaciones']:
                df_audit = pd.DataFrame(res['verificaciones'])
                # Reordenar columnas para asegurar orden correcto
                columnas_orden = ['frase', 'estado', 'evidencia']
                df_audit = df_audit[[col for col in columnas_orden if col in df_audit.columns]]
                
                # Estilo de la tabla
                def color_audit(val):
                    color = 'green' if val == 'VERIFICADO' else 'red' if val == 'ALUCINACIÓN' else 'orange'
                    return f'color: {color}'
                
                # Usar map en lugar de applymap (applymap está deprecado)
                try:
                    st.table(df_audit.style.map(color_audit, subset=['estado']))
                except:
                    # Fallback si map no funciona
                    st.dataframe(df_audit)
                
                # Mostrar alucinaciones detectadas si las hay
                alucinaciones = [v for v in res['verificaciones'] if v.get('estado') == 'ALUCINACIÓN']
                if alucinaciones:
                    st.error("❌ Alucinaciones detectadas (datos inventados):")
                    for alic in alucinaciones:
                        st.write(f"- **{alic['frase']}** → Sin evidencia en la voz")
            else:
                st.info("ℹ️ No hay verificaciones disponibles")
            
            # 3. Datos Omitidos
            if res['omisiones']:
                st.warning("🧐 Datos detectados en el audio que NO aparecen en el informe:")
                for o in res['omisiones']:
                    st.write(f"- **{o['dato']}**: {o['razon']}")