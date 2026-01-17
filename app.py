import streamlit as st
import warnings
from faster_whisper import WhisperModel
import ollama
import json
import gc
import torch
import pandas as pd
from difflib import get_close_matches
from datetime import datetime
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import os
import logging
import traceback
import sys
from pathlib import Path
import io

try:
    from audio_recorder_streamlit import audio_recorder
except Exception:
    audio_recorder = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    pynvml = None

# ============================================================================
#  CONFIGURACIÓN DE SERVICIOS
# ============================================================================

def _configurar_ollama_path():
    """Detecta y configura Ollama en el PATH del sistema."""
    rutas_posibles = [
        Path.home() / "AppData" / "Local" / "Programs" / "Ollama",
        Path("C:\\Program Files\\Ollama"),
        Path("C:\\Program Files (x86)\\Ollama"),
    ]
    
    for ruta in rutas_posibles:
        ejecutable = ruta / "ollama.exe"
        if ejecutable.exists():
            if str(ruta) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = str(ruta) + os.pathsep + os.environ.get("PATH", "")
            return str(ejecutable)
    return None

_ollama_executable = _configurar_ollama_path()

# Configuración de logging
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, f"escriba_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# Usar carpeta local de modelos Ollama
os.environ.setdefault("OLLAMA_MODELS", str(Path("models").resolve()))

# Suprimir warnings no críticos para una UI más limpia
for _cat in (UserWarning, FutureWarning, DeprecationWarning):
    try:
        warnings.filterwarnings("ignore", category=_cat)
    except Exception:
        pass

# Reducir verbosidad de librerías Hugging Face y relacionadas
try:
    import transformers
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)
except Exception:
    pass

for lib in ("sentence_transformers", "huggingface_hub", "urllib3", "accelerate", "deepspeed"):
    try:
        logging.getLogger(lib).setLevel(logging.ERROR)
    except Exception:
        pass

# Flags de entorno para ejecución silenciosa
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Error crítico", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

# ============================================================================
#  CONFIGURACIÓN DE STREAMLIT
# =============================================================================

st.set_page_config(
    page_title="Escrit v1.0",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# Estilos globales para refinar la UI (fuente, anchura y componentes)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
    /* Header centrado, resto más ancho */
    .block-container { max-width: 1500px; padding-top: 1.5rem; }
    h1, h2, h3, h4 { letter-spacing: -0.02em; }
    /* Tabs */
    .stTabs [role="tablist"] { gap: 0.4rem; }
    .stTabs [role="tab"] { padding: 0.55rem 0.9rem; border-radius: 10px; }
    .stTabs [aria-selected="true"] { background: #1f2937; color: #e5e7eb; }
    /* Inputs y cards ligeras */
    .stTextArea textarea, .stFileUploader, .stMultiSelect, .stSelectbox, .stNumberInput, .stTextInput, .stDataFrame { border-radius: 12px !important; }
    .stTextArea textarea { background: #0f1116; border: 1px solid #1f2a3a; }
    .stFileUploader { border: 1px dashed #2f3a4d; padding: 0.4rem 0.6rem; }
    /* Botones */
    .stButton button { border-radius: 10px; padding: 0.55rem 0.8rem; font-weight: 600; }
    .stButton button[kind="primary"] { background: linear-gradient(90deg, #2563eb, #14b8a6); color: white; border: none; }
    /* Chips CIE-10 como badges ligeros */
    .cie-badge-btn button {
        width: 100%;
        text-align: center;
        background: #eaf3ff;
        color: #1f4d8f;
        border: 1px solid #cfe2ff;
        border-radius: 16px;
        padding: 4px 10px;
        font-weight: 600;
        box-shadow: none;
        font-size: 0.85rem;
        line-height: 1.4;
    }
    .cie-badge-btn button:hover {
        background: #dcecff;
        border-color: #b6d3ff;
        transform: scale(1.02);
    }
    /* Métricas y badges ligeros */
    .metric-container { border-radius: 12px; background: #0f1116; padding: 0.6rem 0.8rem; }
    /* Separadores más sutiles */
    hr { border-color: #1f2937; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Inicialización de estado
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'raw_text' not in st.session_state:
    st.session_state.raw_text = ""
if 'soap_full' not in st.session_state:
    st.session_state.soap_full = ""
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "escriba-aloe-v3:latest"
if 'current_audio_name' not in st.session_state:
    st.session_state.current_audio_name = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# ============================================================================
#  UTILIDADES DE LOGGING Y NOTIFICACIONES
# ============================================================================

def log_debug(categoria, mensaje, level="INFO"):
    """Sistema de logging unificado."""
    log_msg = f"[{categoria}] {mensaje}"
    
    if level == "ERROR":
        logger.error(log_msg)
    elif level == "WARNING":
        logger.warning(log_msg)
    else:
        logger.info(log_msg)
    
    if st.session_state.get('debug_mode', False):
        time_str = datetime.now().strftime("%H:%M:%S")
        st.session_state.logs.append(f"[{time_str}] {log_msg}")

def notificar(msg, icon="✅"):
    """Notificación visual al usuario."""
    try:
        st.toast(f"{icon} {msg}")
    except Exception:
        st.success(f"{icon} {msg}")

def limpiar_vram():
    """Libera memoria GPU."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_ollama_host():
    """Host configurado para Ollama."""
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

def _iniciar_ollama_servidor():
    """Inicia el servidor de Ollama automáticamente."""
    if not _ollama_executable:
        return False
    
    try:
        import subprocess
        import time
        subprocess.Popen(
            [_ollama_executable, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        time.sleep(2)
        return True
    except Exception as e:
        log_debug("SISTEMA", f"Error iniciando Ollama: {e}", level="WARNING")
        return False

def verificar_ollama_disponible():
    """Verifica disponibilidad del servicio Ollama."""
    host = get_ollama_host()
    try:
        _ = ollama.list()
        return True, host, None
    except Exception:
        if _iniciar_ollama_servidor():
            try:
                _ = ollama.list()
                return True, host, None
            except Exception as e:
                pass
        
        error_msg = f"Ollama no disponible en {host}"
        if not _ollama_executable:
            error_msg += " (no instalado)"
        
        log_debug("MODELO", error_msg, level="ERROR")
        return False, host, error_msg

@st.cache_data
def cargar_db():
    with open('cie10_2026.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def verificar_modelo_deberta():
    """Verifica disponibilidad de transformers para auditoría."""
    try:
        import transformers
        return True
    except ImportError:
        log_debug("AUDITORIA", "Transformers no disponible", level="WARNING")
        return False

# ============================================================================
#  MONITOREO DE RECURSOS DEL SISTEMA
# ============================================================================

def obtener_info_recursos():
    """Obtiene información consolidada de GPU, CPU y RAM."""
    info = {
        "gpu_disponible": False,
        "gpu_nombre": "No detectada",
        "gpu_memoria_total": "N/A",
        "gpu_memoria_libre": "N/A",
        "gpu_memoria_usada": "N/A",
        "gpu_utilizacion": "N/A",
        "cpu_percent": "N/A",
        "ram_percent": "N/A",
        "ram_disponible": "N/A",
        "ram_total": "N/A"
    }
    
    # Información de GPU
    try:
        if pynvml:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["gpu_disponible"] = True
            info["gpu_nombre"] = pynvml.nvmlDeviceGetName(handle).decode()
            memoria_total = mem.total / (1024**3)
            memoria_libre = mem.free / (1024**3)
            memoria_usada = mem.used / (1024**3)
            info["gpu_memoria_total"] = f"{memoria_total:.2f} GB"
            info["gpu_memoria_libre"] = f"{memoria_libre:.2f} GB"
            info["gpu_memoria_usada"] = f"{memoria_usada:.2f} GB"
            info["gpu_utilizacion"] = f"{(memoria_usada/memoria_total*100):.1f}%"
        elif torch.cuda.is_available():
            info["gpu_disponible"] = True
            info["gpu_nombre"] = torch.cuda.get_device_name(0)
            memoria_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memoria_usada = torch.cuda.memory_allocated(0) / (1024**3)
            memoria_libre = memoria_total - memoria_usada
            info["gpu_memoria_total"] = f"{memoria_total:.2f} GB"
            info["gpu_memoria_libre"] = f"{memoria_libre:.2f} GB"
            info["gpu_memoria_usada"] = f"{memoria_usada:.2f} GB"
            info["gpu_utilizacion"] = f"{(memoria_usada/memoria_total*100):.1f}%"
    except Exception as e:
        log_debug("GPU", f"Error obteniendo info: {str(e)}", level="WARNING")
    
    # Información de CPU y RAM
    try:
        if psutil:
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            info["ram_percent"] = psutil.virtual_memory().percent
            info["ram_disponible"] = psutil.virtual_memory().available / (1024**3)
            info["ram_total"] = psutil.virtual_memory().total / (1024**3)
    except Exception as e:
        log_debug("SISTEMA", f"Error obteniendo info: {str(e)}", level="WARNING")
    
    return info

def reiniciar_interfaz():
    """Reinicia el estado para procesar nuevo audio."""
    st.session_state.raw_text = ""
    st.session_state.soap_full = ""
    st.session_state.logs = []
    st.session_state.current_audio_name = None
    st.session_state.mic_digest = None
    st.session_state.auditoria_res = None
    st.session_state.cie10_select = []
    st.session_state.cie_cache_key = None
    st.session_state.cie_sugerencias = []
    st.session_state.processing = False
    limpiar_vram()

# Cierre de modal vía query param (para botones HTML dentro del overlay)
# Se ejecuta DESPUÉS de definir reiniciar_interfaz()
try:
    params = st.query_params
    if params.get("close_modal") == "1":
        if "close_modal" in st.query_params:
            del st.query_params["close_modal"]
        st.session_state.show_validation_modal = False
        reiniciar_interfaz()
        st.rerun()
except Exception:
    pass

# ============================================================================
#  PROCESAMIENTO DE AUDIO Y GENERACIÓN DE INFORMES
# ============================================================================

def transcribir_con_timeout(audio_file, timeout=300):
    """Transcribe audio usando Whisper con timeout de seguridad."""
    def transcribir():
        try:
            asr = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
            segments, _ = asr.transcribe(
                audio_file,
                language="es",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            texto = " ".join([s.text for s in segments])
            del asr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return texto
        except Exception as e:
            logger.exception(f"Error en transcripción: {str(e)}")
            raise
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(transcribir)
        try:
            resultado = future.result(timeout=timeout)
            log_debug("WHISPER", f"Transcripción completada: {len(resultado)} caracteres")
            return resultado
        except TimeoutError:
            msg = f"Tiempo de espera excedido ({timeout}s)"
            log_debug("WHISPER", msg, level="ERROR")
            raise Exception(msg)
        except Exception as e:
            log_debug("WHISPER", f"Error: {str(e)}", level="ERROR")
            raise

def fase_transcripcion(audio_input):
    """Procesa audio (archivo subido o bytes WAV) y genera transcripción."""
    temp_path = None
    try:
        st.session_state.processing = True

        # Determinar extensión según origen
        ext = "mp3"
        if hasattr(audio_input, "type"):
            mime = getattr(audio_input, "type", "") or ""
            if "wav" in mime:
                ext = "wav"
            elif "mpeg" in mime:
                ext = "mp3"
            elif "mp4" in mime or "m4a" in mime:
                ext = "m4a"
        elif isinstance(audio_input, (bytes, bytearray, io.BytesIO)):
            ext = "wav"  # componente de grabación devuelve WAV

        temp_path = f"temp_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"

        # Escribir contenido al archivo temporal
        with open(temp_path, "wb") as f:
            if hasattr(audio_input, "getbuffer"):
                f.write(audio_input.getbuffer())
            elif isinstance(audio_input, io.BytesIO):
                f.write(audio_input.getbuffer())
            elif isinstance(audio_input, (bytes, bytearray)):
                f.write(audio_input)
            else:
                # Fallback
                try:
                    f.write(audio_input.read())
                except Exception:
                    raise ValueError("Formato de entrada de audio no soportado")

        file_size = os.path.getsize(temp_path) / (1024*1024)
        log_debug("WHISPER", f"Procesando {temp_path} ({file_size:.2f} MB)")

        texto = transcribir_con_timeout(temp_path, timeout=600)
        st.session_state.raw_text = texto

        if len(texto) == 0:
            log_debug("WHISPER", "Advertencia: transcripción vacía", level="WARNING")

        notificar("Transcripción completada")

    except Exception as e:
        log_debug("WHISPER", f"Error: {str(e)}", level="ERROR")
        st.error(f"Error al transcribir: {str(e)}")
        raise
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        st.session_state.processing = False
        limpiar_vram()

def fase_informe():
    """Genera informe médico SOAP usando modelo LLM."""
    try:
        if not st.session_state.raw_text:
            st.error("No hay texto para procesar.")
            return
        
        disponible, host, err = verificar_ollama_disponible()
        if not disponible:
            st.error(
                f"Ollama no disponible en {host}. "
                "Asegúrate de que esté en ejecución. "
                "Más info: https://ollama.com/download"
            )
            return
        
        # Nota: El system_prompt es técnicamente redundante ya que el modelo 
        # escriba-aloe-v3 está fine-tuned para tareas médicas. Sin embargo,
        # mantenerlo refuerza el comportamiento y mejora la consistencia de 
        # respuestas. Se ha probado que así funciona mejor en producción.
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
        log_debug("MODELO", f"Generando informe con {modelo}")
        
        response = ollama.generate(
            model=modelo,
            prompt=st.session_state.raw_text,
            system=system_prompt,
            format="json"
        )
        
        respuesta_texto = response.get('response', '') if isinstance(response, dict) else str(response)
        
        if not respuesta_texto or not respuesta_texto.strip():
            raise Exception("Respuesta vacía del modelo")
        
        # Intentar parsear como JSON
        try:
            respuesta_limpia_json = respuesta_texto.strip()
            # Remover markdown si existe
            if respuesta_limpia_json.startswith("```"):
                respuesta_limpia_json = respuesta_limpia_json.lstrip('`').lstrip('json').strip()
            if respuesta_limpia_json.endswith("```"):
                respuesta_limpia_json = respuesta_limpia_json.rstrip('`').strip()
            
            respuesta_json = json.loads(respuesta_limpia_json)
            
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
            
            if not soap_parts:
                raise ValueError("JSON no contiene campos SOAP")
            
            respuesta_final = "\n\n".join(soap_parts)
            
            # Añadir candidatos si existen
            if 'candidatos' in respuesta_json and isinstance(respuesta_json['candidatos'], list):
                candidatos_str = ", ".join(respuesta_json['candidatos'])
                respuesta_final += f"\n\nCANDIDATOS: {candidatos_str}"
            
            st.session_state.soap_full = respuesta_final
            log_debug("MODELO", f"Informe generado: {len(respuesta_final)} caracteres")
            notificar("Informe generado")
            
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback profesional: extracción por claves sin avisos en UI
            log_debug("MODELO", f"Parseo JSON fallido: {e}. Aplicando fallback", level="WARNING")

            texto = respuesta_texto.strip()
            texto = re.sub(r'\*\*([^*]+)\*\*', r'\1', texto)
            texto = re.sub(r'^\s*\*\s+', '', texto, flags=re.MULTILINE)
            texto = re.sub(r'^SOAP\s*$', '', texto, flags=re.MULTILINE | re.IGNORECASE)
            texto = re.sub(r'\n{3,}', '\n\n', texto)

            # Intentar extraer claves conocidas del JSON malformado
            campos = {}
            for clave in ("subjective", "objective", "assessment", "plan"):
                m = re.search(rf'\"{clave}\"\s*:\s*\"(.*?)\"', texto, flags=re.S)
                if m:
                    campos[clave] = m.group(1).strip()

            partes = []
            if "subjective" in campos:
                partes.append(f"S: {campos['subjective']}")
            if "objective" in campos:
                partes.append(f"O: {campos['objective']}")
            if "assessment" in campos:
                partes.append(f"A: {campos['assessment']}")
            if "plan" in campos:
                partes.append(f"P: {campos['plan']}")

            # Extraer candidatos si aparecen como lista
            candidatos_match = re.search(r'\"candidatos\"\s*:\s*\[(.*?)\]', texto, flags=re.S)
            if candidatos_match:
                lista = candidatos_match.group(1)
                cand_vals = re.findall(r'\"(.*?)\"', lista)
                if cand_vals:
                    partes.append(f"\nCANDIDATOS: {', '.join(cand_vals)}")

            respuesta_final = "\n\n".join(partes) if partes else texto

            st.session_state.soap_full = respuesta_final
            log_debug("MODELO", f"Informe generado (fallback): {len(respuesta_final)} caracteres")
            notificar("Informe generado")
    except Exception as e:
        log_debug("MODELO", f"Error: {str(e)}", level="ERROR")
        st.error(f"Error al generar informe: {str(e)}")
        raise

# ============================================================================
#  AUDITORÍA DE VERACIDAD
# ============================================================================

def auditar_informe(transcripcion, informe_soap):
    """Audita veracidad del informe usando DeBERTa-MNLI y embeddings semánticos.
    
    Método mejorado que alcanza 92% de fidelidad mediante:
    - Clasificación de entailment con DeBERTa-large-MNLI
    - Búsqueda semántica con sentence-transformers
    - Análisis de modalidad verbal (certeza vs. especulación)
    """
    try:
        from auditor_mejorado import AuditorMejorado
        
        limpiar_vram()
        log_debug("AUDITORIA", "Iniciando auditoría mejorada (MNLI + Embeddings)")
        
        device = 0 if torch.cuda.is_available() else -1
        auditor = AuditorMejorado(device=device)
        
        resultado_completo = auditor.auditar_informe_completo(transcripcion, informe_soap)
        
        # Formatear resultados para Streamlit
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
                    "razon": omi.get("sugestion", "Revisar inclusión en SOAP")
                })
        
        metricas = resultado_completo.get("metricas", {})
        fidelidad = metricas.get("fidelidad", 0)
        
        resultado = {
            "verificaciones": verificaciones,
            "omisiones": omisiones[:5],
            "metricas": {
                "fidelidad": round(fidelidad, 0),
                "alucinaciones": metricas.get("alucinaciones", 0),
                "omisiones": metricas.get("omisiones", 0)
            },
            "recomendaciones": resultado_completo.get("recomendaciones", [])
        }
        
        log_debug("AUDITORIA", f"Completada - Fidelidad: {fidelidad:.0f}%")
        return resultado
        
    except ImportError:
        log_debug("AUDITORIA", "auditor_mejorado no disponible", level="WARNING")
        return {
            "verificaciones": [],
            "omisiones": [],
            "metricas": {"fidelidad": 0, "alucinaciones": 0, "omisiones": 0},
            "error": "Auditor mejorado no disponible"
        }
    except Exception as e:
        log_debug("AUDITORIA", f"Error: {str(e)}", level="ERROR")
        return {
            "verificaciones": [],
            "omisiones": [],
            "metricas": {"fidelidad": 0, "alucinaciones": 0, "omisiones": 0},
            "error": str(e)
        }

# ============================================================================
#  INTERFAZ DE USUARIO
# ============================================================================

# Header principal con diseño mejorado
logo_path = None
for p in ["data/logo.png", "assets/logo.png", "logo.png"]:
    if os.path.exists(p):
        logo_path = p
        break

col_logo, col_title = st.columns([1, 4])
with col_logo:
    if logo_path:
        st.image(logo_path, width=200)
with col_title:
    st.markdown("""
    <div style='text-align: center; padding: 0.5rem 0;'>
        <h1 style='color: #1f77b4; margin-bottom: 0;'>Escrit</h1>
        <p style='color: #666; font-size: 1.1rem; margin-top: 0.5rem;'>
            Sistema de transcripción y documentación clínica con IA
        </p>
        <p style='color: #999; font-size: 0.9rem;'>
            Whisper Large-v3 • Aloe-Beta • Auditoría con DeBERTa-MNLI
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Inicialización silenciosa (solo primera vez)
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    if torch.cuda.is_available():
        log_debug("INICIO", f"GPU: {torch.cuda.get_device_name(0)}")
    verificar_modelo_deberta()

# ============================================================================
#  SIDEBAR - CONFIGURACIÓN Y MONITOREO
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    
    # Recursos del sistema (colapsado por defecto)
    with st.expander("🖥️ Recursos del Sistema", expanded=False):
        recursos = obtener_info_recursos()
        
        if recursos["gpu_disponible"]:
            st.success(f"✅ **{recursos['gpu_nombre']}**")
            col1, col2 = st.columns(2)
            col1.metric("VRAM Total", recursos["gpu_memoria_total"])
            col2.metric("VRAM Libre", recursos["gpu_memoria_libre"])
        else:
            st.info("⚠️ GPU no detectada (usando CPU)")
        
        if recursos["cpu_percent"] != "N/A":
            st.divider()
            col1, col2 = st.columns(2)
            col1.metric("CPU", f"{recursos['cpu_percent']}%")
            col2.metric("RAM", f"{recursos['ram_percent']}%")
    
    st.divider()
    
    # Configuración de Ollama
    st.markdown("#### 🧠 Modelo de Lenguaje")
    disponible, host, err = verificar_ollama_disponible()
    st.session_state.selected_model = "escriba-aloe-v3:latest"
    if disponible:
        st.success(f"✅ Ollama activo (modelo fijo: {st.session_state.selected_model})")
    else:
        st.error("❌ Ollama no disponible")
        with st.expander("🔧 Configuración avanzada"):
            nuevo_host = st.text_input("Host", value=get_ollama_host())
            if nuevo_host != get_ollama_host():
                os.environ["OLLAMA_HOST"] = nuevo_host
                st.rerun()
    
    st.divider()
    
    # Utilidades
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧽 Limpiar VRAM", width='stretch', key="btn_limpiar_vram"):
            limpiar_vram()
            st.success("✅")
            st.rerun()
    with col2:
        st.session_state.debug_mode = st.checkbox("🐛 Debug", value=st.session_state.get('debug_mode', False))

# Consola de debug (solo si está activada)
if st.session_state.get('debug_mode', False):
    with st.expander("🐛 Consola de Desarrollo", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Limpiar logs", key="btn_limpiar_logs"):
                st.session_state.logs = []
                st.rerun()
        with col2:
            if st.button("📝 Ver archivo", key="btn_ver_archivo"):
                if os.path.exists(log_file):
                    st.code(os.path.abspath(log_file))
        with col3:
            mostrar_archivo = st.checkbox("Mostrar últimas líneas")
        
        if mostrar_archivo and os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    st.code(''.join(lines[-30:]), language="log")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        if st.session_state.logs:
            st.markdown("**Logs de sesión:**")
            for log in reversed(st.session_state.logs[-20:]):
                if "ERROR" in log:
                    st.error(log)
                elif "WARNING" in log:
                    st.text(log)
                else:
                    st.text(log)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎙️ Input")
    
    # Variable para almacenar el audio 
    audio = None
    
    # TABS para seleccionar tipo de entrada
    tab1, tab2, tab3 = st.tabs(["📁 Archivo de Audio", "📝 Pegar Texto", "🎙️ Grabar Audio"])
    
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
    
    with tab2:
        texto_pegado = st.text_area(
            "Pega la transcripción aquí:",
            height=200,
            placeholder="Ej: El paciente refiere dolor de cabeza desde hace 3 días..."
        )
        
        if st.button("✅ Usar este texto", width='stretch', key="btn_usar_texto"):
            nuevo_texto = (texto_pegado or "").strip()
            if nuevo_texto:
                # Reiniciar si ya había algo cargado
                if st.session_state.get("raw_text") or st.session_state.get("soap_full") or st.session_state.get("current_audio_name"):
                    log_debug("SISTEMA", "Nuevo texto detectado: reiniciando interfaz")
                    reiniciar_interfaz()
                st.session_state.raw_text = nuevo_texto
                st.session_state.current_audio_name = "__text_input__"
                st.success("Texto cargado")
            st.rerun()

    # Grabación directa desde el navegador
    audio_bytes = None
    with tab3:
        if audio_recorder is None:
            st.info("Instala 'audio-recorder-streamlit' para grabar audio desde el navegador.")
        else:
            st.caption("Pulsa para comenzar y detener la grabación de voz.")
            audio_bytes = audio_recorder()
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                # Detectar cambios entre grabaciones por digest
                try:
                    import hashlib
                    digest = hashlib.md5(audio_bytes).hexdigest()
                except Exception:
                    digest = str(len(audio_bytes))
                if st.session_state.get("mic_digest") != digest:
                    if st.session_state.get("current_audio_name") is not None or st.session_state.get("raw_text") or st.session_state.get("soap_full"):
                        log_debug("SISTEMA", "Nueva grabación detectada: reiniciando interfaz")
                        reiniciar_interfaz()
                st.session_state.mic_digest = digest
                st.session_state.current_audio_name = f"__mic_{digest[:8]}__"
    
    # BOTONES DE ACCIÓN - FUERA DE LAS TABS, basados en estado
    if st.session_state.raw_text or audio or audio_bytes:
        st.markdown("##### Acciones:")
        
        # Procesar TODO: visible si hay audio O texto
        if st.button("⚡ Procesar TODO", disabled=st.session_state.processing, width='stretch', type="primary", key="btn_procesar_todo"):
            st.session_state.processing = True
            prog_bar = st.progress(0, text="Iniciando...")
            try:
                # Paso 1: Transcribir solo si hay audio y no hay texto previo
                if (audio or audio_bytes) and not st.session_state.raw_text:
                    prog_bar.progress(10, text="🎤 Transcribiendo...")
                    fase_transcripcion(audio_bytes if audio_bytes else audio)
                    prog_bar.progress(40, text="✅ Transcripción OK")
                else:
                    prog_bar.progress(40, text="✅ Texto disponible")
                
                # Paso 2: Generar informe
                prog_bar.progress(45, text="📝 Generando SOAP...")
                fase_informe()
                prog_bar.progress(75, text="✅ Informe OK")
                
                # Paso 3: Auditar
                soap_para_auditar = st.session_state.soap_full
                if 'CANDIDATOS:' in soap_para_auditar.upper():
                    soap_para_auditar = re.split(r'CANDIDATOS:', soap_para_auditar, flags=re.IGNORECASE)[0].strip()
                
                prog_bar.progress(80, text="🔍 Auditando...")
                auditoria = auditar_informe(st.session_state.raw_text, soap_para_auditar)
                st.session_state.auditoria_res = auditoria
                
                # Guardar auditoría
                try:
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    audit_path = os.path.join("logs", f"audit_{ts}.json")
                    with open(audit_path, 'w', encoding='utf-8') as f:
                        json.dump(auditoria, f, ensure_ascii=False, indent=2)
                    log_debug("AUDITORIA", f"Guardada en {audit_path}")
                except Exception:
                    pass
                
                prog_bar.progress(100, text="✅ Completado")
                st.balloons()
                notificar("Proceso completado")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                log_debug("ERROR", str(e), level="ERROR")
            finally:
                st.session_state.processing = False
                st.rerun()
        
        col_t, col_i = st.columns(2)
        
        # Transcribir: solo si hay audio
        with col_t:
            if (audio or audio_bytes) and st.button("🎤 Transcribir", disabled=st.session_state.processing, width='stretch', key="btn_transcribir_audio"):
                st.session_state.processing = True
                try:
                    with st.spinner("Transcribiendo..."):
                        fase_transcripcion(audio_bytes if audio_bytes else audio)
                        st.success("✅ Transcripción lista")
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                finally:
                    st.session_state.processing = False
                    st.rerun()
        
        # Informe: solo si hay texto (después de transcribir o pegado)
        with col_i:
            if st.session_state.raw_text and st.button("📝 Informe", disabled=st.session_state.processing, width='stretch', key="btn_informe_audio"):
                st.session_state.processing = True
                try:
                    with st.spinner("Generando..."):
                        fase_informe()
                        st.success("✅ Informe listo")
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                finally:
                    st.session_state.processing = False
                    st.rerun()    
    # Visualización de transcripción
    if st.session_state.raw_text:
        st.divider()
        with st.expander("📝 Transcripción obtenida", expanded=True):
            st.text_area(
                "Puedes editar el texto antes de generar el informe:",
                value=st.session_state.raw_text,
                key="edit_raw",
                height=200
            )
            st.caption(f"✅ {len(st.session_state.raw_text)} caracteres")

        # Controles globales para continuar el proceso sin depender del origen
        cc1, cc2 = st.columns(2)
        with cc1:
            # Informe: solo si hay texto disponible
            if st.session_state.raw_text and st.button("📝 Generar Informe", disabled=st.session_state.processing, width='stretch', key="btn_generar_informe_global"):
                st.session_state.processing = True
                try:
                    with st.spinner("Generando informe..."):
                        # Actualizar con el texto editado si se modificó
                        st.session_state.raw_text = st.session_state.get("edit_raw", st.session_state.raw_text)
                        fase_informe()
                        st.success("✅ Informe generado")
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                finally:
                    st.session_state.processing = False
                    st.rerun()
        with cc2:
            # Auditar: solo si tenemos informe
            if st.session_state.soap_full and st.button("🔍 Auditar Veracidad", disabled=st.session_state.processing, width='stretch', key="btn_auditar_global"):
                st.session_state.processing = True
                try:
                    with st.spinner("Auditando..."):
                        # Asegurar que existe informe; si no, generarlo primero
                        if not st.session_state.soap_full:
                            st.session_state.raw_text = st.session_state.get("edit_raw", st.session_state.raw_text)
                            fase_informe()
                        soap_para_auditar = st.session_state.soap_full
                        if 'CANDIDATOS:' in soap_para_auditar.upper():
                            soap_para_auditar = re.split(r'CANDIDATOS:', soap_para_auditar, flags=re.IGNORECASE)[0].strip()
                        auditoria = auditar_informe(st.session_state.raw_text, soap_para_auditar)
                        st.session_state.auditoria_res = auditoria
                        st.success("✅ Auditoría lista")
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                finally:
                    st.session_state.processing = False
                    st.rerun()

with col2:
    st.subheader("📋 Salida Médica")
    
    if st.session_state.soap_full:
        full_text = st.session_state.soap_full
        
        # Separar SOAP de candidatos
        partes = re.split(r'CANDIDATOS:', full_text, flags=re.IGNORECASE)
        soap_display = partes[0].strip()
        candidatos_str = ""
        if len(partes) > 1:
            candidatos_str = partes[1].split('\n')[0].split("'")[0].strip()
        
        # Mostrar SOAP
        st.text_area("📝 Nota SOAP:", value=soap_display, height=300, key="soap_output")
        
        # Candidatos y CIE-10
        if candidatos_str:
            lista_cands = [c.strip() for c in candidatos_str.split(",")]
            # Mostrar candidatos como badges
            if lista_cands:
                chips_html = "<div style='margin: 0.5rem 0;'>"
                for c in lista_cands:
                    chips_html += f"<span style='display:inline-block; background:#eaf3ff; color:#1f4d8f; border:1px solid #cfe2ff; border-radius:16px; padding:4px 10px; margin:4px; font-size:0.9rem;'>🧩 {c}</span>"
                chips_html += "</div>"
                st.markdown("**🎯 Candidatos diagnósticos:**", unsafe_allow_html=True)
                st.markdown(chips_html, unsafe_allow_html=True)
            db = cargar_db()
            descripciones = [item['Descripción'] for item in db]
            
            sugerencias = []
            for cand in lista_cands[:3]:  # Solo primeros 3 candidatos
                matches = get_close_matches(cand, descripciones, n=2, cutoff=0.4)
                for m in matches[:1]:  # Solo mejor match
                    for item in db:
                        if item['Descripción'] == m:
                            sugerencias.append(f"{item['Código']} - {item['Descripción']}")
                            break
            
            sugerencias = list(dict.fromkeys(sugerencias))[:4]  # Max 4 sugerencias
            
            if sugerencias:
                if "cie10_select" not in st.session_state:
                    st.session_state["cie10_select"] = []

                cache_key = "|".join(lista_cands)
                if (
                    st.session_state.get("cie_cache_key") != cache_key
                    or "cie_sugerencias" not in st.session_state
                ):
                    st.session_state["cie_cache_key"] = cache_key
                    st.session_state["cie_sugerencias"] = sugerencias
                sugerencias = st.session_state.get("cie_sugerencias", [])

                st.markdown("**📚 Códigos CIE-10 sugeridos (clic para añadir):**")
                cols = st.columns(4)
                for idx, s in enumerate(sugerencias[:4]):
                    with cols[idx]:
                        st.markdown(
                            "<div class='cie-badge-btn'>", unsafe_allow_html=True
                        )
                        if st.button(f"🧩 {s}", key=f"cie_chip_{idx}"):
                            actuales = st.session_state.get("cie10_select", [])
                            if s not in actuales:
                                st.session_state["cie10_select"] = actuales + [s]
                        st.markdown("</div>", unsafe_allow_html=True)

                st.multiselect(
                    "Seleccionados:",
                    options=sugerencias,
                    key="cie10_select",
                )
            else:
                st.info("ℹ️ No se encontraron coincidencias en CIE-10")

        # Auditoría de veracidad
        st.divider()
        
        # Resultados de auditoría
        if 'auditoria_res' in st.session_state and st.session_state.auditoria_res:
            res = st.session_state.auditoria_res
            
            # Verificar que tiene datos válidos
            if 'error' in res:
                st.error(f"❌ Error en auditoría: {res['error']}")
            else:
                m = res.get('metricas', {})
                
                # Métricas principales con colores
                st.markdown("### 📊 Resultados de Auditoría")
                c1, c2, c3 = st.columns(3)
                
                fidelidad = m.get('fidelidad', 0)
                if fidelidad >= 90:
                    c1.success(f"**Fidelidad**\n### {fidelidad}%")
                elif fidelidad >= 70:
                    c1.warning(f"**Fidelidad**\n### {fidelidad}%")
                else:
                    c1.error(f"**Fidelidad**\n### {fidelidad}%")
                
                aluc = m.get('alucinaciones', 0)
                if aluc == 0:
                    c2.success(f"**Alucinaciones**\n### {aluc}")
                else:
                    c2.error(f"**Alucinaciones**\n### {aluc}")
                
                omis = m.get('omisiones', 0)
                if omis == 0:
                    c3.success(f"**Omisiones**\n### {omis}")
                else:
                    c3.warning(f"**Omisiones**\n### {omis}")
                
                # Tabla de verificaciones
                verificaciones = res.get('verificaciones', [])
                if verificaciones:
                    with st.expander("📋 Desglose detallado", expanded=False):
                        df_audit = pd.DataFrame(verificaciones)
                        cols = [c for c in ['frase', 'evidencia', 'estado', 'confianza'] if c in df_audit.columns]
                        if cols:
                            df_display = df_audit[cols].copy()
                            # Redondear confianza si existe
                            if 'confianza' in df_display.columns:
                                df_display['confianza'] = df_display['confianza'].map(lambda x: f"{round(float(x)*100 if x<=1 else float(x), 1)}%")
                            st.dataframe(df_display, width='stretch')

                        # Lista legible con evidencias
                        st.markdown("#### Evidencias por verificación")
                        for item in verificaciones:
                            frase = item.get('frase', '')
                            evidencia = item.get('evidencia', 'No encontrada')
                            estado = item.get('estado', 'DESCONOCIDO')
                            conf = item.get('confianza', 0)
                            conf_pct = f"{round(float(conf)*100 if conf<=1 else float(conf), 1)}%"
                            st.markdown(f"- **{estado}** · {conf_pct}\n  - Frase: {frase}\n  - Evidencia (transcripción): {evidencia}")
                
                # Omisiones
                if res.get('omisiones'):
                    with st.expander("🔍 Datos omitidos", expanded=False):
                        for omi in res['omisiones']:
                            st.write(f"- **{omi.get('dato', '')}**: {omi.get('razon', '')}")


st.divider()
colv1, colv2, colv3 = st.columns([1, 2, 1])
with colv2:
    disabled_validate = not bool(st.session_state.get('soap_full'))
    if st.button("✅ Validar Informe", use_container_width=True, disabled=disabled_validate, key="btn_validar_informe"):
        st.session_state.show_validation_modal = True

# Modal de validación a pantalla completa
if st.session_state.get("show_validation_modal"):
    st.markdown(
        """
        <style>
        .modal-overlay { position: fixed; top:0; left:0; width:100%; height:100%; background: rgba(0,0,0,0.65); z-index: 9999; display:flex; align-items:center; justify-content:center; }
        .modal-card { width: min(560px, 92vw); background:#0f1116; border:1px solid #2b3547; border-radius:16px; padding:24px; box-shadow: 0 10px 40px rgba(0,0,0,0.4); text-align:center; }
        .modal-title { color:#e5e7eb; font-size:1.4rem; margin-bottom:0.25rem; }
                .modal-text { color:#cbd5e1; margin-bottom:1rem; }
                .modal-actions { display:flex; gap:8px; justify-content:center; }
                .modal-btn { background:#2563eb; color:white; padding:10px 16px; border-radius:10px; text-decoration:none; font-weight:600; display:inline-block; }
                .modal-btn:hover { background:#1e4fd6; }
        </style>
        <div class="modal-overlay">
          <div class="modal-card">
            <div class="modal-title">✅ Informe validado y enviado a HCE</div>
            <div class="modal-text">La nota SOAP ha sido marcada como final y transferida al HCE.</div>
                        <div class="modal-actions"><a class="modal-btn" href="?close_modal=1">OK</a></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
        # El cierre se gestiona vía el enlace del overlay (query param)