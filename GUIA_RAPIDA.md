# 🚀 GUÍA RÁPIDA PARA EL PROFESOR

## Opción A: Descarga y Setup Automático (RECOMENDADO)

Pasos para descargar, instalar y ejecutar **Escriba Médico Soberano** automáticamente.

### 1. Clonar el Repositorio

```powershell
git clone https://github.com/Seiksssss/escriba-tfm-min.git
cd escriba-tfm-min
```

### 2. Instalar Ollama (OBLIGATORIO)

Descarga e instala Ollama desde: **https://ollama.com/download**

Una vez instalado, abre una terminal PowerShell y ejecuta:

```powershell
ollama serve
```

Déjalo corriendo (es el servidor de modelos).

### 3. En OTRA terminal: Ejecutar script de descarga automática

```powershell
.\setup_model.ps1
```

Este script automáticamente:
- ✅ Descarga Aloe-Beta-8B (5-7 GB)
- ✅ Crea el modelo personalizado `escriba-aloe-v3`
- ✅ Todo listo en ~15-20 minutos

## 4. Instalar Dependencias de Python

En la misma terminal (con Ollama aún corriendo):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## 5. Ejecutar la App

```powershell
streamlit run app.py
```

Se abrirá automáticamente en tu navegador (http://localhost:8501).

---

## Opción B: Docker (TODO en un contenedor)

Si prefieres evitar instalaciones, usa Docker:

```bash
docker build -t escriba-medico:latest .
docker run -p 8501:8501 escriba-medico:latest
```

Ver más en [DOCKER.md](DOCKER.md).

---

## ✅ ¿Qué debería ver?

1. **UI Streamlit** con título "🩺 Escriba Médico Soberano v0.9"
2. **Barra lateral** con:
   - Información del Sistema (GPU, RAM)
   - Estado de Ollama (debe decir "✅ Conectado")
   - Selector de modelo (por defecto: **escriba-aloe-v3** ← modelo personalizado con prompt médico)

## 📁 Archivos Importantes

- `MODELO_ALOE_BETA.md`: Instrucciones detalladas del modelo
- `README.md`: Documentación completa del proyecto
- `app.py`: Aplicación principal
- `data/conversaciones/`: Audios de prueba (conv1.mp3, conv2.mp3, etc.)

## 🧪 Probar la App

1. Sube uno de los audios de prueba: `data/conversaciones/conv1.mp3`
2. Haz clic en **"🔄 Procesar TODO"**
3. Espera a que se complete (transcripción → informe → auditoría)

---

## ⏰ Tiempo de Instalación

| Paso | Tiempo |
|------|--------|
| Clonar repo | 1 min |
| Instalar Ollama | 5 min |
| Ejecutar `setup_model.ps1` | 10-15 min |
| Instalar Python deps | 5 min |
| Ejecutar app | 1 min |
| **Total** | **~30 min** |

**O con Docker:** ~30 min para compilar la primera vez (luego es instantáneo).
