# 🚀 GUÍA RÁPIDA PARA EL PROFESOR

Pasos para descargar, instalar y ejecutar **Escriba Médico Soberano** en tu equipo.

## 1. Clonar el Repositorio

```powershell
git clone https://github.com/Seiksssss/escriba-tfm-min.git
cd escriba-tfm-min
```

## 2. Instalar Ollama (OBLIGATORIO)

Descarga e instala Ollama desde: **https://ollama.com/download**

Una vez instalado, abre una terminal PowerShell y ejecuta:

```powershell
ollama serve
```

Déjalo corriendo (es el servidor de modelos).

## 3. En OTRA terminal: Descargar Modelo Aloe-Beta-8B

```powershell
ollama pull hf.co/mradermacher/Llama3.1-Aloe-Beta-8B-GGUF:Q4_K_M
```

Esto tardará unos 5-10 minutos (descarga ~5-7 GB).

## 3b. (IMPORTANTE) Crear Modelo Personalizado con Prompt Médico

```powershell
ollama create escriba-aloe-v3 -f Modelfile
```

Este comando crea el modelo personalizado que usará la app. **Sin este paso, no tendrá el prompt médico especializado.**

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

## ⚠️ Requisitos Mínimos

- **Windows 10+, macOS o Linux**
- **Python 3.11**
- **RAM:** 16 GB (mínimo)
- **GPU NVIDIA** (opcional pero recomendado)
- **Conexión a internet** (solo para descargas iniciales)

## 🆘 Solución de Problemas

| Problema | Solución |
|----------|----------|
| "ollama command not found" | Reinicia la terminal después de instalar Ollama |
| "Connection refused" a Ollama | Verifica que `ollama serve` esté corriendo en otra terminal |
| App lenta / no responde | Asegúrate de tener RAM suficiente (16+ GB) |
| Modelo no se descarga | Ejecuta manualmente: `ollama pull hf.co/mradermacher/Llama3.1-Aloe-Beta-8B-GGUF:Q4_K_M` |

## 📞 Contacto

Repositorio: https://github.com/Seiksssss/escriba-tfm-min

---

**¡Listo!** Si todo va bien, deberías poder transcribir audios y generar notas SOAP automáticamente. 🎉
