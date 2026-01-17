# Modelo Aloe-Beta-v3 (Ollama)

## Resumen

**Escriba Médico** utiliza **Aloe-Beta-8B** como su generador de notas SOAP. Este modelo es:

- **Desarrollado por:** Barcelona Supercomputing Center (BSC-CNS)
- **Base:** Llama 3.1 Fine-tuned en español médico
- **Cuantización:** Q4_K_M (GGUF)
- **Ejecutable:** Localmente vía Ollama (sin depender de APIs externas)

## Instalación y Configuración

### 1. Descargar e Instalar Ollama

Si aún no tienes Ollama, descárgalo desde: https://ollama.com/download

Una vez instalado, ejecuta el servicio:

```powershell
ollama serve
```

### 2. Hacer Pull del Modelo Aloe-Beta-8B

En otra terminal, ejecuta:

```powershell
ollama pull hf.co/mradermacher/Llama3.1-Aloe-Beta-8B-GGUF:Q4_K_M
```

Esto descargará automáticamente el modelo (aprox. 5-7 GB).

### 3. Cargar el Modelo Personalizado (Modelfile)

Opcionalmente, puedes crear un modelo personalizado con instrucciones médicas. Usa el archivo `Modelfile` incluido:

```powershell
ollama create escriba-aloe-v3 -f Modelfile
ollama run escriba-aloe-v3
```

## Verificar el Modelo

Desde la app:

1. Abre Streamlit: `streamlit run app.py`
2. Expande la barra lateral: **🧠 Modelo local (Ollama)**
3. Si ves "✅ Conectado a Ollama", el servicio está activo
4. Selecciona el modelo en el dropdown (por defecto: `escriba-aloe-v3` o `hf.co/mradermacher/Llama3.1-Aloe-Beta-8B-GGUF:Q4_K_M`)

## Requisitos de Sistema

- **RAM:** 16 GB (mínimo), 32+ recomendado
- **GPU:** NVIDIA con CUDA (opcional pero recomendado para velocidad)
- **Disco:** 10-15 GB libres
- **Ancho de banda:** Para la descarga inicial

## Características Médicas

El modelo Aloe-Beta está entrenado para:

- Generar notas SOAP estructuradas en español
- Preservar terminología médica precisa
- Entender contexto rural y geriátrico
- Sugerir diagnósticos candidatos con CIE-10

## Notas Técnicas

- **Cuantización Q4_K_M:** Reduce tamaño sin perder precisión significativa
- **Latencia:** ~5-10 seg. por informe en GPU NVIDIA, ~30-60 seg. en CPU
- **Acceso:** El modelo NO se puede acceder desde internet; corre 100% localmente

## Troubleshooting

- **Error: "ollama command not found"**
  - Verifica que Ollama esté en PATH o usa la ruta completa: `C:\Program Files\Ollama\ollama.exe serve`

- **Error: "Model not found"**
  - Ejecuta: `ollama pull hf.co/mradermacher/Llama3.1-Aloe-Beta-8B-GGUF:Q4_K_M`

- **Respuesta vacía o lenta**
  - Verifica disponibilidad de RAM y GPU
  - Comprueba que el puerto 11434 (Ollama) no esté bloqueado

## Referencias

- Aloe-Beta: https://huggingface.co/BSC-LT/Aloe-Llama-3.1-8B
- Ollama: https://ollama.com
- GGUF Format: https://github.com/ggerganov/ggml
