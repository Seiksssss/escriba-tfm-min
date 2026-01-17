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

### 2b. (IMPORTANTE) Crear Modelo Personalizado con Prompt Médico

Este proyecto incluye un `Modelfile` con el prompt especializado en medicina. Para cargarlo:

```powershell
ollama create escriba-aloe-v3 -f Modelfile
```

Este comando crea un modelo llamado `escriba-aloe-v3` que:
- ✅ Usa Aloe-Beta-8B como base
- ✅ Integra el prompt médico especializado (REGLAS DE ORO para SOAP)
- ✅ Configura parámetros óptimos (temperatura=0.05 para precisión)

**Nota:** La app usará automáticamente este modelo si lo creas. Si no lo creas, utilizará el modelo base sin el prompt personalizado.

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

El modelo **escriba-aloe-v3** (creado con nuestro Modelfile) está entrenado/configurado para:

- ✅ Generar notas SOAP estructuradas en español
- ✅ Aplicar **REGLAS DE ORO** para máxima precisión clínica
- ✅ Preservar terminología médica exacta (tecnicismos)
- ✅ Capturar números precisos (TA, FC, dosis, tiempos)
- ✅ Entender contexto rural y geriátrico
- ✅ Sugerir diagnósticos candidatos con nivel de probabilidad
- ✅ Mantener baja temperatura (0.05) para determinismo

**Sistema Prompt:** Incluye 6 secciones + Reglas de Oro para evitar alucinaciones.

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
