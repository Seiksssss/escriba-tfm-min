
<div align="center">

# Escriba Médico Soberano — TFM

Generación asistida de notas clínicas (SOAP) a partir de voz/texto con auditoría de veracidad.

</div>

## Resumen

Este repositorio contiene el mínimo necesario para demostrar la aplicación del TFM ante el tribunal. La interfaz está construida con Streamlit, la transcripción usa Whisper (vía `faster-whisper`), la generación corre en un modelo local vía Ollama, y la auditoría combina MNLI (DeBERTa) y búsqueda semántica por embeddings.

## Objetivos del TFM

- Reducir el tiempo de documentación clínica mediante transcripción y síntesis.
- Generar notas SOAP estructuradas y legibles.
- Auditar la veracidad del informe con evidencias de la transcripción.
- Sugerir códigos CIE-10 (2026) a partir de candidatos del modelo.

## Características

- Entrada por audio (mp3/wav/m4a) o texto pegado.
- Nota SOAP en formato claro (S/O/A/P) y candidatos diagnósticos.
- Auditoría mejorada: MNLI (entailment), modalidad verbal y búsqueda semántica.
- Visualización de métricas: fidelidad, alucinaciones y omisiones.

## Requisitos

- Python 3.11 (recomendado)
- Windows (probado) o Linux/macOS
- GPU opcional (acelera Whisper y NLI), CPU compatible
- Ollama instalado y en ejecución para el modelo local

## Instalación rápida

```powershell
python -m venv .venv
.# .venv\Scripts\Activate.ps1 en Windows
.# source .venv/bin/activate en Linux/macOS
\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

Si prefiere un arranque asistido en Windows, use [github_min/start.ps1](start.ps1).

## Ejecutar la app

```powershell
streamlit run app.py
```

## Modelos (Ollama)

**Ollama es obligatorio.** Instálalo desde https://ollama.com/download, luego ejecuta:

```powershell
ollama serve
```

En otra terminal, descarga el modelo Aloe-Beta-8B:

```powershell
ollama pull hf.co/mradermacher/Llama3.1-Aloe-Beta-8B-GGUF:Q4_K_M
```

Ver instrucciones detalladas: [MODELO_ALOE_BETA.md](MODELO_ALOE_BETA.md)

La app detectará Ollama automáticamente. Si necesita cambiar el host, hágalo desde la barra lateral (`OLLAMA_HOST`).

## Arquitectura (alto nivel)

1. **Transcripción:** Whisper Large-v3 (faster-whisper) con VAD y beam search.
2. **Generación (Aloe-Beta):** Modelo local en Ollama (5-7 GB). Prompt estructurado y salida en JSON para asegurar formato SOAP.
3. **Auditoría:** DeBERTa MNLI (entailment), búsqueda semántica (SentenceTransformers) y análisis de modalidad.
4. **Sugerencias CIE-10:** Búsqueda aproximada sobre [github_min/cie10_2026.json](cie10_2026.json).

## Demostración

- Suba un audio de prueba o pegue texto de transcripción.
- Pulse "Procesar TODO" para transcribir, generar informe y auditar.
- Revise métricas y evidencias; opcionalmente, seleccione códigos CIE-10 sugeridos.

## Solución de problemas

- **Ollama no conecta:** Descarga Ollama desde https://ollama.com/download, luego ejecuta `ollama serve`.
- **Modelo no se descarga:** Ejecuta manualmente: `ollama pull hf.co/mradermacher/Llama3.1-Aloe-Beta-8B-GGUF:Q4_K_M`
- **Descarga de modelos de `transformers`:** La primera ejecución de DeBERTa puede tardar (descarga automática).
- **Sin GPU:** Todo funciona en CPU, pero más lento (~30-60 seg. por informe vs. 5-10 seg. en GPU).

Ver más detalles en [MODELO_ALOE_BETA.md](MODELO_ALOE_BETA.md).

## Estructura mínima incluida

- [MODELO_ALOE_BETA.md](MODELO_ALOE_BETA.md): **instrucciones detalladas para configurar Aloe-Beta-8B.**
- [app.py](app.py): UI Streamlit + lógica principal.
- [auditor_mejorado.py](auditor_mejorado.py): auditoría con MNLI + embeddings.
- [requirements.txt](requirements.txt): dependencias.
- [cie10_2026.json](cie10_2026.json): catálogo CIE-10 (2026).
- [.gitignore](.gitignore): excluye venv, logs y cachés.
- [start.ps1](start.ps1): script opcional de arranque en Windows.
- [data/conversaciones](data/conversaciones): audios de prueba.

## Nota ética

Esta herramienta es de apoyo. Las decisiones clínicas deben ser tomadas por profesionales de la salud. Verifique siempre la veracidad del informe y ajuste según criterio médico.

