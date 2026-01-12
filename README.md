
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

- Descargue Ollama: https://ollama.com/download y abra la aplicación para iniciar el servicio.
- La app intentará detectar Ollama. Si necesita cambiar el host, hágalo desde la barra lateral (`OLLAMA_HOST`).
- Seleccione el modelo en la barra lateral (valor por defecto `escriba-tfm`).

## Arquitectura (alto nivel)

1. Transcripción: Whisper Large-v3 (faster-whisper) con VAD y beam search.
2. Generación: Modelo local en Ollama. Prompt estructurado y salida en JSON para asegurar formato SOAP.
3. Auditoría: DeBERTa MNLI (entailment), búsqueda semántica (SentenceTransformers) y análisis de modalidad.
4. Sugerencias CIE-10: búsqueda aproximada sobre [github_min/cie10_2026.json](cie10_2026.json).

## Demostración

- Suba un audio de prueba o pegue texto de transcripción.
- Pulse "Procesar TODO" para transcribir, generar informe y auditar.
- Revise métricas y evidencias; opcionalmente, seleccione códigos CIE-10 sugeridos.

## Solución de problemas

- "Ollama no disponible": abra la app de Ollama o ejecute `ollama serve` en una terminal.
- Descarga de modelos de `transformers`: la primera ejecución de DeBERTa puede tardar.
- Sin GPU: todo funciona en CPU, pero más lento.

## Estructura mínima incluida

- [github_min/app.py](app.py): UI Streamlit + lógica principal.
- [github_min/auditor_mejorado.py](auditor_mejorado.py): auditoría con MNLI + embeddings.
- [github_min/requirements.txt](requirements.txt): dependencias.
- [github_min/cie10_2026.json](cie10_2026.json): catálogo CIE-10 (2026).
- [github_min/.gitignore](.gitignore): excluye venv, logs y cachés.
- [github_min/start.ps1](start.ps1): script opcional de arranque en Windows.

## Nota ética

Esta herramienta es de apoyo. Las decisiones clínicas deben ser tomadas por profesionales de la salud. Verifique siempre la veracidad del informe y ajuste según criterio médico.

