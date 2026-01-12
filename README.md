
# Escriba Médico Soberano (Presentación TFM)

Este paquete contiene el mínimo necesario para mostrar la aplicación a sus profesores.
Incluye la interfaz de Streamlit, el auditor mejorado y la base CIE-10 (2026).

## Requisitos

- Python 3.11 recomendado
- Windows (probado) o Linux/Mac
- GPU opcional (mejora Whisper y auditoría), CPU funciona
- Ollama instalado y ejecutándose para el modelo local

### Dependencias

Instale las dependencias:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

### Modelos (Ollama)

Instale Ollama desde https://ollama.com/download y asegúrese de que esté ejecutándose.
El archivo `app.py` intentará conectarse automáticamente a Ollama.

Si su modelo se llama diferente, puede elegirlo en la barra lateral.

## Ejecutar

Desde el directorio `github_min`:

```
streamlit run app.py
```

Suba un archivo de audio (mp3/wav/m4a) o pegue texto y genere el informe SOAP.
Opcionalmente, ejecute la auditoría mejorada.

## Qué incluye y qué no

Incluido:
- `app.py` (UI y lógica principal)
- `auditor_mejorado.py` (auditoría con MNLI + embeddings)
- `requirements.txt` (dependencias)
- `cie10_2026.json` (catálogo CIE-10 2026)
- `.gitignore` (excluye logs, venv, cachés, modelos pesados)
- `README.md` (este documento)

Excluido (por tamaño/privacidad):
- Carpeta `logs/` y datos crudos
- Modelos de Ollama o binarios
- Carpeta `models/` local y artefactos

## Notas

- Si no tiene GPU, cambiará a CPU automáticamente (más lento).
- Para la auditoría, se descargará `microsoft/deberta-large-mnli` la primera vez.
- Puede ajustar el host de Ollama desde la barra lateral.
