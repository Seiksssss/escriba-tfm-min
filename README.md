
<div align="center">

# Escriba Médico — TFM

Automated medical note generation (SOAP) from speech/text with fidelity audit.

</div>

## Overview

Medical scribe assistant using local open-source components: Whisper Large-v3 (transcription), Aloe-Beta-8B via Ollama (SOAP generation), DeBERTa-MNLI + embeddings (audit). Structured JSON output with ICD-10 suggestions.

## Quick Start

See [SETUP.md](SETUP.md) for detailed installation.

```bash
git clone https://github.com/Seiksssss/escriba-tfm-min.git
cd escriba-tfm-min
.\setup_model.ps1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download es_core_news_sm
streamlit run app.py
```

## Components

### 1. Transcription
- Model: Whisper Large-v3
- Library: faster-whisper
- Features: VAD, beam search

### 2. Generation
- Model: Aloe-Beta-8B
- Executor: Ollama (local, no API)
- Quantization: Q4_K_M (GGUF)
- Prompt: Medical SOAP structure

### 3. Audit
- Entailment: DeBERTa-MNLI
- Semantic: SentenceTransformers embeddings
- Modality: Verbal mode detection
- Metrics: Fidelity %, hallucinations, omissions

### 4. ICD-10
- Database: cie10_2026.json
- Method: Fuzzy string matching

## Requirements

- Python 3.11+
- Ollama + Aloe-Beta-8B (5-7 GB)
- 16 GB RAM minimum
- GPU optional (NVIDIA CUDA recommended)

## Installation

See [SETUP.md](SETUP.md).

## Files

- `app.py` — Streamlit UI + orchestration
- `auditor_mejorado.py` — Audit engine
- `Modelfile` — Aloe-Beta configuration
- `requirements.txt` — Python dependencies
- `cie10_2026.json` — ICD-10 2026 database
- `data/conversaciones/` — Test audio samples
- `setup_model.ps1` — Automated setup
- `SETUP.md` — Detailed setup instructions

## Modelfile

See [Modelfile](Modelfile) for configuration.

Parameters:
- temperature: 0.05
- top_p: 0.9
- repeat_penalty: 1.2

Prompt rules: medication-pathology correlation, no unverbalized findings, exact numeric values, medical terminology.

## Docker

```bash
docker build -t escriba-medico:latest .
docker run -p 8501:8501 escriba-medico:latest
```

## Repository

https://github.com/Seiksssss/escriba-tfm-min

