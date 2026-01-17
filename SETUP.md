# Setup

## Prerequisites

- Python 3.11+
- Ollama (https://ollama.com/download)
- 16 GB RAM minimum
- GPU optional (NVIDIA CUDA recommended)

## Installation

### 1. Clone repository

```bash
git clone https://github.com/Seiksssss/escriba-tfm-min.git
cd escriba-tfm-min
```

### 2. Start Ollama service

```bash
ollama serve
```

### 3. Download and configure model

Run in separate terminal:

```bash
.\setup_model.ps1
```

This script:
- Downloads Aloe-Beta-8B (5-7 GB)
- Creates `escriba-aloe-v3` model with custom prompt
- Verifies installation

Or manually:

```bash
ollama pull hf.co/mradermacher/Llama3.1-Aloe-Beta-8B-GGUF:Q4_K_M
ollama create escriba-aloe-v3 -f Modelfile
```

### 4. Install Python dependencies

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

### 5. Run application

```bash
streamlit run app.py
```

Opens at http://localhost:8501

## Modelfile Configuration

The `Modelfile` defines:
- Base: Aloe-Beta-8B (GGUF Q4_K_M)
- System prompt: Medical SOAP generation
- Parameters: temperature 0.05, top_p 0.9, repeat_penalty 1.2

Prompt rules:
- Medication → include associated pathology
- No unverbalized findings
- Exact numeric values with units
- Medical terminology (MIR style)
- SOAP structure: S/O/A/P + Medication + Candidate diagnoses

## Docker (Alternative)

```bash
docker build -t escriba-medico:latest .
docker run -p 8501:8501 escriba-medico:latest
```

## Verification

Confirm model creation:

```bash
ollama list
# Output should include: escriba-aloe-v3
```

## Test Data

Audio samples in `data/conversaciones/`:
- conv1.mp3, conv2.mp3, conv3.mp3
- Unidad 5 Dialogo 1 La Visita al medico.mp3

## Architecture

1. **Transcription**: Whisper Large-v3 (faster-whisper) with VAD
2. **Generation**: Aloe-Beta-8B via Ollama with structured prompt
3. **Audit**: DeBERTa-MNLI entailment + semantic search + modality analysis
4. **ICD-10**: Fuzzy search on cie10_2026.json

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama not found | Restart terminal after installation |
| Connection refused | Ensure `ollama serve` running in separate terminal |
| Slow response | Check RAM availability (16+ GB recommended) |
| Model not created | Run: `ollama create escriba-aloe-v3 -f Modelfile` |

## Time estimates

| Step | Time |
|------|------|
| Clone + dependencies | 5 min |
| Ollama installation | 5 min |
| Model download + setup | 15 min |
| Python environment | 5 min |
| **Total** | ~30 min |

Docker: ~30 min first build (instant after)
