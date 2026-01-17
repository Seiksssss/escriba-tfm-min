FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos del proyecto
COPY requirements.txt .
COPY app.py .
COPY auditor_mejorado.py .
COPY cie10_2026.json .
COPY Modelfile .
COPY data/ ./data/

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download es_core_news_sm

# Instalar Ollama (en el contenedor)
RUN curl -fsSL https://ollama.ai/install.sh | sh || true

# Exponer puerto Streamlit
EXPOSE 8501

# Script de inicio
CMD ["sh", "-c", "ollama serve & sleep 5 && ollama pull hf.co/mradermacher/Llama3.1-Aloe-Beta-8B-GGUF:Q4_K_M && ollama create escriba-aloe-v3 -f Modelfile && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
