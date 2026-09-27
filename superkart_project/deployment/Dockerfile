# ---------- Base image ----------
FROM python:3.11-slim

# ---------- Configuration ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MODEL_REPO=your-hf-username/superkart-sales-model

WORKDIR /app

# ---------- Dependencies (cached layer) ----------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------- Application code ----------
COPY app.py predictor.py ./

# Hugging Face Spaces runs containers as user 1000; give it a writable home for the HF cache
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ---------- Network & start command ----------
EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
