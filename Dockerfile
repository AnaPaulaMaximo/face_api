FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# build-essential: compilar o insightface | libgl1/libglib2.0-0: OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

# Baixa os modelos (buffalo_l + liveness) já no build, para o container não baixar a cada cold start
RUN python -c "import face_engine as fe; fe.get_engine()"

# O Render informa a porta na variável PORT; localmente usa 8000
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
