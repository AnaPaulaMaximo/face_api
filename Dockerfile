FROM python:3.11-slim

# FACE_MODEL=buffalo_s: o buffalo_l (ArcFace R50) passa de 670 MB ao carregar e não cabe nos
# 512 MB do Render. ORT_THREADS/OMP/OPENBLAS: uma thread só, a CPU do plano é fração de núcleo.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FACE_MODEL=buffalo_s \
    ORT_THREADS=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# libgl1/libglib2.0-0: OpenCV (todas as dependências Python já vêm como wheel, sem compilar)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

# Baixa os modelos (FACE_MODEL + liveness) já no build, para o container não baixar a cada cold start.
# Depois deixa só detecção (det_*) e reconhecimento (w600k_*): o FaceAnalysis carrega todo .onnx
# da pasta para descobrir o que é cada um, e o 1k3d68 sozinho custa ~140 MB no pico.
RUN python -c "import face_engine as fe; fe.get_engine()" \
    && cd /root/.insightface/models \
    && rm -f "$FACE_MODEL.zip" "$FACE_MODEL/1k3d68.onnx" "$FACE_MODEL/2d106det.onnx" "$FACE_MODEL/genderage.onnx" \
    && ls "$FACE_MODEL"

# O Render informa a porta na variável PORT; localmente usa 8000
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
