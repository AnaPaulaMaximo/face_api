"""
face_engine.py
Motor de detecção + reconhecimento facial usando InsightFace (RetinaFace + ArcFace),
rodando via ONNX Runtime. Não depende de dlib/face_recognition.
"""

import base64
import binascii
import io
import logging
import re
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("face_engine")

_APP = None  # instância singleton do FaceAnalysis

_DATA_URI_RE = re.compile(r"^data:(?P<mime>[\w/\-.+]+)?;base64,", re.IGNORECASE)


def get_engine():
    """
    Cria (uma única vez) e retorna a instância do FaceAnalysis do InsightFace.
    Usa o pacote 'buffalo_l' (detector RetinaFace + embeddings ArcFace, 512-d).
    Troque providers para ['CUDAExecutionProvider', 'CPUExecutionProvider']
    se você tiver GPU NVIDIA + onnxruntime-gpu instalado.
    """
    global _APP
    if _APP is None:
        from insightface.app import FaceAnalysis

        logger.info("Carregando modelo InsightFace (buffalo_l)...")
        _APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _APP.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("Modelo carregado.")
    return _APP


def decode_base64_image(foto: str) -> Tuple[bytes, str]:
    """
    Converte a foto vinda do banco (string base64) em bytes de imagem.
    Aceita tanto base64 puro quanto data URI ("data:image/jpeg;base64,...").
    Retorna (bytes, mime_type).

    Levanta ValueError se a string não for um base64 válido.
    """
    if not foto:
        raise ValueError("Foto vazia.")

    mime = "image/jpeg"
    texto = foto.strip()

    match = _DATA_URI_RE.match(texto)
    if match:
        mime = match.group("mime") or mime
        texto = texto[match.end():]

    # Remove quebras de linha/espaços que costumam vir de base64 formatado.
    texto = re.sub(r"\s+", "", texto)

    try:
        image_bytes = base64.b64decode(texto, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Base64 inválido: {exc}") from exc

    if not match:
        mime = _sniff_mime(image_bytes)
    return image_bytes, mime


def _sniff_mime(image_bytes: bytes) -> str:
    """Descobre o tipo da imagem pelos primeiros bytes (fallback: jpeg)."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"GIF8"):
        return "image/gif"
    if image_bytes.startswith(b"BM"):
        return "image/bmp"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


class InvalidImageError(ValueError):
    """Arquivo enviado não é uma imagem válida."""


def bytes_to_bgr_image(image_bytes: bytes) -> np.ndarray:
    """Converte bytes de imagem (jpg/png/etc) em array BGR (formato OpenCV)."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise InvalidImageError("O arquivo enviado não é uma imagem válida.") from exc
    arr = np.array(img)  # RGB
    return arr[:, :, ::-1].copy()  # -> BGR


def extract_faces(image_bytes: bytes):
    """
    Detecta todos os rostos em uma imagem e retorna a lista de objetos Face
    do InsightFace (cada um com .bbox, .kps, .det_score, .embedding, .normed_embedding).
    """
    engine = get_engine()
    img_bgr = bytes_to_bgr_image(image_bytes)
    faces = engine.get(img_bgr)
    return faces, img_bgr


def get_single_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Retorna o embedding normalizado (512-d, float32) do rosto de maior confiança
    encontrado na imagem, ou None se nenhum rosto for detectado.
    """
    faces, _ = extract_faces(image_bytes)
    if not faces:
        return None
    best = max(faces, key=lambda f: f.det_score)
    return best.normed_embedding.astype(np.float32)


def embedding_from_base64(foto: str) -> Optional[np.ndarray]:
    """
    Atalho: recebe a foto em base64 (como está salva na tabela `pessoa`) e
    devolve o embedding do rosto. None se não houver rosto detectável.
    """
    image_bytes, _ = decode_base64_image(foto)
    return get_single_embedding(image_bytes)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similaridade de cosseno entre dois embeddings normalizados (-1 a 1)."""
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))