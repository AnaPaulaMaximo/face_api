"""
face_engine.py

Detecção + reconhecimento facial com InsightFace (SCRFD + ArcFace) rodando
no ONNX Runtime.

Só dois modelos do pacote são carregados (detecção e reconhecimento): os de
idade/gênero e landmarks 2D/3D não servem para a catraca e dobravam o tempo
de cada requisição.
"""

import base64
import binascii
import logging
import re
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger("face_engine")

_APP = None
# O ONNX Runtime já usa todos os núcleos numa inferência; duas em paralelo
# só se atrapalham. O lock deixa a latência previsível.
_LOCK = threading.Lock()

_DATA_URI_RE = re.compile(r"^data:[\w/\-.+]*;base64,", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


class InvalidImageError(ValueError):
    """Os bytes recebidos não são uma imagem decodificável."""


def get_engine():
    """Cria (uma única vez) o FaceAnalysis com detecção + reconhecimento e aquece o modelo."""
    global _APP
    with _LOCK:
        if _APP is None:
            from insightface.app import FaceAnalysis

            logger.info("Carregando modelo InsightFace (%s)...", config.FACE_MODEL)
            app = FaceAnalysis(
                name=config.FACE_MODEL,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(config.FACE_DET_SIZE, config.FACE_DET_SIZE))
            # A primeira inferência é mais lenta (alocações internas do ONNX
            # Runtime): faz uma agora para a primeira requisição real não pagar.
            dummy = np.zeros((config.FACE_DET_SIZE, config.FACE_DET_SIZE, 3), dtype=np.uint8)
            app.get(dummy, max_num=1)
            _APP = app
            logger.info("Modelo carregado (det_size=%d).", config.FACE_DET_SIZE)
    return _APP


def decode_base64_image(foto: str) -> bytes:
    """
    Converte a foto vinda do banco (string base64, com ou sem prefixo data URI)
    em bytes de imagem. Levanta ValueError se não for base64 válido.
    """
    if not foto:
        raise ValueError("Foto vazia.")
    texto = _DATA_URI_RE.sub("", foto.strip(), count=1)
    texto = _WS_RE.sub("", texto)
    try:
        return base64.b64decode(texto, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Base64 inválido: {exc}") from exc


def bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decodifica jpg/png/webp direto para BGR com OpenCV (2-3x mais rápido que PIL + conversão)."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("O arquivo enviado não é uma imagem válida.")
    return img


def best_face_embedding(image_bytes: bytes) -> Tuple[Optional[np.ndarray], float]:
    """
    Detecta o rosto principal da imagem (o maior e mais central — numa catraca,
    quem está na frente da câmera) e devolve seu embedding L2-normalizado
    (512-d, float32) e a confiança da detecção. (None, 0.0) se não houver rosto.

    Só esse rosto passa pelo modelo de reconhecimento, que é a etapa mais cara.
    """
    img = bytes_to_bgr(image_bytes)
    engine = get_engine()
    with _LOCK:
        faces = engine.get(img, max_num=1)
    if not faces:
        return None, 0.0
    face = faces[0]
    return face.normed_embedding.astype(np.float32), float(face.det_score)


def embedding_from_base64(foto: str) -> Optional[np.ndarray]:
    """Embedding do rosto de uma foto como está salva na tabela `pessoa`."""
    emb, _ = best_face_embedding(decode_base64_image(foto))
    return emb
