"""
face_engine.py

Detecção + reconhecimento facial com InsightFace (SCRFD + ArcFace) rodando
no ONNX Runtime, com anti-spoofing opcional (addon `liveness`).

Só dois modelos do pacote são carregados (detecção e reconhecimento): os de
idade/gênero e landmarks 2D/3D não servem para a catraca e dobravam o tempo
de cada requisição.

Com FACE_LIVENESS=true o addon `liveness` do InsightFace roda entre a
detecção e o reconhecimento: classifica o rosto como vivo ou não (foto no
celular, impressão, tela). Rosto reprovado nem passa pelo ArcFace.
"""

import base64
import binascii
import logging
import re
import threading
from typing import NamedTuple, Optional

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


class FaceResult(NamedTuple):
    """Resultado da análise do rosto principal de uma imagem."""

    embedding: Optional[np.ndarray]  # None: sem rosto, ou rosto reprovado no anti-spoofing
    det_score: float                 # confiança da detecção (0.0 = nenhum rosto)
    live: Optional[bool]             # None: anti-spoofing desligado ou não avaliado
    live_score: Optional[float]      # probabilidade de ser rosto vivo (0 a 1)
    liveness_status: Optional[str]   # "ok" | "input_rejected" | None

    @property
    def has_face(self) -> bool:
        return self.det_score > 0.0


_NO_FACE = FaceResult(None, 0.0, None, None, None)


def get_engine():
    """Cria (uma única vez) o FaceAnalysis com detecção + reconhecimento (+ liveness) e aquece os modelos."""
    global _APP
    with _LOCK:
        if _APP is None:
            from insightface.app import FaceAnalysis

            logger.info(
                "Carregando modelo InsightFace (%s, anti-spoofing %s)...",
                config.FACE_MODEL,
                "ligado" if config.FACE_LIVENESS else "desligado",
            )
            app = FaceAnalysis(
                name=config.FACE_MODEL,
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
                liveness_mode="normal",  # rosto reprovado não gera embedding
                liveness_threshold=config.LIVENESS_THRESHOLD,
            )
            if config.FACE_LIVENESS:
                app.addons["liveness"] = _build_liveness()
            app.prepare(ctx_id=0, det_size=(config.FACE_DET_SIZE, config.FACE_DET_SIZE))
            # A primeira inferência é mais lenta (alocações internas do ONNX
            # Runtime): faz uma agora para a primeira requisição real não pagar.
            dummy = np.zeros((config.FACE_DET_SIZE, config.FACE_DET_SIZE, 3), dtype=np.uint8)
            app.get(dummy, max_num=1)
            liveness = app.addons.get("liveness")
            if liveness is not None:
                # A imagem vazia não tem rosto, então o liveness não rodou: aquece direto.
                liveness.session.run(
                    [liveness.output_name],
                    {liveness.input_name: np.zeros((1, 3, 80, 80), dtype=np.float32)},
                )
            _APP = app
            logger.info(
                "Modelo carregado (det_size=%d, liveness_threshold=%.2f).",
                config.FACE_DET_SIZE,
                config.LIVENESS_THRESHOLD,
            )
    return _APP


def _build_liveness():
    """
    Sessão ONNX do addon `liveness` com UMA thread e sem spin-wait.

    O modelo é minúsculo (80x80, ~1,5 MB) e roda em ~9 ms numa thread só. Com
    o pool padrão (todos os núcleos), as threads dele ficam girando depois da
    inferência e roubam CPU do ArcFace, que roda logo em seguida: medido, o
    addon custava +76 ms por requisição; assim, +12 ms.

    `ensure_addon` baixa `liveness.onnx` para ~/.insightface/addons/ na
    primeira vez (e confere o SHA-256).
    """
    import onnxruntime as ort
    from insightface.addons import Liveness, ensure_addon

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return Liveness(
        ensure_addon("liveness"),
        threshold=config.LIVENESS_THRESHOLD,
        providers=["CPUExecutionProvider"],
        sess_options=so,
    )


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


def analyze_main_face(image_bytes: bytes, check_liveness: bool = True) -> FaceResult:
    """
    Detecta o rosto principal da imagem (o maior e mais central — numa catraca,
    quem está na frente da câmera), opcionalmente verifica se é um rosto vivo
    e devolve seu embedding L2-normalizado (512-d, float32).

    Só esse rosto passa pelo modelo de reconhecimento, que é a etapa mais cara.
    Com o anti-spoofing ligado (FACE_LIVENESS) e `check_liveness=True`, rosto
    reprovado volta com `embedding=None` e `live=False` (ou `live=None` com
    `liveness_status="input_rejected"` quando o rosto está cortado demais na
    imagem para ser avaliado).

    `check_liveness=False` serve para as fotos de cadastro: elas SÃO fotos e
    precisam virar embedding mesmo assim.
    """
    img = bytes_to_bgr(image_bytes)
    engine = get_engine()
    with _LOCK:
        gate = bool(engine.addons) and not check_liveness
        if gate:
            # "observe": o addon ainda roda, mas não impede o reconhecimento.
            engine.liveness_mode = "observe"
        try:
            faces = engine.get(img, max_num=1)
        finally:
            if gate:
                engine.liveness_mode = "normal"
    if not faces:
        return _NO_FACE
    face = faces[0]
    liveness = face.get("liveness")
    embedding = face.get("embedding")
    if embedding is not None:
        embedding = face.normed_embedding.astype(np.float32)
    if liveness is None:
        return FaceResult(embedding, float(face.det_score), None, None, None)
    return FaceResult(
        embedding,
        float(face.det_score),
        liveness["is_live"],
        liveness["live_score"],
        liveness["status"],
    )


def embedding_from_base64(foto: str) -> Optional[np.ndarray]:
    """Embedding do rosto de uma foto como está salva na tabela `pessoa` (sem anti-spoofing: é uma foto)."""
    return analyze_main_face(decode_base64_image(foto), check_liveness=False).embedding
