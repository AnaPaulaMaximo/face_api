"""
main.py
API REST de reconhecimento facial para catraca (FastAPI + InsightFace).

O CADASTRO NÃO ACONTECE AQUI: as pessoas são cadastradas por outra API, que
grava na tabela `pessoa` (cpf, nome, foto em base64). Esta API apenas:

  1. lê as fotos da tabela `pessoa` e mantém os embeddings em memória;
  2. verifica se o rosto enviado pelo frontend é de uma pessoa viva (e não
     uma foto no celular/impressa), reconhece e responde liberado/recusado;
  3. grava o log de entrada/saída na tabela `logs` (em segundo plano).

Endpoints:
  GET    /health            -> healthcheck
  POST   /faces/recognize   -> reconhece o rosto e registra entrada/saída
  GET    /access/logs       -> lista os logs de entrada/saída (com filtros)
  GET    /faces/cache       -> status do cache de embeddings
  POST   /faces/reload      -> força reler a tabela `pessoa`

Rode com:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import List, Literal, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

import access
import config
import database as db
import face_engine as fe
import face_store as store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("face_api")


# Sinaliza o fim do carregamento inicial (modelo + cache + últimos registros).
_startup_concluido = threading.Event()
# Erros do carregamento inicial, expostos no /health para diagnosticar o deploy.
_erros_startup: List[str] = []


def _carregar_em_segundo_plano():
    """Conecta no banco e carrega o modelo e o cache sem segurar a porta HTTP.

    Em máquinas fracas (ex.: plano Free do Render, 0,1 CPU) o modelo leva
    minutos para carregar, e o Render desiste se a porta não abrir antes.
    Falha no banco também não derruba o processo: aparece no /health.
    O POST /faces/recognize espera este carregamento terminar.
    """
    try:
        try:
            db.init_db()
        except Exception as exc:
            logger.exception("Falha ao conectar no Cloudflare D1 no startup.")
            _erros_startup.append(f"banco: {exc}")
        fe.get_engine()  # carrega e aquece o modelo
        try:
            store.refresh(force=True)
            logger.info("Últimos registros carregados: %d pessoa(s).", access.carregar_ultimos())
        except Exception as exc:  # banco fora do ar não deve derrubar a API inteira
            logger.warning("Não foi possível carregar os dados do banco no startup: %s", exc)
            _erros_startup.append(f"cache: {exc}")
        store.start_background_refresh()
        logger.info("API pronta.")
    except Exception as exc:
        logger.exception("Falha ao carregar o modelo no startup.")
        _erros_startup.append(f"modelo: {exc}")
    finally:
        # Mesmo com falha libera as requisições: elas tentam carregar o modelo de novo
        _startup_concluido.set()


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_carregar_em_segundo_plano, name="startup", daemon=True).start()
    yield
    store.stop_background_refresh()


app = FastAPI(
    title="Face Recognition API",
    description=(
        "Reconhecimento facial para catraca (InsightFace: SCRFD + ArcFace). "
        "As pessoas e suas fotos (base64) vêm da tabela `pessoa`, alimentada por "
        "outra API. Aqui só são gravados os logs de entrada/saída."
    ),
    version="1.3.3",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(fe.InvalidImageError)
async def invalid_image_handler(_: Request, exc: fe.InvalidImageError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(db.D1Error)
async def d1_error_handler(_: Request, exc: db.D1Error):
    logger.error("Erro no Cloudflare D1: %s", exc)
    return JSONResponse(status_code=503, content={"detail": f"Banco de dados indisponível: {exc}"})


# ---------------------------------------------------------------------------
# Modelos de resposta
# ---------------------------------------------------------------------------

class Acesso(BaseModel):
    tipo: Literal["entrada", "saida"]
    data_hora: str
    registrado: bool  # False = já havia registro dentro do cooldown (frame repetido)


class Reconhecimento(BaseModel):
    autorizado: bool
    motivo: Optional[Literal["nenhum_rosto", "rosto_nao_vivo", "rosto_cortado", "nao_reconhecido"]] = None
    cpf: Optional[str] = None
    nome: Optional[str] = None
    similaridade: float
    # Probabilidade (0 a 1) de o rosto ser de uma pessoa real. None quando o
    # anti-spoofing está desligado (FACE_LIVENESS=false) ou não foi avaliado.
    live_score: Optional[float] = None
    acesso: Optional[Acesso] = None
    tempo_ms: int


class LogRecord(BaseModel):
    id: int
    cpf: str
    nome: Optional[str] = None
    tipo: str
    data_hora: str


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "pronta": _startup_concluido.is_set(),
        "modelo_carregado": fe.is_loaded(),
        "variaveis_faltando": config.missing_d1_settings(),
        "erros_startup": _erros_startup,
        "embeddings_em_cache": store.status()["embeddings_em_cache"],
        "anti_spoofing": config.FACE_LIVENESS,
    }


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


# Função síncrona de propósito: o FastAPI roda em threadpool e o event loop
# não fica bloqueado durante a inferência (que leva centenas de ms na CPU).
@app.post("/faces/recognize", response_model=Reconhecimento)
def recognize_face(background: BackgroundTasks, file: UploadFile = File(...)):
    """
    Recebe a foto da câmera, identifica o rosto principal, confere se é um
    rosto vivo (com FACE_LIVENESS=true) e responde se o acesso está
    autorizado. Quando autorizado, decide entrada/saída na hora e grava o log
    no banco em segundo plano (a resposta não espera o D1).
    """
    _startup_concluido.wait()
    inicio = time.perf_counter()

    image_bytes = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Imagem maior que o limite permitido.")

    rosto = fe.analyze_main_face(image_bytes)
    if not rosto.has_face:
        return _resposta(inicio, autorizado=False, motivo="nenhum_rosto", similaridade=0.0)

    if rosto.embedding is None:
        # Anti-spoofing reprovou (foto/tela/impressão) ou não conseguiu avaliar
        # porque o rosto está cortado demais na imagem (precisa de margem ao redor).
        motivo = "rosto_cortado" if rosto.liveness_status == "input_rejected" else "rosto_nao_vivo"
        return _resposta(inicio, autorizado=False, motivo=motivo, similaridade=0.0, live_score=rosto.live_score)

    cpf, nome, sim = store.identify(rosto.embedding)
    if cpf is None:
        return _resposta(
            inicio, autorizado=False, motivo="nao_reconhecido", similaridade=sim, live_score=rosto.live_score
        )

    decisao = access.decidir(cpf)
    if decisao["registrado"]:
        background.add_task(access.persistir, cpf, decisao["tipo"], decisao["data_hora"])

    return _resposta(
        inicio,
        autorizado=True,
        cpf=cpf,
        nome=nome,
        similaridade=sim,
        live_score=rosto.live_score,
        acesso={
            "tipo": decisao["tipo"],
            "data_hora": decisao["data_hora"].isoformat(),
            "registrado": decisao["registrado"],
        },
    )


def _resposta(inicio: float, **campos) -> dict:
    campos["similaridade"] = round(campos["similaridade"], 4)
    if campos.get("live_score") is not None:
        campos["live_score"] = round(campos["live_score"], 4)
    campos["tempo_ms"] = int((time.perf_counter() - inicio) * 1000)
    return campos


@app.get("/access/logs", response_model=List[LogRecord])
def access_logs(
    cpf: Optional[str] = Query(None, description="Filtrar pelo CPF da pessoa"),
    tipo: Optional[Literal["entrada", "saida"]] = Query(None, description="Filtrar por tipo"),
    date_from: Optional[date] = Query(None, description="Data inicial (YYYY-MM-DD)"),
    date_to: Optional[date] = Query(None, description="Data final (YYYY-MM-DD)"),
    limit: int = Query(200, ge=1, le=1000),
):
    """Lista os registros de entrada/saída, mais recentes primeiro, com filtros opcionais."""
    rows = db.list_logs(cpf=cpf, tipo=tipo, date_from=date_from, date_to=date_to, limit=limit)
    return [{**r, "data_hora": r["data_hora"].isoformat()} for r in rows]


@app.get("/faces/cache")
def cache_status():
    """Quantos rostos estão em memória e quais fotos cadastradas deram problema."""
    return store.status()


@app.post("/faces/reload")
def reload_faces():
    """
    Relê a tabela `pessoa` recalculando todos os embeddings e recarrega o
    último registro de cada pessoa. O cache também se atualiza sozinho a cada
    FACE_CACHE_TTL_SECONDS; use isto para efeito imediato após um cadastro.
    """
    resumo = store.refresh(force=True)
    resumo["ultimos_registros_carregados"] = access.carregar_ultimos()
    return resumo
