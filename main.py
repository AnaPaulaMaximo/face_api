"""
main.py
API REST de reconhecimento facial com FastAPI + InsightFace (ArcFace/RetinaFace).

O CADASTRO NÃO ACONTECE AQUI: as pessoas são cadastradas por outra API, que
grava na tabela `pessoa` (cpf, nome, foto em base64). Esta API apenas:

  1. lê as fotos base64 da tabela `pessoa` e gera os embeddings faciais;
  2. reconhece o rosto enviado pela câmera;
  3. grava o log de entrada/saída na tabela `logs`.

Endpoints:
  GET    /health                  -> healthcheck
  GET    /pessoas                 -> lista as pessoas vindas do banco
  GET    /pessoas/{cpf}/foto      -> devolve a foto do banco já decodificada
  POST   /faces/recognize         -> identifica rosto(s) e JÁ REGISTRA entrada/saída
  POST   /faces/verify            -> compara dois rostos (form-data: file1, file2)
  GET    /faces/cache             -> status do cache de embeddings
  POST   /faces/reload            -> força reler a tabela `pessoa`

  POST   /access/register         -> reconhece + registra ponto (totem dedicado)
  GET    /access/status/{cpf}     -> diz se a pessoa está "dentro" ou "fora" agora
  GET    /access/logs             -> lista os logs de entrada/saída (com filtros)

Rode com:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from datetime import date, datetime, timezone
from typing import List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import database as db
import face_engine as fe
import face_store as store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_api")

app = FastAPI(
    title="Face Recognition API",
    description=(
        "API de reconhecimento facial (InsightFace: ArcFace + RetinaFace). "
        "As pessoas e suas fotos (base64) vêm da tabela `pessoa`, alimentada por "
        "outra API. Aqui só são gravados os logs de entrada/saída."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(fe.InvalidImageError)
async def invalid_image_handler(request: Request, exc: fe.InvalidImageError):
    """Arquivo que não é imagem -> 422 em vez de 500."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.on_event("startup")
def startup():
    db.init_db()
    # Pré-carrega o modelo e os embeddings, para a primeira request não ser lenta.
    fe.get_engine()
    try:
        store.refresh(force=True)
    except Exception as exc:  # banco fora do ar não deve derrubar a API inteira
        logger.warning("Não foi possível carregar os rostos no startup: %s", exc)
    logger.info("API pronta.")


# ---------------------------------------------------------------------------
# Modelos de resposta
# ---------------------------------------------------------------------------

class Pessoa(BaseModel):
    cpf: str
    nome: str
    foto_url: str


class LogRecord(BaseModel):
    id: int
    cpf: str
    nome: Optional[str] = None
    tipo: str
    data_hora: str


# ---------------------------------------------------------------------------
# Registro automático de ponto
# ---------------------------------------------------------------------------

def _registrar_ponto(cpf: str) -> dict:
    """
    Registra automaticamente entrada/saída para uma pessoa já reconhecida.
    O tipo alterna com base no último registro, e registros repetidos dentro de
    ACCESS_COOLDOWN_SECONDS são ignorados (evita duplicar com frames seguidos).
    """
    ultimo = db.get_last_log(cpf)

    if ultimo is not None:
        decorrido = (datetime.now(timezone.utc) - ultimo["data_hora"]).total_seconds()
        if decorrido < config.ACCESS_COOLDOWN_SECONDS:
            return {
                "skipped": True,
                "tipo": ultimo["tipo"],
                "data_hora": ultimo["data_hora"].isoformat(),
            }

    tipo: Literal["entrada", "saida"] = (
        "entrada" if ultimo is None or ultimo["tipo"] == "saida" else "saida"
    )

    log = db.add_log(cpf=cpf, tipo=tipo)

    return {
        "skipped": False,
        "log_id": log["id"],
        "tipo": log["tipo"],
        "data_hora": log["data_hora"].isoformat(),
    }


# ---------------------------------------------------------------------------
# Básico / frontend
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def serve_frontend():
    """Serve a página web simples de reconhecimento via câmera."""
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")


# ---------------------------------------------------------------------------
# Pessoas (leitura do banco — cadastro é feito por outra API)
# ---------------------------------------------------------------------------

@app.get("/pessoas", response_model=List[Pessoa])
def listar_pessoas():
    """Lista as pessoas cadastradas na tabela `pessoa` (sem trazer o base64)."""
    return [
        {"cpf": p["cpf"], "nome": p["nome"], "foto_url": f"/pessoas/{p['cpf']}/foto"}
        for p in db.list_pessoas()
    ]


@app.get("/pessoas/{cpf}/foto")
def foto_pessoa(cpf: str):
    """Devolve a foto da pessoa (armazenada em base64) já decodificada."""
    pessoa = db.get_pessoa(cpf)
    if pessoa is None:
        raise HTTPException(status_code=404, detail=f"Pessoa '{cpf}' não encontrada.")
    if not pessoa["foto"]:
        raise HTTPException(status_code=404, detail=f"Pessoa '{cpf}' não tem foto cadastrada.")

    try:
        image_bytes, mime = fe.decode_base64_image(pessoa["foto"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Foto inválida no banco: {exc}")

    return Response(content=image_bytes, media_type=mime)


@app.get("/faces/cache")
def cache_status():
    """Mostra quantos rostos estão em memória e quais fotos deram problema."""
    return store.status()


@app.post("/faces/reload")
def reload_faces(force: bool = Query(True, description="Recalcular todos os embeddings")):
    """
    Relê a tabela `pessoa` e recalcula os embeddings.
    Útil logo depois que a API de cadastro insere alguém novo — embora o cache
    também se atualize sozinho a cada FACE_CACHE_TTL_SECONDS.
    """
    return store.refresh(force=force)


# ---------------------------------------------------------------------------
# Reconhecimento
# ---------------------------------------------------------------------------

@app.post("/faces/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """
    Detecta todos os rostos na imagem e identifica cada um comparando com os
    embeddings gerados a partir das fotos da tabela `pessoa`. Para cada pessoa
    reconhecida, a entrada/saída é registrada automaticamente na tabela `logs`.
    """
    image_bytes = await file.read()
    faces, _ = fe.extract_faces(image_bytes)

    if not faces:
        return {"faces_found": 0, "results": []}

    results = []
    for face in faces:
        emb = face.normed_embedding.astype("float32")
        cpf, nome, sim = store.identify(emb)

        acesso = _registrar_ponto(cpf) if cpf else None

        results.append(
            {
                "cpf": cpf,
                "nome": nome or "desconhecido",
                "reconhecido": cpf is not None,
                "similarity": round(sim, 4),
                "bbox": [float(x) for x in face.bbox],
                "det_score": float(face.det_score),
                "access": acesso,
            }
        )

    return {"faces_found": len(faces), "results": results}


@app.post("/faces/verify")
async def verify_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Compara dois rostos e diz se são a mesma pessoa."""
    emb1 = fe.get_single_embedding(await file1.read())
    emb2 = fe.get_single_embedding(await file2.read())

    if emb1 is None:
        raise HTTPException(status_code=422, detail="Nenhum rosto detectado na primeira imagem.")
    if emb2 is None:
        raise HTTPException(status_code=422, detail="Nenhum rosto detectado na segunda imagem.")

    sim = fe.cosine_similarity(emb1, emb2)
    return {
        "similarity": round(sim, 4),
        "same_person": sim >= config.SIMILARITY_THRESHOLD,
        "threshold": config.SIMILARITY_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# Registro de entrada / saída (ponto)
# ---------------------------------------------------------------------------

@app.post("/access/register")
async def register_access(file: UploadFile = File(...)):
    """
    Reconhece o rosto na foto enviada e registra a entrada ou saída da pessoa.
    Endpoint independente para integrações que queiram registrar ponto sem
    passar pelo /faces/recognize (ex: um totem dedicado).
    """
    emb = fe.get_single_embedding(await file.read())
    if emb is None:
        raise HTTPException(status_code=422, detail="Nenhum rosto detectado na imagem.")

    cpf, nome, sim = store.identify(emb)
    if cpf is None:
        raise HTTPException(
            status_code=404,
            detail=f"Rosto não reconhecido (melhor similaridade: {round(sim, 4)}).",
        )

    acesso = _registrar_ponto(cpf)

    return {
        "message": (
            f"Ponto já registrado recentemente para '{nome}', ignorando duplicidade."
            if acesso["skipped"]
            else f"{acesso['tipo'].capitalize()} registrada automaticamente para '{nome}'."
        ),
        "cpf": cpf,
        "nome": nome,
        "similarity": round(sim, 4),
        **acesso,
    }


@app.get("/access/status/{cpf}")
def access_status(cpf: str):
    """Diz se a pessoa está 'dentro' ou 'fora' com base no último log."""
    ultimo = db.get_last_log(cpf)
    if ultimo is None:
        return {"cpf": cpf, "status": "sem_registro"}
    return {
        "cpf": cpf,
        "status": "dentro" if ultimo["tipo"] == "entrada" else "fora",
        "ultimo_tipo": ultimo["tipo"],
        "ultima_data_hora": ultimo["data_hora"].isoformat(),
    }


@app.get("/access/logs", response_model=List[LogRecord])
def access_logs(
    cpf: Optional[str] = Query(None, description="Filtrar pelo CPF da pessoa"),
    tipo: Optional[Literal["entrada", "saida"]] = Query(None, description="Filtrar por tipo"),
    date_from: Optional[date] = Query(None, description="Data inicial (YYYY-MM-DD)"),
    date_to: Optional[date] = Query(None, description="Data final (YYYY-MM-DD)"),
    limit: int = Query(200, le=1000),
):
    """Lista os registros de entrada/saída, com filtros opcionais."""
    rows = db.list_logs(cpf=cpf, tipo=tipo, date_from=date_from, date_to=date_to, limit=limit)
    return [
        {
            "id": r["id"],
            "cpf": r["cpf"],
            "nome": r["nome"],
            "tipo": r["tipo"],
            "data_hora": r["data_hora"].isoformat(),
        }
        for r in rows
    ]