"""
face_store.py

Cache em memória dos embeddings faciais das pessoas cadastradas.

O cadastro é feito por OUTRA API (que grava a foto em base64 na tabela
`pessoa`). Esta API converte essas fotos em embeddings uma única vez e guarda
tudo numa matriz NumPy (N x 512): identificar alguém vira uma multiplicação
matriz-vetor, em microssegundos, independente do tamanho do cadastro.

Como a atualização funciona (sem pesar nas requisições):

  - uma thread de fundo relê a tabela `pessoa` a cada FACE_CACHE_TTL_SECONDS;
  - a releitura pede ao banco só uma "assinatura" barata de cada foto
    (tamanho + início + fim do base64), não o base64 inteiro;
  - o base64 só é baixado, e o embedding só é recalculado, para quem é novo
    ou cuja assinatura mudou;
  - quem some da tabela some do cache.

POST /faces/reload força uma releitura completa (recalcula todo mundo).
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
import database as db
import face_engine as fe

logger = logging.getLogger("face_store")

_EMB_DIM = 512

_refresh_lock = threading.Lock()   # só um refresh por vez
_meta: Dict[str, dict] = {}        # cpf -> {nome, assinatura, embedding}
_invalidas: Dict[str, str] = {}    # cpf -> motivo de não ter virado embedding
_last_refresh: float = 0.0

# Snapshot imutável consultado pelo reconhecimento. Trocado atomicamente
# (atribuição de referência é atômica no CPython), então identify() não
# precisa de lock.
_cpfs: List[str] = []
_nomes: List[str] = []
_matrix: np.ndarray = np.zeros((0, _EMB_DIM), dtype=np.float32)

_stop = threading.Event()
_thread: Optional[threading.Thread] = None


def refresh(force: bool = False) -> dict:
    """
    Sincroniza o cache com a tabela `pessoa`. Com force=True recalcula o
    embedding de todo mundo. Retorna um resumo do que aconteceu.
    """
    global _last_refresh, _cpfs, _nomes, _matrix

    with _refresh_lock:
        inicio = time.time()
        pessoas = db.list_pessoas_assinaturas()

        novo_meta: Dict[str, dict] = {}
        novas_invalidas: Dict[str, str] = {}
        pendentes: Dict[str, dict] = {}
        reaproveitados = 0

        for p in pessoas:
            cpf = p["cpf"]
            antigo = _meta.get(cpf)
            if not force and antigo is not None and antigo["assinatura"] == p["assinatura"]:
                # mesma foto de antes: reaproveita (o nome pode ter mudado)
                novo_meta[cpf] = {**antigo, "nome": p["nome"]}
                if antigo["embedding"] is None:
                    novas_invalidas[cpf] = _invalidas.get(cpf, "Foto inválida.")
                else:
                    reaproveitados += 1
            else:
                pendentes[cpf] = p

        fotos = db.get_fotos(list(pendentes)) if pendentes else {}
        criados = 0
        for cpf, p in pendentes.items():
            motivo = None
            try:
                embedding = fe.embedding_from_base64(fotos.get(cpf, ""))
                if embedding is None:
                    motivo = "Nenhum rosto detectado na foto cadastrada."
            except ValueError as exc:  # base64 inválido ou imagem indecodificável
                motivo = str(exc)
            except Exception as exc:  # qualquer outra falha do modelo
                motivo = f"Erro ao processar a foto: {exc}"

            if motivo:
                # guarda a assinatura mesmo assim, para não baixar a mesma foto ruim a cada ciclo
                novas_invalidas[cpf] = motivo
                novo_meta[cpf] = {"nome": p["nome"], "assinatura": p["assinatura"], "embedding": None}
                logger.warning("Foto do CPF %s ignorada: %s", cpf, motivo)
                continue

            novo_meta[cpf] = {"nome": p["nome"], "assinatura": p["assinatura"], "embedding": embedding}
            criados += 1

        validos = [(cpf, m) for cpf, m in novo_meta.items() if m["embedding"] is not None]
        nova_matrix = (
            np.stack([m["embedding"] for _, m in validos]).astype(np.float32)
            if validos
            else np.zeros((0, _EMB_DIM), dtype=np.float32)
        )

        _meta.clear()
        _meta.update(novo_meta)
        _invalidas.clear()
        _invalidas.update(novas_invalidas)
        _cpfs = [cpf for cpf, _ in validos]
        _nomes = [m["nome"] for _, m in validos]
        _matrix = nova_matrix
        _last_refresh = time.time()

        resumo = {
            "pessoas_na_tabela": len(pessoas),
            "embeddings_em_cache": len(_cpfs),
            "novos_ou_atualizados": criados,
            "reaproveitados": reaproveitados,
            "fotos_com_problema": len(_invalidas),
            "duracao_segundos": round(time.time() - inicio, 3),
        }
        logger.info("Cache de rostos atualizado: %s", resumo)
        return resumo


def identify(embedding: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
    """
    Compara o embedding (já L2-normalizado) com todos os cadastrados de uma
    vez só. Retorna (cpf, nome, similaridade) do melhor match, ou
    (None, None, melhor_similaridade) se ninguém passar do limiar.
    """
    cpfs, nomes, matrix = _cpfs, _nomes, _matrix  # snapshot consistente
    if matrix.shape[0] == 0:
        return None, None, 0.0

    scores = matrix @ embedding
    i = int(np.argmax(scores))
    melhor = float(scores[i])
    if melhor < config.SIMILARITY_THRESHOLD:
        return None, None, melhor
    return cpfs[i], nomes[i], melhor


def status() -> dict:
    """Informações do cache, úteis para diagnóstico."""
    return {
        "embeddings_em_cache": len(_cpfs),
        "ultima_atualizacao": (
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_last_refresh)) if _last_refresh else None
        ),
        "ttl_segundos": config.FACE_CACHE_TTL_SECONDS,
        "limiar_similaridade": config.SIMILARITY_THRESHOLD,
        "det_size": config.FACE_DET_SIZE,
        "fotos_com_problema": [{"cpf": cpf, "motivo": motivo} for cpf, motivo in _invalidas.items()],
    }


# ---------------------------------------------------------------------------
# Atualização em segundo plano
# ---------------------------------------------------------------------------

def _loop():
    while not _stop.wait(config.FACE_CACHE_TTL_SECONDS):
        try:
            refresh()
        except Exception as exc:  # banco fora do ar não pode matar a thread
            logger.warning("Falha ao atualizar o cache de rostos: %s", exc)


def start_background_refresh() -> None:
    """Inicia a thread que mantém o cache sincronizado com a tabela `pessoa`."""
    global _thread
    if _thread is None or not _thread.is_alive():
        _stop.clear()
        _thread = threading.Thread(target=_loop, name="face-store-refresh", daemon=True)
        _thread.start()


def stop_background_refresh() -> None:
    _stop.set()
