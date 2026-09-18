"""
face_store.py
Cache em memória dos embeddings faciais das pessoas cadastradas.

Como o cadastro é feito por OUTRA API (que grava a foto em base64 na tabela
`pessoa`), esta API precisa converter essas fotos em embeddings para poder
reconhecer alguém. Calcular embedding é caro, então:

  - o resultado fica em memória, indexado por CPF;
  - a tabela `pessoa` é relida a cada FACE_CACHE_TTL_SECONDS;
  - só recalcula o embedding de quem é novo ou trocou de foto (comparação
    por hash da string base64);
  - quem some da tabela some do cache.

Também é possível forçar a releitura chamando POST /faces/reload.
"""

import hashlib
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
import database as db
import face_engine as fe

logger = logging.getLogger("face_store")

_lock = threading.RLock()
_cache: Dict[str, dict] = {}        # cpf -> {nome, foto_hash, embedding}
_invalidas: Dict[str, str] = {}     # cpf -> motivo de não ter virado embedding
_last_refresh: float = 0.0


def _hash(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8", errors="ignore")).hexdigest()


def refresh(force: bool = False) -> dict:
    """
    Relê a tabela `pessoa` e atualiza o cache de embeddings.
    Com force=True, recalcula o embedding de todo mundo.
    Retorna um resumo do que aconteceu.
    """
    global _last_refresh

    with _lock:
        inicio = time.time()
        pessoas = db.list_pessoas_com_foto()

        novo_cache: Dict[str, dict] = {}
        novas_invalidas: Dict[str, str] = {}
        criados = reaproveitados = falhas = 0

        for pessoa in pessoas:
            cpf = str(pessoa["cpf"])
            nome = pessoa["nome"]
            foto = pessoa["foto"]
            foto_hash = _hash(foto)

            antigo = _cache.get(cpf)
            if not force and antigo is not None and antigo["foto_hash"] == foto_hash:
                antigo["nome"] = nome  # o nome pode ter mudado
                novo_cache[cpf] = antigo
                reaproveitados += 1
                continue

            try:
                embedding = fe.embedding_from_base64(foto)
            except ValueError as exc:
                novas_invalidas[cpf] = str(exc)
                falhas += 1
                logger.warning("Foto inválida para o CPF %s: %s", cpf, exc)
                continue
            except Exception as exc:  # imagem corrompida, formato não suportado...
                novas_invalidas[cpf] = f"Erro ao ler a imagem: {exc}"
                falhas += 1
                logger.warning("Erro ao processar a foto do CPF %s: %s", cpf, exc)
                continue

            if embedding is None:
                novas_invalidas[cpf] = "Nenhum rosto detectado na foto cadastrada."
                falhas += 1
                logger.warning("Nenhum rosto detectado na foto do CPF %s.", cpf)
                continue

            novo_cache[cpf] = {
                "cpf": cpf,
                "nome": nome,
                "foto_hash": foto_hash,
                "embedding": embedding,
            }
            criados += 1

        _cache.clear()
        _cache.update(novo_cache)
        _invalidas.clear()
        _invalidas.update(novas_invalidas)
        _last_refresh = time.time()

        resumo = {
            "pessoas_na_tabela": len(pessoas),
            "embeddings_em_cache": len(_cache),
            "novos_ou_atualizados": criados,
            "reaproveitados": reaproveitados,
            "fotos_com_problema": falhas,
            "duracao_segundos": round(time.time() - inicio, 3),
        }
        logger.info("Cache de rostos atualizado: %s", resumo)
        return resumo


def ensure_loaded() -> None:
    """Atualiza o cache se ele nunca foi carregado ou se o TTL expirou."""
    if time.time() - _last_refresh >= config.FACE_CACHE_TTL_SECONDS:
        refresh()


def all_embeddings() -> List[Tuple[str, str, np.ndarray]]:
    """Retorna [(cpf, nome, embedding), ...] com o cache já atualizado."""
    ensure_loaded()
    with _lock:
        return [(v["cpf"], v["nome"], v["embedding"]) for v in _cache.values()]


def identify(embedding: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
    """
    Compara o embedding recebido com todos os cadastrados.
    Retorna (cpf, nome, similaridade) do melhor match, ou (None, None, sim)
    se ninguém passar do limiar de similaridade.
    """
    melhor_cpf = melhor_nome = None
    melhor_sim = -1.0

    for cpf, nome, conhecido in all_embeddings():
        sim = fe.cosine_similarity(embedding, conhecido)
        if sim > melhor_sim:
            melhor_cpf, melhor_nome, melhor_sim = cpf, nome, sim

    if melhor_sim < config.SIMILARITY_THRESHOLD:
        return None, None, melhor_sim
    return melhor_cpf, melhor_nome, melhor_sim


def status() -> dict:
    """Informações do cache, úteis para diagnóstico."""
    with _lock:
        return {
            "embeddings_em_cache": len(_cache),
            "ultima_atualizacao": (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_last_refresh))
                if _last_refresh
                else None
            ),
            "ttl_segundos": config.FACE_CACHE_TTL_SECONDS,
            "limiar_similaridade": config.SIMILARITY_THRESHOLD,
            "fotos_com_problema": [
                {"cpf": cpf, "motivo": motivo} for cpf, motivo in _invalidas.items()
            ],
        }
