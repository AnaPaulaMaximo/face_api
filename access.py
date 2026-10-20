"""
access.py

Decide e registra a entrada/saída de uma pessoa já reconhecida.

Regras:
  - o tipo alterna com base no último registro da pessoa: sem registro
    anterior ou último = `saida` -> grava `entrada`; senão -> `saida`;
  - registros da mesma pessoa dentro de ACCESS_COOLDOWN_SECONDS são
    ignorados (a câmera manda vários frames seguidos da mesma pessoa).

Para a catraca responder rápido, nada disso espera o banco:
  - o último registro de cada pessoa fica em memória (carregado numa única
    query na subida e atualizado a cada decisão);
  - a gravação no D1 acontece depois que a resposta já foi enviada
    (BackgroundTasks do FastAPI).
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Optional

import config
import database as db

logger = logging.getLogger("access")

_lock = threading.Lock()
_ultimos: Dict[str, Optional[dict]] = {}  # cpf -> {tipo, data_hora} | None (sem registro)


def carregar_ultimos() -> int:
    """Carrega da base o último registro de cada pessoa. Retorna quantos foram carregados."""
    ultimos = db.get_last_logs()
    with _lock:
        _ultimos.clear()
        _ultimos.update(ultimos)
    return len(ultimos)


def _ultimo(cpf: str) -> Optional[dict]:
    """Último registro da pessoa, consultando o banco só na primeira vez que o CPF aparece."""
    if cpf in _ultimos:
        return _ultimos[cpf]
    ultimo = db.get_last_log(cpf)
    _ultimos[cpf] = ultimo
    return ultimo


def decidir(cpf: str) -> dict:
    """
    Decide o registro para a pessoa reconhecida, atualizando o estado em
    memória. Não toca o banco (exceto na primeira vez que o CPF aparece).

    Retorna {"registrado": bool, "tipo": "entrada"|"saida", "data_hora": datetime}.
    Com registrado=False o tipo/data_hora são do registro anterior (cooldown).
    """
    agora = datetime.now(timezone.utc)
    with _lock:
        ultimo = _ultimo(cpf)
        if ultimo is not None:
            decorrido = (agora - ultimo["data_hora"]).total_seconds()
            if decorrido < config.ACCESS_COOLDOWN_SECONDS:
                return {"registrado": False, "tipo": ultimo["tipo"], "data_hora": ultimo["data_hora"]}

        tipo = "entrada" if ultimo is None or ultimo["tipo"] == "saida" else "saida"
        _ultimos[cpf] = {"tipo": tipo, "data_hora": agora}

    return {"registrado": True, "tipo": tipo, "data_hora": agora}


def persistir(cpf: str, tipo: str, data_hora: datetime) -> None:
    """Grava o log no D1. Roda em segundo plano, depois da resposta."""
    try:
        db.add_log(cpf=cpf, tipo=tipo, data_hora=data_hora)
    except Exception as exc:
        logger.error("Falha ao gravar log (%s, %s, %s): %s", cpf, tipo, data_hora.isoformat(), exc)
