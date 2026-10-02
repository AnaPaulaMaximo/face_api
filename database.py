"""
database.py

Acesso ao Cloudflare D1 (SQLite serverless) pela API REST da Cloudflare.

Duas tabelas:

pessoa (mantida por OUTRA API — aqui é só leitura)
    cpf   PK
    nome
    foto  TEXT -> imagem em base64 (aceita data URI "data:image/jpeg;base64,...")

logs (escrita por esta API)
    id          PK
    pessoa_cpf  FK -> pessoa(cpf)
    tipo        'entrada' | 'saida'
    data_hora   TEXT em ISO 8601 UTC (ex: 2026-01-20T13:05:11.204Z)

O D1 não tem tipo de data/hora nativo, então `data_hora` é guardado como texto
ISO 8601 em UTC e convertido para `datetime` (com timezone) ao ler, mantendo
o mesmo contrato que o resto da aplicação já esperava.
"""

import logging
import re
import threading
from datetime import date, datetime, timezone
from typing import Any, List, Optional

import requests

import config

logger = logging.getLogger("database")

_API_BASE = "https://api.cloudflare.com/client/v4"
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_SESSION: Optional[requests.Session] = None
_SESSION_LOCK = threading.Lock()


class D1Error(RuntimeError):
    """Erro devolvido pela API do Cloudflare D1 (ou falha de comunicação)."""


def _ident(name: str) -> str:
    """
    Valida e protege um identificador (tabela/coluna) vindo do config.
    Só aceita letras, números e underscore, para nunca virar SQL injection.
    """
    if not _IDENT_RE.match(name):
        raise ValueError(f"Identificador SQL inválido: {name!r}")
    return f'"{name}"'


# Identificadores usados nas queries (vêm do config).
_PESSOA = _ident(config.PESSOA_TABLE)
_CPF = _ident(config.PESSOA_PK)
_NOME = _ident(config.PESSOA_NOME_COL)
_FOTO = _ident(config.PESSOA_FOTO_COL)
_LOGS = _ident(config.LOGS_TABLE)
_FK = _ident(config.LOGS_FK_COL)
_DT = _ident(config.LOGS_DATETIME_COL)


# ---------------------------------------------------------------------------
# Cliente HTTP do D1
# ---------------------------------------------------------------------------

def _get_session() -> requests.Session:
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            session = requests.Session()
            session.headers.update(
                {
                    "Authorization": f"Bearer {config.CLOUDFLARE_API_TOKEN}",
                    "Content-Type": "application/json",
                }
            )
            _SESSION = session
        return _SESSION


def _query(sql: str, params: Optional[list] = None) -> List[dict]:
    """
    Executa UMA instrução SQL no D1 e devolve as linhas como lista de dicts.
    Os parâmetros (`?`) são enviados separados do SQL, nunca concatenados.
    """
    missing = config.missing_d1_settings()
    if missing:
        raise D1Error(
            "Configuração do Cloudflare D1 incompleta. Defina no .env: "
            + ", ".join(missing)
        )

    url = (
        f"{_API_BASE}/accounts/{config.CLOUDFLARE_ACCOUNT_ID}"
        f"/d1/database/{config.CLOUDFLARE_D1_DATABASE_ID}/query"
    )
    body: dict = {"sql": sql}
    if params:
        body["params"] = [None if p is None else str(p) for p in params]

    try:
        resp = _get_session().post(url, json=body, timeout=config.D1_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise D1Error(f"Falha ao comunicar com o Cloudflare D1: {exc}") from exc

    try:
        payload = resp.json()
    except ValueError:
        payload = {}

    if not resp.ok or not payload.get("success", False):
        detalhe = payload.get("errors") or resp.text[:300]
        raise D1Error(f"Cloudflare D1 respondeu com erro (HTTP {resp.status_code}): {detalhe}")

    result = payload.get("result") or []
    if not result:
        return []
    return result[0].get("results") or []


def _parse_dt(value: Any) -> Optional[datetime]:
    """Converte o texto ISO 8601 do D1 em datetime com timezone (UTC)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00").replace(" ", "T"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _fix_log_row(row: dict) -> dict:
    row = dict(row)
    row["data_hora"] = _parse_dt(row.get("data_hora"))
    return row


# ---------------------------------------------------------------------------
# Inicialização
# ---------------------------------------------------------------------------

def init_db():
    """
    Verifica a conexão com o D1 e, se AUTO_CREATE_TABLES estiver ligado, cria
    as tabelas que ainda não existirem. A tabela `pessoa` é criada apenas para
    facilitar o ambiente de desenvolvimento — em produção quem cria/alimenta
    ela é a API de cadastro.
    """
    if not config.AUTO_CREATE_TABLES:
        _query("SELECT 1")
        logger.info("Conexão com o Cloudflare D1 OK (AUTO_CREATE_TABLES desligado).")
        return

    _query(
        f"""
        CREATE TABLE IF NOT EXISTS {_PESSOA} (
            {_CPF}  TEXT PRIMARY KEY,
            {_NOME} TEXT NOT NULL,
            {_FOTO} TEXT
        )
        """
    )
    _query(
        f"""
        CREATE TABLE IF NOT EXISTS {_LOGS} (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            {_FK} TEXT NOT NULL REFERENCES {_PESSOA}({_CPF}) ON DELETE CASCADE,
            tipo  TEXT NOT NULL CHECK (tipo IN ('entrada', 'saida')),
            {_DT} TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """
    )
    _query(f"CREATE INDEX IF NOT EXISTS idx_logs_pessoa ON {_LOGS} ({_FK})")
    _query(f"CREATE INDEX IF NOT EXISTS idx_logs_data_hora ON {_LOGS} ({_DT})")
    logger.info("Tabelas verificadas/criadas no Cloudflare D1.")


# ---------------------------------------------------------------------------
# Pessoas (somente leitura — o cadastro é feito por outra API)
# ---------------------------------------------------------------------------

def list_pessoas() -> List[dict]:
    """Lista cpf e nome de todas as pessoas (sem trazer a foto, que é pesada)."""
    return _query(
        f"SELECT {_CPF} AS cpf, {_NOME} AS nome FROM {_PESSOA} ORDER BY {_NOME}"
    )


def list_pessoas_com_foto() -> List[dict]:
    """Lista cpf, nome e foto (base64) de todas as pessoas que têm foto."""
    return _query(
        f"""
        SELECT {_CPF} AS cpf, {_NOME} AS nome, {_FOTO} AS foto
        FROM {_PESSOA}
        WHERE {_FOTO} IS NOT NULL AND {_FOTO} <> ''
        """
    )


def get_pessoa(cpf: str) -> Optional[dict]:
    rows = _query(
        f"""
        SELECT {_CPF} AS cpf, {_NOME} AS nome, {_FOTO} AS foto
        FROM {_PESSOA} WHERE {_CPF} = ?
        """,
        [cpf],
    )
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Logs de entrada / saída
# ---------------------------------------------------------------------------

def add_log(cpf: str, tipo: str) -> dict:
    """Insere um registro de entrada/saída e devolve a linha criada."""
    rows = _query(
        f"""
        INSERT INTO {_LOGS} ({_FK}, tipo)
        VALUES (?, ?)
        RETURNING id, {_FK} AS cpf, tipo, {_DT} AS data_hora
        """,
        [cpf, tipo],
    )
    return _fix_log_row(rows[0])


def get_last_log(cpf: str) -> Optional[dict]:
    """Último registro (entrada ou saída) de uma pessoa, se houver."""
    rows = _query(
        f"""
        SELECT id, {_FK} AS cpf, tipo, {_DT} AS data_hora
        FROM {_LOGS}
        WHERE {_FK} = ?
        ORDER BY {_DT} DESC, id DESC
        LIMIT 1
        """,
        [cpf],
    )
    return _fix_log_row(rows[0]) if rows else None


def list_logs(
    cpf: Optional[str] = None,
    tipo: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    limit: int = 200,
) -> List[dict]:
    """Lista os logs (com o nome da pessoa via JOIN), com filtros opcionais."""
    query = f"""
        SELECT l.id, l.{_FK} AS cpf, p.{_NOME} AS nome, l.tipo, l.{_DT} AS data_hora
        FROM {_LOGS} l
        LEFT JOIN {_PESSOA} p ON p.{_CPF} = l.{_FK}
        WHERE 1=1
    """
    params: list = []

    if cpf:
        query += f" AND l.{_FK} = ?"
        params.append(cpf)
    if tipo:
        query += " AND l.tipo = ?"
        params.append(tipo)
    if date_from:
        query += f" AND l.{_DT} >= ?"
        params.append(date_from.isoformat())
    if date_to:
        # inclui o dia inteiro informado em date_to
        query += f" AND l.{_DT} < date(?, '+1 day')"
        params.append(date_to.isoformat())

    # `limit` vem validado como int pela rota; int() garante que nada além de
    # um número entra no SQL.
    query += f" ORDER BY l.{_DT} DESC, l.id DESC LIMIT {int(limit)}"

    return [_fix_log_row(r) for r in _query(query, params)]