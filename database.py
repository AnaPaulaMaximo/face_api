"""
database.py

Acesso ao Cloudflare D1 (SQLite serverless) pela API REST da Cloudflare.

pessoa (mantida por OUTRA API — aqui é só leitura)
    cpf   PK
    nome
    foto  TEXT -> imagem em base64 (aceita data URI "data:image/jpeg;base64,...")

logs (escrita por esta API)
    id          PK
    pessoa_cpf  FK -> pessoa(cpf)
    tipo        'entrada' | 'saida'
    data_hora   TEXT em ISO 8601 UTC (ex: 2026-01-20T13:05:11.204Z)

Cada chamada ao D1 custa ~200-300 ms de rede, então as consultas foram
desenhadas para o caminho quente (reconhecer + registrar) não depender delas:
o cache de rostos é atualizado por uma thread de fundo (`face_store.py`) e o
último log de cada pessoa fica em memória (`access.py`).
"""

import logging
import re
import threading
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config

logger = logging.getLogger("database")

_API_BASE = "https://api.cloudflare.com/client/v4"
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# O D1 aceita no máximo 100 parâmetros por query.
_D1_MAX_PARAMS = 100

_SESSION: Optional[requests.Session] = None
_SESSION_LOCK = threading.Lock()


class D1Error(RuntimeError):
    """Erro devolvido pela API do Cloudflare D1 (ou falha de comunicação)."""


def _ident(name: str) -> str:
    """Valida um identificador (tabela/coluna) vindo do config: só letras, números e `_`."""
    if not _IDENT_RE.match(name):
        raise ValueError(f"Identificador SQL inválido: {name!r}")
    return f'"{name}"'


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
            # Conexão keep-alive + retry curto para instabilidade momentânea do D1.
            retry = Retry(
                total=2,
                backoff_factor=0.3,
                status_forcelist=(502, 503, 504),
                allowed_methods=frozenset({"POST"}),
            )
            session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=4))
            _SESSION = session
        return _SESSION


def _query(sql: str, params: Optional[list] = None) -> List[dict]:
    """Executa UMA instrução SQL no D1 e devolve as linhas como lista de dicts."""
    missing = config.missing_d1_settings()
    if missing:
        raise D1Error(
            "Configuração do Cloudflare D1 incompleta. Defina no .env: " + ", ".join(missing)
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


def format_dt(dt: datetime) -> str:
    """Formata no mesmo padrão do DEFAULT da coluna (ISO 8601 UTC, milissegundos, sufixo Z)."""
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _fix_log_row(row: dict) -> dict:
    row = dict(row)
    row["data_hora"] = _parse_dt(row.get("data_hora"))
    return row


def _chunks(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


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

def list_pessoas_assinaturas() -> List[dict]:
    """
    Lista cpf, nome e uma "assinatura" barata da foto (tamanho + início + fim
    do base64) de todas as pessoas com foto. Serve para descobrir quem é novo
    ou trocou de foto SEM baixar os base64 inteiros a cada atualização.
    """
    rows = _query(
        f"""
        SELECT {_CPF} AS cpf, {_NOME} AS nome,
               length({_FOTO}) AS tam,
               substr({_FOTO}, 1, 64) AS ini,
               substr({_FOTO}, -64) AS fim
        FROM {_PESSOA}
        WHERE {_FOTO} IS NOT NULL AND {_FOTO} <> ''
        """
    )
    return [
        {
            "cpf": str(r["cpf"]),
            "nome": r["nome"],
            "assinatura": f"{r['tam']}:{r['ini']}:{r['fim']}",
        }
        for r in rows
    ]


def get_fotos(cpfs: List[str]) -> Dict[str, str]:
    """Devolve {cpf: foto_base64} só das pessoas pedidas (em lotes, pelo limite de parâmetros do D1)."""
    fotos: Dict[str, str] = {}
    for lote in _chunks(list(cpfs), _D1_MAX_PARAMS):
        marcadores = ", ".join("?" * len(lote))
        rows = _query(
            f"SELECT {_CPF} AS cpf, {_FOTO} AS foto FROM {_PESSOA} WHERE {_CPF} IN ({marcadores})",
            lote,
        )
        for r in rows:
            fotos[str(r["cpf"])] = r["foto"]
    return fotos


# ---------------------------------------------------------------------------
# Logs de entrada / saída
# ---------------------------------------------------------------------------

def add_log(cpf: str, tipo: str, data_hora: datetime) -> dict:
    """Insere um registro de entrada/saída (com a data/hora decidida pela API) e devolve a linha."""
    rows = _query(
        f"""
        INSERT INTO {_LOGS} ({_FK}, tipo, {_DT})
        VALUES (?, ?, ?)
        RETURNING id, {_FK} AS cpf, tipo, {_DT} AS data_hora
        """,
        [cpf, tipo, format_dt(data_hora)],
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


def get_last_logs() -> Dict[str, dict]:
    """Último registro de CADA pessoa, numa única query: {cpf: {tipo, data_hora}}."""
    rows = _query(
        f"""
        SELECT cpf, tipo, data_hora FROM (
            SELECT {_FK} AS cpf, tipo, {_DT} AS data_hora,
                   ROW_NUMBER() OVER (PARTITION BY {_FK} ORDER BY {_DT} DESC, id DESC) AS rn
            FROM {_LOGS}
        ) WHERE rn = 1
        """
    )
    return {str(r["cpf"]): _fix_log_row(r) for r in rows}


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
