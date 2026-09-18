"""
database.py
Acesso ao PostgreSQL.

Duas tabelas:

  pessoa (mantida por OUTRA API — aqui é só leitura)
      cpf   PK
      nome
      foto  TEXT  -> imagem em base64 (aceita data URI "data:image/jpeg;base64,...")

  logs (escrita por esta API)
      id         PK
      pessoa_cpf FK -> pessoa(cpf)
      tipo       'entrada' | 'saida'
      data_hora  TIMESTAMPTZ

Usa psycopg2 com um pool de conexões simples.
"""

import logging
from contextlib import contextmanager
from datetime import date
from typing import List, Optional

import psycopg2
import psycopg2.pool
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

import config

logger = logging.getLogger("database")

_POOL: Optional[psycopg2.pool.SimpleConnectionPool] = None

# Identificadores usados nas queries (vêm do config, então são "montados" com
# psycopg2.sql.Identifier para não virar concatenação de string crua).
_PESSOA = sql.Identifier(config.PESSOA_TABLE)
_CPF = sql.Identifier(config.PESSOA_PK)
_NOME = sql.Identifier(config.PESSOA_NOME_COL)
_FOTO = sql.Identifier(config.PESSOA_FOTO_COL)
_LOGS = sql.Identifier(config.LOGS_TABLE)
_FK = sql.Identifier(config.LOGS_FK_COL)
_DT = sql.Identifier(config.LOGS_DATETIME_COL)


def _get_pool() -> psycopg2.pool.SimpleConnectionPool:
    global _POOL
    if _POOL is None:
        _POOL = psycopg2.pool.SimpleConnectionPool(1, 10, dsn=config.get_dsn())
    return _POOL


@contextmanager
def get_conn():
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def init_db():
    """
    Verifica a conexão e, se AUTO_CREATE_TABLES estiver ligado, cria as tabelas
    que ainda não existirem. A tabela `pessoa` é criada apenas para facilitar o
    ambiente de desenvolvimento — em produção quem cria/alimenta ela é a API
    de cadastro.
    """
    if not config.AUTO_CREATE_TABLES:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        logger.info("Conexão com o PostgreSQL OK (AUTO_CREATE_TABLES desligado).")
        return

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {pessoa} (
                        {cpf}  VARCHAR(14) PRIMARY KEY,
                        {nome} TEXT NOT NULL,
                        {foto} TEXT
                    )
                    """
                ).format(pessoa=_PESSOA, cpf=_CPF, nome=_NOME, foto=_FOTO)
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {logs} (
                        id SERIAL PRIMARY KEY,
                        {fk} VARCHAR(14) NOT NULL
                            REFERENCES {pessoa}({cpf}) ON DELETE CASCADE,
                        tipo TEXT NOT NULL CHECK (tipo IN ('entrada', 'saida')),
                        {dt} TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                ).format(logs=_LOGS, fk=_FK, pessoa=_PESSOA, cpf=_CPF, dt=_DT)
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS idx_logs_pessoa ON {logs} ({fk})"
                ).format(logs=_LOGS, fk=_FK)
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS idx_logs_data_hora ON {logs} ({dt})"
                ).format(logs=_LOGS, dt=_DT)
            )
        conn.commit()
    logger.info("Tabelas verificadas/criadas no PostgreSQL.")


# ---------------------------------------------------------------------------
# Pessoas (somente leitura — o cadastro é feito por outra API)
# ---------------------------------------------------------------------------

def list_pessoas() -> List[dict]:
    """Lista cpf e nome de todas as pessoas (sem trazer a foto, que é pesada)."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                sql.SQL(
                    "SELECT {cpf} AS cpf, {nome} AS nome FROM {pessoa} ORDER BY {nome}"
                ).format(cpf=_CPF, nome=_NOME, pessoa=_PESSOA)
            )
            return [dict(r) for r in cur.fetchall()]


def list_pessoas_com_foto() -> List[dict]:
    """Lista cpf, nome e foto (base64) de todas as pessoas que têm foto."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT {cpf} AS cpf, {nome} AS nome, {foto} AS foto
                    FROM {pessoa}
                    WHERE {foto} IS NOT NULL AND {foto} <> ''
                    """
                ).format(cpf=_CPF, nome=_NOME, foto=_FOTO, pessoa=_PESSOA)
            )
            return [dict(r) for r in cur.fetchall()]


def get_pessoa(cpf: str) -> Optional[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT {cpf} AS cpf, {nome} AS nome, {foto} AS foto
                    FROM {pessoa} WHERE {cpf} = %s
                    """
                ).format(cpf=_CPF, nome=_NOME, foto=_FOTO, pessoa=_PESSOA),
                (cpf,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


# ---------------------------------------------------------------------------
# Logs de entrada / saída
# ---------------------------------------------------------------------------

def add_log(cpf: str, tipo: str) -> dict:
    """Insere um registro de entrada/saída e devolve a linha criada."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {logs} ({fk}, tipo)
                    VALUES (%s, %s)
                    RETURNING id, {fk} AS cpf, tipo, {dt} AS data_hora
                    """
                ).format(logs=_LOGS, fk=_FK, dt=_DT),
                (cpf, tipo),
            )
            row = cur.fetchone()
        conn.commit()
        return dict(row)


def get_last_log(cpf: str) -> Optional[dict]:
    """Último registro (entrada ou saída) de uma pessoa, se houver."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT id, {fk} AS cpf, tipo, {dt} AS data_hora
                    FROM {logs}
                    WHERE {fk} = %s
                    ORDER BY {dt} DESC
                    LIMIT 1
                    """
                ).format(logs=_LOGS, fk=_FK, dt=_DT),
                (cpf,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def list_logs(
    cpf: Optional[str] = None,
    tipo: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    limit: int = 200,
) -> List[dict]:
    """Lista os logs (com o nome da pessoa via JOIN), com filtros opcionais."""
    query = sql.SQL(
        """
        SELECT l.id, l.{fk} AS cpf, p.{nome} AS nome, l.tipo, l.{dt} AS data_hora
        FROM {logs} l
        LEFT JOIN {pessoa} p ON p.{cpf_col} = l.{fk}
        WHERE 1=1
        """
    ).format(logs=_LOGS, fk=_FK, dt=_DT, pessoa=_PESSOA, cpf_col=_CPF, nome=_NOME)

    params: list = []
    if cpf:
        query = query + sql.SQL(" AND l.{fk} = %s").format(fk=_FK)
        params.append(cpf)
    if tipo:
        query = query + sql.SQL(" AND l.tipo = %s")
        params.append(tipo)
    if date_from:
        query = query + sql.SQL(" AND l.{dt} >= %s").format(dt=_DT)
        params.append(date_from)
    if date_to:
        # inclui o dia inteiro informado em date_to
        query = query + sql.SQL(
            " AND l.{dt} < (%s::date + INTERVAL '1 day')"
        ).format(dt=_DT)
        params.append(date_to)

    query = query + sql.SQL(" ORDER BY l.{dt} DESC LIMIT %s").format(dt=_DT)
    params.append(limit)

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, tuple(params))
            return [dict(r) for r in cur.fetchall()]
