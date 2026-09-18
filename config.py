"""
config.py
Configurações da aplicação lidas de variáveis de ambiente (.env).

Nesta versão a API NÃO cadastra ninguém: as pessoas (e suas fotos em base64)
vêm da tabela `pessoa`, alimentada por outra API. Aqui só gravamos os logs
de entrada/saída.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- PostgreSQL ---
# Você pode configurar via DATABASE_URL (ex: postgresql://user:senha@host:5432/face_api)
# ou via variáveis separadas (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD).
DATABASE_URL = os.getenv("DATABASE_URL")

PGHOST = os.getenv("PGHOST", "localhost")
PGPORT = os.getenv("PGPORT", "5432")
PGDATABASE = os.getenv("PGDATABASE", "face_api")
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "postgres")

BASE_DIR = Path(__file__).resolve().parent

# --- Nomes de tabela/coluna ---
# A tabela `pessoa` pertence à outra API. Se lá os nomes forem diferentes,
# ajuste por variável de ambiente sem precisar mexer no código.
PESSOA_TABLE = os.getenv("PESSOA_TABLE", "pessoa")
PESSOA_PK = os.getenv("PESSOA_PK", "cpf")
PESSOA_NOME_COL = os.getenv("PESSOA_NOME_COL", "nome")
PESSOA_FOTO_COL = os.getenv("PESSOA_FOTO_COL", "foto")

LOGS_TABLE = os.getenv("LOGS_TABLE", "logs")
LOGS_FK_COL = os.getenv("LOGS_FK_COL", "pessoa_cpf")
LOGS_DATETIME_COL = os.getenv("LOGS_DATETIME_COL", "data_hora")

# Se True, a API cria as tabelas caso não existam (útil em dev).
# Em produção, deixe False: quem manda no schema é a API de cadastro.
AUTO_CREATE_TABLES = os.getenv("AUTO_CREATE_TABLES", "true").lower() in ("1", "true", "yes")

# Tempo mínimo (segundos) entre dois registros de ponto da MESMA pessoa.
# Evita contar entrada/saída duplicada se a câmera mandar vários frames seguidos.
ACCESS_COOLDOWN_SECONDS = int(os.getenv("ACCESS_COOLDOWN_SECONDS", "4"))

# Limiar de similaridade de cosseno para considerar "é a mesma pessoa".
# Com ArcFace (buffalo_l), 0.35–0.45 costuma ser um bom ponto de corte.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.40"))

# De quanto em quanto tempo (segundos) a API relê a tabela `pessoa` para
# descobrir gente nova cadastrada pela outra API. Só recalcula o embedding
# de quem é novo ou teve a foto alterada.
FACE_CACHE_TTL_SECONDS = int(os.getenv("FACE_CACHE_TTL_SECONDS", "60"))


def get_dsn() -> str:
    """Retorna a string de conexão (DSN) do PostgreSQL."""
    if DATABASE_URL:
        return DATABASE_URL
    return (
        f"host={PGHOST} port={PGPORT} dbname={PGDATABASE} "
        f"user={PGUSER} password={PGPASSWORD}"
    )
