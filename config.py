"""
config.py

Configurações da aplicação lidas de variáveis de ambiente (.env).

Nesta versão a API NÃO cadastra ninguém: as pessoas (e suas fotos em base64)
vêm da tabela `pessoa`, alimentada por outra API. Aqui só gravamos os logs
de entrada/saída.

O banco é o Cloudflare D1 (SQLite serverless), acessado pela API REST da
Cloudflare.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Cloudflare D1 ---
# Account ID, Database ID (UUID) e API Token com permissão "D1 Edit".
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_D1_DATABASE_ID = os.getenv("CLOUDFLARE_D1_DATABASE_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")

# Timeout (segundos) de cada chamada HTTP ao D1.
D1_TIMEOUT_SECONDS = float(os.getenv("D1_TIMEOUT_SECONDS", "15"))

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


def missing_d1_settings() -> list:
    """Nomes das variáveis do Cloudflare D1 que ainda não foram preenchidas."""
    required = {
        "CLOUDFLARE_ACCOUNT_ID": CLOUDFLARE_ACCOUNT_ID,
        "CLOUDFLARE_D1_DATABASE_ID": CLOUDFLARE_D1_DATABASE_ID,
        "CLOUDFLARE_API_TOKEN": CLOUDFLARE_API_TOKEN,
    }
    return [name for name, value in required.items() if not value]