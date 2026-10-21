"""
config.py

Configurações da aplicação lidas de variáveis de ambiente (.env).

Esta API não cadastra ninguém: as pessoas (e suas fotos em base64) vêm da
tabela `pessoa`, alimentada por outra API. Aqui só gravamos os logs de
entrada/saída no Cloudflare D1 (SQLite serverless, via API REST).
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


# --- Cloudflare D1 ---
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_D1_DATABASE_ID = os.getenv("CLOUDFLARE_D1_DATABASE_ID", "")
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")

# Timeout (segundos) de cada chamada HTTP ao D1.
D1_TIMEOUT_SECONDS = float(os.getenv("D1_TIMEOUT_SECONDS", "15"))

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
AUTO_CREATE_TABLES = _env_bool("AUTO_CREATE_TABLES", "true")

# --- Regras de acesso ---
# Tempo mínimo (segundos) entre dois registros de ponto da MESMA pessoa.
# Evita contar entrada/saída duplicada se a câmera mandar vários frames seguidos.
ACCESS_COOLDOWN_SECONDS = float(os.getenv("ACCESS_COOLDOWN_SECONDS", "4"))

# Limiar de similaridade de cosseno para considerar "é a mesma pessoa".
# Com ArcFace (buffalo_l), 0.35–0.45 costuma ser um bom ponto de corte.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.40"))

# De quanto em quanto tempo (segundos) a thread de fundo relê a tabela
# `pessoa` para descobrir gente nova ou foto trocada.
FACE_CACHE_TTL_SECONDS = float(os.getenv("FACE_CACHE_TTL_SECONDS", "60"))

# --- Motor facial ---
# Pacote de modelos do InsightFace. `buffalo_l` é o mais preciso (ArcFace
# ResNet-50). `buffalo_s` é bem mais leve, mas o limiar precisa ser recalibrado.
FACE_MODEL = os.getenv("FACE_MODEL", "buffalo_l")

# Resolução (lado, em px) em que o detector trabalha. Numa catraca o rosto
# fica perto da câmera, então 320 basta e é ~4x mais rápido que 640.
# Suba para 480/640 se a câmera ficar longe e rostos pequenos não forem achados.
FACE_DET_SIZE = int(os.getenv("FACE_DET_SIZE", "320"))

# Tamanho máximo (bytes) da imagem aceita em POST /faces/recognize.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

# --- Anti-spoofing (rosto vivo) ---
# Com True, o addon `liveness` do InsightFace avalia se o rosto na frente da
# câmera é de uma pessoa real ou de uma foto/tela/impressão ANTES de reconhecer.
# Rosto reprovado não passa pelo ArcFace e a resposta é `rosto_nao_vivo`.
# Na primeira execução o InsightFace baixa `liveness.onnx` (~1,5 MB) para
# ~/.insightface/addons/ (para uso offline, copie o arquivo para lá antes).
FACE_LIVENESS = _env_bool("FACE_LIVENESS", "true")

# Probabilidade mínima (0 a 1) de ser rosto vivo. Abaixo disso o acesso é
# recusado. 0.8 é o padrão do InsightFace; baixe para 0.6–0.7 se pessoas reais
# estiverem sendo recusadas (iluminação ruim), suba se foto ainda passar.
LIVENESS_THRESHOLD = float(os.getenv("LIVENESS_THRESHOLD", "0.8"))


def missing_d1_settings() -> list:
    """Nomes das variáveis do Cloudflare D1 que ainda não foram preenchidas."""
    required = {
        "CLOUDFLARE_ACCOUNT_ID": CLOUDFLARE_ACCOUNT_ID,
        "CLOUDFLARE_D1_DATABASE_ID": CLOUDFLARE_D1_DATABASE_ID,
        "CLOUDFLARE_API_TOKEN": CLOUDFLARE_API_TOKEN,
    }
    return [name for name, value in required.items() if not value]
