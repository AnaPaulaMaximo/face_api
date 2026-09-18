# Face API — reconhecimento facial + registro de ponto

API REST em FastAPI que reconhece rostos com InsightFace (RetinaFace + ArcFace)
e grava **apenas os logs de entrada/saída** no PostgreSQL.

O **cadastro não é feito aqui**. Outra API cadastra as pessoas na tabela
`pessoa`, salvando a foto como **string base64**. Esta API lê essas fotos,
gera os embeddings faciais e usa isso para reconhecer quem está na câmera.

```
 API de cadastro  ──►  tabela pessoa (cpf, nome, foto base64)
                                  │ (leitura)
                                  ▼
 câmera ──► POST /faces/recognize ──►  tabela logs (id, pessoa_cpf, tipo, data_hora)
```

## Estrutura do banco

```sql
CREATE TABLE pessoa (          -- mantida pela OUTRA API (aqui é só leitura)
    cpf  VARCHAR(14) PRIMARY KEY,
    nome TEXT NOT NULL,
    foto TEXT                  -- imagem em base64 (aceita "data:image/jpeg;base64,...")
);

CREATE TABLE logs (            -- escrita por esta API
    id         SERIAL PRIMARY KEY,
    pessoa_cpf VARCHAR(14) NOT NULL REFERENCES pessoa(cpf) ON DELETE CASCADE,
    tipo       TEXT NOT NULL CHECK (tipo IN ('entrada','saida')),
    data_hora  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Se os nomes das colunas na sua base forem outros, não precisa mexer no código:
ajuste `PESSOA_TABLE`, `PESSOA_PK`, `PESSOA_NOME_COL`, `PESSOA_FOTO_COL`,
`LOGS_TABLE`, `LOGS_FK_COL` e `LOGS_DATETIME_COL` no `.env`.

## Como rodar

```bash
cp .env.example .env          # ajuste host/porta/senha do Postgres
docker compose up -d          # sobe um Postgres na porta 5433 (opcional)

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Abra `http://localhost:8000` para a tela com câmera.
Documentação automática em `http://localhost:8000/docs`.

Na primeira execução o InsightFace baixa o modelo `buffalo_l` (~300 MB).

## Endpoints

| Método | Rota | O que faz |
|---|---|---|
| GET | `/health` | healthcheck |
| GET | `/pessoas` | lista as pessoas da tabela `pessoa` (sem o base64) |
| GET | `/pessoas/{cpf}/foto` | devolve a foto do banco já decodificada (imagem) |
| POST | `/faces/recognize` | identifica o(s) rosto(s) e **grava o log** de entrada/saída |
| POST | `/faces/verify` | compara duas fotos enviadas (file1, file2) |
| GET | `/faces/cache` | quantos rostos estão em memória e quais fotos deram problema |
| POST | `/faces/reload` | força reler a tabela `pessoa` e recalcular os embeddings |
| POST | `/access/register` | reconhece + registra ponto (para totem dedicado) |
| GET | `/access/status/{cpf}` | diz se a pessoa está `dentro`, `fora` ou `sem_registro` |
| GET | `/access/logs` | lista os logs (filtros: `cpf`, `tipo`, `date_from`, `date_to`, `limit`) |

Exemplo de resposta do `/faces/recognize`:

```json
{
  "faces_found": 1,
  "results": [
    {
      "cpf": "12345678900",
      "nome": "Fulano",
      "reconhecido": true,
      "similarity": 0.7412,
      "bbox": [120.0, 88.0, 310.0, 340.0],
      "det_score": 0.93,
      "access": {
        "skipped": false,
        "log_id": 42,
        "tipo": "entrada",
        "data_hora": "2026-01-20T13:05:11.204Z"
      }
    }
  ]
}
```

## Como funciona o cache de rostos

Converter base64 → embedding é caro, então os embeddings ficam em memória
(`face_store.py`), indexados por CPF:

- a tabela `pessoa` é relida a cada `FACE_CACHE_TTL_SECONDS` (padrão 60s);
- só recalcula o embedding de quem é novo ou trocou de foto (hash do base64);
- quem sai da tabela sai do cache;
- quer efeito imediato após um cadastro? chame `POST /faces/reload`.

Fotos que não são base64 válido, ou nas quais nenhum rosto é detectado, ficam
listadas em `GET /faces/cache` com o motivo — bom para depurar cadastro ruim.

## Entrada ou saída: como é decidido

O tipo alterna com base no último log da pessoa: sem log anterior ou último
igual a `saida` → grava `entrada`; caso contrário → `saida`. Registros da mesma
pessoa dentro de `ACCESS_COOLDOWN_SECONDS` são ignorados (`skipped: true`),
para a câmera não gerar vários logs com frames seguidos.

## Testando sem a API de cadastro

```bash
python seed_pessoa.py 12345678900 "Fulano de Tal" foto_fulano.jpg
curl -X POST http://localhost:8000/faces/reload
python test_client.py
```

## Arquivos

| Arquivo | Papel |
|---|---|
| `main.py` | endpoints FastAPI |
| `face_engine.py` | InsightFace + decodificação do base64 + similaridade |
| `face_store.py` | cache em memória dos embeddings vindos da tabela `pessoa` |
| `database.py` | acesso ao PostgreSQL (lê `pessoa`, escreve `logs`) |
| `config.py` | configurações via `.env` |
| `static/index.html` | tela com câmera (reconhecer + ver pessoas e logs) |
| `seed_pessoa.py` | utilitário de teste para popular a tabela `pessoa` |
