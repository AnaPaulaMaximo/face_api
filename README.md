# Face API — reconhecimento facial para catraca

API REST em FastAPI que recebe a foto da câmera, confere que é um **rosto
vivo** (e não uma foto no celular ou impressa), reconhece com InsightFace
(SCRFD + ArcFace) e responde **na hora** se o acesso está autorizado, gravando
o log de entrada/saída no **Cloudflare D1**.

O **cadastro não é feito aqui**. Outra API cadastra as pessoas na tabela
`pessoa`, salvando a foto como **string base64**. Esta API lê essas fotos,
gera os embeddings faciais uma única vez e usa isso para reconhecer quem está
na câmera.

```
 API de cadastro  ──►  tabela pessoa (cpf, nome, foto base64)
                                  │ (leitura, em segundo plano)
                                  ▼
 frontend ──► POST /faces/recognize ──► { autorizado, nome, tipo: entrada|saida }
                                  │ (gravação em segundo plano)
                                  ▼
                        tabela logs (id, pessoa_cpf, tipo, data_hora)
```

## Como rodar

```bash
cp .env.example .env          # preencha as credenciais do Cloudflare D1

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Na subida a API testa a conexão com o D1, cria as tabelas (se
`AUTO_CREATE_TABLES=true`), carrega e aquece o modelo, gera os embeddings de
todas as pessoas e carrega o último registro de cada uma. Só então começa a
aceitar requisições. Na primeira execução o InsightFace baixa o pacote
`buffalo_l` (~300 MB) e, com `FACE_LIVENESS=true` (padrão), o modelo de
anti-spoofing `liveness.onnx` (~1,5 MB, para `~/.insightface/addons/`). Para
rodar sem internet, copie os dois para lá antes.

Documentação interativa em `http://localhost:8000/docs` (a rota `/` redireciona para ela).

## Endpoints

| Método | Rota | O que faz |
|---|---|---|
| POST | `/faces/recognize` | recebe a foto, confere se é um **rosto vivo**, identifica e **responde autorizado/recusado**, gravando o log |
| GET | `/access/logs` | lista os logs (filtros: `cpf`, `tipo`, `date_from`, `date_to`, `limit`) |
| GET | `/health` | healthcheck (quantos rostos estão em memória e se o anti-spoofing está ligado) |
| GET | `/faces/cache` | status do cache de embeddings e quais fotos cadastradas deram problema |
| POST | `/faces/reload` | força reler a tabela `pessoa` e recalcular todos os embeddings |

### `POST /faces/recognize`

`multipart/form-data` com o campo `file` (jpg/png/webp, até `MAX_UPLOAD_BYTES`).

```js
const form = new FormData();
form.append("file", blob, "frame.jpg");
const r = await fetch("http://localhost:8000/faces/recognize", { method: "POST", body: form });
const { autorizado, nome, acesso } = await r.json();
```

Resposta quando a pessoa é reconhecida:

```json
{
  "autorizado": true,
  "motivo": null,
  "cpf": "12345678900",
  "nome": "Fulano",
  "similaridade": 0.7412,
  "live_score": 0.9758,
  "acesso": { "tipo": "entrada", "data_hora": "2026-01-20T13:05:11.204000+00:00", "registrado": true },
  "tempo_ms": 232
}
```

Quando não é:

```json
{ "autorizado": false, "motivo": "nao_reconhecido", "similaridade": 0.21, "live_score": 0.93, "tempo_ms": 235 }
{ "autorizado": false, "motivo": "rosto_nao_vivo",  "similaridade": 0.0,  "live_score": 0.01, "tempo_ms": 75 }
{ "autorizado": false, "motivo": "rosto_cortado",   "similaridade": 0.0,  "live_score": null, "tempo_ms": 65 }
{ "autorizado": false, "motivo": "nenhum_rosto",    "similaridade": 0.0,  "live_score": null, "tempo_ms": 60 }
```

| `motivo` | O que aconteceu | O que o frontend pode mostrar |
|---|---|---|
| `nenhum_rosto` | nenhum rosto detectado | "Aproxime-se da câmera" |
| `rosto_nao_vivo` | o rosto parece ser uma foto (celular, impressão, tela): `live_score` abaixo de `LIVENESS_THRESHOLD` | "Acesso negado" |
| `rosto_cortado` | o rosto está colado demais na borda ou ocupa a imagem inteira; o anti-spoofing precisa de margem ao redor para avaliar | "Afaste-se um pouco da câmera" |
| `nao_reconhecido` | rosto vivo, mas não bate com ninguém cadastrado | "Acesso negado" |

- `live_score` é a probabilidade (0 a 1) de o rosto ser de uma pessoa real.
  Vem `null` com `FACE_LIVENESS=false` ou quando não deu para avaliar.

- `acesso.registrado = false` significa que a pessoa foi reconhecida, mas já
  havia um registro dela dentro de `ACCESS_COOLDOWN_SECONDS` (frame repetido);
  `tipo`/`data_hora` são do registro anterior. O acesso continua autorizado.
- Só o **rosto principal** da imagem (o maior e mais central) é avaliado —
  na catraca é quem está na frente da câmera.
- Erros: `422` arquivo que não é imagem, `413` imagem maior que o limite,
  `503` banco indisponível.

## Entrada ou saída: como é decidido

O tipo alterna com base no último registro da pessoa: sem registro anterior
ou último igual a `saida` → grava `entrada`; caso contrário → `saida`.
Registros da mesma pessoa dentro de `ACCESS_COOLDOWN_SECONDS` são ignorados,
para a câmera não gerar vários logs com frames seguidos.

## Por que é rápido

Numa CPU comum (8 threads) o `POST /faces/recognize` responde em **~230 ms**
(era ~1,8 s), ou ~240 ms com o anti-spoofing ligado. O que foi feito:

| Antes | Agora |
|---|---|
| Carregava 5 modelos (detecção, reconhecimento, idade/gênero, landmarks 2D e 3D) e rodava todos em cada rosto | Só **detecção + reconhecimento** |
| Detector a 640 px | Detector a **320 px** (`FACE_DET_SIZE`); o rosto na catraca é grande |
| Reconhecia todos os rostos da imagem | Só o rosto principal passa pelo ArcFace (`max_num=1`) |
| Decodificava a imagem com PIL + conversão para BGR | `cv2.imdecode` direto em BGR |
| Comparava embeddings num loop Python | Uma multiplicação matriz × vetor (NumPy), independente do tamanho do cadastro |
| 2 chamadas ao D1 por requisição (último log + insert), ~250 ms cada | **Nenhuma** no caminho da resposta: último log fica em memória; o insert roda depois da resposta |
| A cada 60 s uma requisição pagava a releitura de **todas** as fotos base64 do banco | Thread de fundo compara só uma assinatura barata (tamanho + início + fim do base64) e baixa apenas fotos novas/alteradas |
| Endpoint `async` fazendo inferência no event loop (travava o servidor) | Endpoint síncrono em threadpool + lock de inferência |
| Primeira requisição lenta (alocação do ONNX Runtime) | Modelo aquecido na subida |

Frames sem rosto são rejeitados em ~60 ms (só a detecção roda).

### Custo do anti-spoofing

Medido na mesma máquina, mediana de 30 envios da mesma foto pelo
`test_client.py` (`tempo_ms` do servidor):

| Situação | `FACE_LIVENESS=false` | `FACE_LIVENESS=true` |
|---|---|---|
| rosto vivo, reconhecido | 230 ms | **~240 ms** (+10 ms) |
| foto de foto (`rosto_nao_vivo`) | 230 ms (e liberava) | **~75 ms** |
| rosto cortado (`rosto_cortado`) | 230 ms | ~65 ms |

O modelo de liveness é pequeno (entrada 80×80, 1,5 MB) e roda em ~9 ms entre
a detecção e o reconhecimento; rosto reprovado nem passa pelo ArcFace, por
isso a recusa é mais rápida que um acesso liberado. Um detalhe que importa:
a sessão ONNX do addon é criada com **uma thread e sem spin-wait**
(`face_engine.py`). Com o pool de threads padrão, as threads do liveness
ficavam girando depois da inferência e roubavam CPU do ArcFace, e o addon
custava +76 ms por requisição em vez de +12 ms.

Se a câmera ficar longe e rostos pequenos não forem detectados, suba
`FACE_DET_SIZE` para 480 ou 640 (custo medido: 320 → 63 ms de detecção,
480 → 143 ms, 640 → 245 ms). Para CPUs fracas, `FACE_MODEL=buffalo_s` é bem
mais leve, mas o `SIMILARITY_THRESHOLD` precisa ser recalibrado.

## Anti-spoofing (rosto vivo)

Sem isso, uma foto de alguém cadastrado mostrada no celular para a câmera
liberava a catraca. Com `FACE_LIVENESS=true` (padrão) o addon `liveness` do
InsightFace 2.0 classifica o rosto detectado como vivo ou não **antes** do
reconhecimento:

- `live_score >= LIVENESS_THRESHOLD` (padrão 0.8): segue para o ArcFace;
- abaixo: `autorizado: false, motivo: "rosto_nao_vivo"` (sem embedding, sem log);
- rosto colado na borda / sem margem ao redor: `motivo: "rosto_cortado"`
  (o addon exige que a região alinhada ao redor do rosto caiba na imagem).

As fotos de cadastro (tabela `pessoa`) **não** passam por essa checagem: elas
são fotos e precisam virar embedding mesmo assim.

Calibrando o limiar: se pessoas reais estiverem sendo recusadas (iluminação
ruim, câmera de baixa qualidade), baixe `LIVENESS_THRESHOLD` para 0.6–0.7 e
observe o `live_score` nas respostas; se foto ainda passar, suba. Para
desligar de vez, `FACE_LIVENESS=false` (o modelo nem é carregado).

É um modelo de imagem única (sem pedir para piscar ou mexer a cabeça):
protege contra foto no celular, impressão e replay em tela, não contra
máscaras 3D.

## Como funciona o cache de rostos

Converter base64 → embedding é caro, então os embeddings ficam em memória
(`face_store.py`), numa matriz N × 512:

- uma thread de fundo relê a tabela `pessoa` a cada `FACE_CACHE_TTL_SECONDS` (padrão 60 s);
- a releitura pede ao banco só `cpf`, `nome` e uma assinatura da foto, não o base64 inteiro;
- só baixa a foto e recalcula o embedding de quem é novo ou trocou de foto;
- quem sai da tabela sai do cache;
- quer efeito imediato após um cadastro? chame `POST /faces/reload`.

Fotos que não são base64 válido, ou nas quais nenhum rosto é detectado, ficam
listadas em `GET /faces/cache` com o motivo — bom para depurar cadastro ruim.

## Estrutura do banco

O D1 é SQLite serverless da Cloudflare. A API fala com ele pela API REST.

```sql
CREATE TABLE pessoa (          -- mantida pela OUTRA API (aqui é só leitura)
    cpf  TEXT PRIMARY KEY,
    nome TEXT NOT NULL,
    foto TEXT                  -- imagem em base64 (aceita "data:image/jpeg;base64,...")
);

CREATE TABLE logs (            -- escrita por esta API
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    pessoa_cpf TEXT NOT NULL REFERENCES pessoa(cpf) ON DELETE CASCADE,
    tipo       TEXT NOT NULL CHECK (tipo IN ('entrada','saida')),
    data_hora  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))  -- ISO 8601 UTC
);
```

Se os nomes das colunas na sua base forem outros, não precisa mexer no código:
ajuste `PESSOA_TABLE`, `PESSOA_PK`, `PESSOA_NOME_COL`, `PESSOA_FOTO_COL`,
`LOGS_TABLE`, `LOGS_FK_COL` e `LOGS_DATETIME_COL` no `.env`.

## Configurando o Cloudflare D1

1. Crie o banco (precisa de Node.js):

```bash
   npx wrangler login
   npx wrangler d1 create face_api       # mostra o database_id (UUID)
   npx wrangler whoami                   # mostra o Account ID
```

   Pelo painel: **Storage & Databases → D1 SQL Database → Create**.

2. Crie um API Token em **My Profile → API Tokens → Create Token → Custom token**
   com a permissão **Account → D1 → Edit**.

3. Preencha o `.env` (veja `.env.example`):

```env
   CLOUDFLARE_ACCOUNT_ID=...
   CLOUDFLARE_D1_DATABASE_ID=...
   CLOUDFLARE_API_TOKEN=...
```

> A tabela `pessoa` precisa estar no **mesmo** banco D1 em que esta API grava os
> logs. Se a API de cadastro ainda usa outro banco, ela também precisa apontar
> para o D1.

## Variáveis de ambiente

| Variável | Padrão | O que faz |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.40` | similaridade mínima para considerar a mesma pessoa (0.35–0.45 com `buffalo_l`) |
| `ACCESS_COOLDOWN_SECONDS` | `4` | intervalo mínimo entre dois registros da mesma pessoa |
| `FACE_CACHE_TTL_SECONDS` | `60` | de quanto em quanto tempo a thread de fundo relê a tabela `pessoa` |
| `FACE_DET_SIZE` | `320` | resolução do detector (320 catraca; 480/640 câmera longe) |
| `FACE_MODEL` | `buffalo_l` | pacote de modelos do InsightFace |
| `FACE_LIVENESS` | `true` | anti-spoofing: recusa foto/tela/impressão (`rosto_nao_vivo`); custa ~10 ms por requisição |
| `LIVENESS_THRESHOLD` | `0.8` | probabilidade mínima de rosto vivo (0 a 1); baixe para 0.6–0.7 se pessoas reais forem recusadas |
| `MAX_UPLOAD_BYTES` | `10485760` | tamanho máximo da imagem enviada |
| `AUTO_CREATE_TABLES` | `true` | cria as tabelas se não existirem (deixe `false` em produção) |
| `D1_TIMEOUT_SECONDS` | `15` | timeout de cada chamada ao D1 |

## Testando

```bash
python test_client.py foto.jpg          # envia a foto 1x e mostra a resposta
python test_client.py foto.jpg -n 20    # repete 20x e mede a latência
```

Se a foto for de alguém cadastrado, o acesso é registrado no banco (a
primeira chamada; as seguintes caem no cooldown).

Para testar o anti-spoofing, aponte a câmera para a mesma foto exibida na
tela do celular ou impressa: a resposta deve vir com `motivo: "rosto_nao_vivo"`
e `live_score` baixo. Um rosto que ocupa a imagem inteira dá `rosto_cortado`.

No Windows use `--url http://127.0.0.1:8000`: `localhost` tenta IPv6 primeiro
e cada requisição ganha ~2 s de espera que não é da API.

## Arquivos

| Arquivo | Papel |
|---|---|
| `main.py` | endpoints FastAPI |
| `face_engine.py` | InsightFace (detecção + anti-spoofing + reconhecimento) e decodificação de imagem |
| `face_store.py` | cache em memória dos embeddings (matriz NumPy) + thread de atualização |
| `access.py` | decide entrada/saída, cooldown e grava o log em segundo plano |
| `database.py` | acesso ao Cloudflare D1 via API REST (lê `pessoa`, escreve `logs`) |
| `config.py` | configurações via `.env` |
| `.env.example` | modelo das variáveis de ambiente |
| `test_client.py` | teste de fumaça e de latência |

## Fluxo de branches (Git Flow)

| Branch | Papel |
|---|---|
| `main` | código em produção; só recebe merge de `release/*` e `hotfix/*`, sempre com tag de versão |
| `develop` | integração do que vai entrar na próxima versão; base de toda `feature/*` |
| `feature/<nome>` | sai de `develop`, volta para `develop` via pull request |
| `release/<versão>` | sai de `develop`; só correções e bump de versão; fecha em `main` (com tag `v<versão>`) e em `develop` |
| `hotfix/<versão>` | sai de `main` para corrigir produção; fecha em `main` (com tag) e em `develop` |

Comandos do dia a dia:

```bash
# nova funcionalidade
git checkout develop && git pull
git checkout -b feature/minha-feature
# ... commits ...
git push -u origin feature/minha-feature   # abrir PR para develop

# nova versão
git checkout develop && git pull
git checkout -b release/1.2.0
# ajustar version= em main.py, corrigir o que faltar
git checkout main && git merge --no-ff release/1.2.0
git tag -a v1.2.0 -m "Release 1.2.0"
git checkout develop && git merge --no-ff release/1.2.0
git push origin main develop release/1.2.0 --tags
```

A versão da API (`version=` em `main.py`) deve acompanhar a tag da release.

## Histórico de versões

### 1.2.0

- Anti-spoofing em `POST /faces/recognize` com o addon `liveness` do
  InsightFace 2.0 (`FACE_LIVENESS`, padrão ligado; `LIVENESS_THRESHOLD`,
  padrão 0.8): foto no celular, impressão ou tela na frente da câmera é
  recusada com `motivo: "rosto_nao_vivo"`. Rosto sem margem ao redor para
  ser avaliado recebe `motivo: "rosto_cortado"`.
- Resposta ganha `live_score`; `GET /health` informa `anti_spoofing`.
- Custo medido: ~+10 ms por acesso liberado (230 para ~240 ms); foto de foto
  é recusada em ~75 ms. A sessão ONNX do addon roda com uma thread e sem
  spin-wait para não atrapalhar o ArcFace.
- Na primeira execução baixa `liveness.onnx` (~1,5 MB) para `~/.insightface/addons/`.
- `test_client.py` mostra também a mediana do `tempo_ms` do servidor.

### 1.1.0

- `POST /faces/recognize` ~7x mais rápido (~230 ms) e com resposta direta
  `autorizado`/`motivo` para a catraca. Nenhuma chamada ao banco no caminho da
  resposta: o log é gravado em segundo plano.
- Carrega só os modelos de detecção e reconhecimento; detector a 320 px
  (`FACE_DET_SIZE`); só o rosto principal é reconhecido; modelo aquecido na subida.
- Cache de embeddings em matriz NumPy, atualizado por thread de fundo que só
  baixa fotos novas ou alteradas.
- Removidos: `/faces/verify`, `/access/register`, `/access/status/{cpf}`,
  `/pessoas`, `/pessoas/{cpf}/foto`, o `docker-compose.yml` do Postgres (o
  banco é o D1) e a dependência direta do Pillow.
- Novos erros: `413` para imagem grande demais e `503` quando o D1 está fora.

### 1.0.0

- Primeira versão: reconhecimento com InsightFace, logs no Cloudflare D1.
