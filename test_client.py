"""
test_client.py - testa o reconhecimento facial usando APENAS as fotos que ja
estao na tabela `pessoa` (baixadas pela propria API).

Uso:
  pip install requests pillow
  python test_client.py --url http://localhost:8000
  python test_client.py --url http://localhost:8000 --com-logs   # testa /faces/recognize (grava logs!)

O que testa (sem gravar nada no banco, a menos que use --com-logs):
  1. Saude da API e do cache (fotos cadastradas com problema)
  2. Pessoas DIFERENTES nao podem ser aceitas como a mesma (falso positivo)
  3. A MESMA pessoa continua sendo aceita com foto degradada (blur, escura,
     baixa resolucao, JPEG ruim, girada, recortada) (falso negativo)
  4. Sugere um limiar de similaridade com base nos numeros medidos
  5. (--com-logs) /faces/recognize identifica o CPF certo, cooldown e alternancia
"""
import argparse, io, itertools, sys, time
import requests
from PIL import Image, ImageFilter, ImageEnhance

G, R, Y = "\033[92m", "\033[91m", "\033[93m"
END = "\033[0m"
falhas = 0


def check(nome, ok, detalhe=""):
    global falhas
    falhas += 0 if ok else 1
    print(f"{G+'[OK]' if ok else R+'[FALHOU]'}{END} {nome} {detalhe}")


def to_jpeg(img, q=90):
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=q); return b.getvalue()


def variantes(raw):
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    cw, ch = int(w * .1), int(h * .1)
    return {
        "baixa resolucao (35%)": to_jpeg(img.resize((max(64, int(w*.35)), max(64, int(h*.35))))),
        "borrada": to_jpeg(img.filter(ImageFilter.GaussianBlur(2.5))),
        "escura": to_jpeg(ImageEnhance.Brightness(img).enhance(0.45)),
        "estourada": to_jpeg(ImageEnhance.Brightness(img).enhance(1.7)),
        "jpeg q=15": to_jpeg(img, 15),
        "girada 12 graus": to_jpeg(img.rotate(12, expand=True, fillcolor=(128, 128, 128))),
        "recortada 80%": to_jpeg(img.crop((cw, ch, w - cw, h - ch))),
        "espelhada": to_jpeg(img.transpose(Image.FLIP_LEFT_RIGHT)),
    }


def verify(base, a, b):
    r = requests.post(f"{base}/faces/verify",
                      files={"file1": ("a.jpg", a, "image/jpeg"), "file2": ("b.jpg", b, "image/jpeg")},
                      timeout=120)
    return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--com-logs", action="store_true", help="tambem testa /faces/recognize (GRAVA logs)")
    ap.add_argument("--max-pares", type=int, default=200, help="maximo de pares de pessoas diferentes")
    a = ap.parse_args()
    base = a.url.rstrip("/")

    print(f"\n=== Testando {base} ===\n")
    try:
        check("GET /health", requests.get(f"{base}/health", timeout=10).ok)
    except requests.RequestException as e:
        print(f"{R}API fora do ar:{END} {e}"); sys.exit(1)

    # ---------- 1. cadastro / cache ----------
    r = requests.post(f"{base}/faces/reload", timeout=600); resumo = r.json() if r.ok else {}
    cache = requests.get(f"{base}/faces/cache", timeout=15).json()
    pessoas = requests.get(f"{base}/pessoas", timeout=30).json()
    print(f"   pessoas na tabela: {len(pessoas)} | embeddings: {cache['embeddings_em_cache']} "
          f"| limiar atual: {cache['limiar_similaridade']}")
    if cache["fotos_com_problema"]:
        print(f"{Y}[AVISO]{END} {len(cache['fotos_com_problema'])} cadastro(s) ignorado(s) (foto invalida):")
        for p in cache["fotos_com_problema"]:
            print(f"   {Y}-> CPF {p['cpf']}: {p['motivo']}{END}")
    check("Ha pelo menos 1 cadastro valido", cache["embeddings_em_cache"] >= 1)
    fotos = {}
    bad = {p["cpf"] for p in cache["fotos_com_problema"]}
    for p in pessoas:
        if p["cpf"] in bad: continue
        r = requests.get(f"{base}{p['foto_url']}", timeout=30)
        if r.ok: fotos[p["cpf"]] = (p["nome"], r.content)
    print(f"   fotos baixadas do banco: {len(fotos)}\n")

    # ---------- 2. pessoas diferentes ----------
    print("== Pessoas diferentes (esperado: NAO e a mesma pessoa) ==")
    pares = list(itertools.combinations(fotos, 2))[: a.max_pares]
    sims_dif, fp = [], []
    for c1, c2 in pares:
        st, j = verify(base, fotos[c1][1], fotos[c2][1])
        if st != 200: print(f"   {Y}par {c1}/{c2}: HTTP {st}{END}"); continue
        sims_dif.append(j["similarity"])
        if j["same_person"]: fp.append((c1, c2, j["similarity"]))
    if not pares:
        print(f"{Y}[PULADO]{END} precisa de 2+ cadastros validos para testar falso positivo")
    else:
        check(f"Falsos positivos entre {len(sims_dif)} pares", not fp,
              f"(maior similaridade entre diferentes: {max(sims_dif):.3f})" if sims_dif else "")
    for c1, c2, s in fp:
        print(f"   {R}-> {fotos[c1][0]} x {fotos[c2][0]} = {s} (aceitos como mesma pessoa){END}")

    # ---------- 3. mesma pessoa, foto degradada ----------
    print("\n== Mesma pessoa com foto degradada (esperado: E a mesma pessoa) ==")
    sims_mesma, por_variante = [], {}
    for cpf, (nome, raw) in fotos.items():
        for nomev, vb in variantes(raw).items():
            st, j = verify(base, raw, vb)
            if st == 200:
                sims_mesma.append(j["similarity"])
                por_variante.setdefault(nomev, []).append((j["same_person"], j["similarity"], nome))
            else:  # normalmente = nenhum rosto detectado na variante
                por_variante.setdefault(nomev, []).append((False, None, nome))
    for nomev, lst in por_variante.items():
        ok = sum(1 for s, _, _ in lst if s)
        vals = [v for _, v, _ in lst if v is not None]
        check(f"{nomev}: {ok}/{len(lst)} reconhecidos", ok == len(lst),
              f"(sim min {min(vals):.3f})" if vals else "(rosto nao detectado)")
        for s, v, n in lst:
            if not s: print(f"   {Y}-> falhou para {n} (sim={v}){END}")

    # ---------- 4. limiar ----------
    print("\n== Limiar ==")
    if not sims_dif:
        print(f"   {Y}Sem pares de pessoas diferentes, nao da para sugerir limiar.{END}")
    elif sims_mesma:
        top_dif, low_same = max(sims_dif), min(sims_mesma)
        print(f"   maior similaridade entre pessoas DIFERENTES : {top_dif:.3f}")
        print(f"   menor similaridade da MESMA pessoa (degradada): {low_same:.3f}")
        print(f"   limiar atual                                 : {cache['limiar_similaridade']}")
        if top_dif < low_same:
            print(f"   {G}Ha margem: qualquer limiar entre {top_dif:.2f} e {low_same:.2f} separa os grupos "
                  f"(sugestao: {(top_dif+low_same)/2:.2f}).{END}")
        else:
            print(f"   {R}Os grupos se sobrepoem: nao existe limiar perfeito com essas fotos. "
                  f"Revise a qualidade das fotos cadastradas.{END}")

    # ---------- 5. recognize (grava logs) ----------
    if a.com_logs:
        print("\n== /faces/recognize (GRAVA logs) ==")
        for cpf, (nome, raw) in fotos.items():
            r = requests.post(f"{base}/faces/recognize", files={"file": ("f.jpg", raw, "image/jpeg")}, timeout=120)
            res = (r.json().get("results") or [{}])[0] if r.ok else {}
            check(f"{nome}: identificado como o CPF certo", res.get("cpf") == cpf,
                  f"(sim={res.get('similarity')}, tipo={(res.get('access') or {}).get('tipo')})")
            r2 = requests.post(f"{base}/faces/recognize", files={"file": ("f.jpg", raw, "image/jpeg")}, timeout=120)
            res2 = (r2.json().get("results") or [{}])[0] if r2.ok else {}
            check(f"{nome}: 2a chamada imediata ignorada (cooldown)", (res2.get("access") or {}).get("skipped") is True)
        # imagem sem rosto (cinza)
        b = io.BytesIO(); Image.new("RGB", (400, 400), (120, 120, 120)).save(b, "JPEG")
        r = requests.post(f"{base}/faces/recognize", files={"file": ("n.jpg", b.getvalue(), "image/jpeg")}, timeout=60)
        check("Imagem sem rosto => faces_found=0", r.ok and r.json().get("faces_found") == 0)
        r = requests.post(f"{base}/faces/recognize", files={"file": ("x.txt", b"nao sou imagem", "text/plain")}, timeout=30)
        check(f"Arquivo invalido nao deve dar 500 (veio {r.status_code})", r.status_code in (400, 415, 422))
    else:
        print(f"\n{Y}(pulei /faces/recognize para nao gravar logs; use --com-logs para incluir){END}")

    print(f"\n{'Tudo certo.' if not falhas else f'{falhas} teste(s) falharam.'}")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()