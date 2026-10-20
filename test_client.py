"""
test_client.py - teste rápido de fumaça e de latência do POST /faces/recognize.

Uso:
  pip install requests
  python test_client.py foto.jpg                       # envia a foto 1x e mostra a resposta
  python test_client.py foto.jpg -n 20                 # repete 20x e mede a latência
  python test_client.py foto.jpg --url http://host:8000

Atenção: se a foto for de alguém cadastrado, o acesso É registrado no banco
(a primeira chamada; as seguintes caem no cooldown).
"""

import argparse
import json
import statistics
import sys
import time

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("foto", help="caminho de uma imagem (jpg/png/webp)")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("-n", type=int, default=1, help="quantas vezes enviar (mede latência)")
    a = ap.parse_args()
    base = a.url.rstrip("/")

    try:
        health = requests.get(f"{base}/health", timeout=10).json()
    except requests.RequestException as exc:
        print(f"API fora do ar: {exc}")
        sys.exit(1)
    print(f"health: {health}")

    with open(a.foto, "rb") as f:
        raw = f.read()

    # arquivo inválido não pode dar 500
    r = requests.post(f"{base}/faces/recognize", files={"file": ("x.txt", b"nao sou imagem", "text/plain")}, timeout=30)
    print(f"arquivo invalido -> HTTP {r.status_code} ({'ok' if r.status_code == 422 else 'ERRADO'})")

    tempos = []
    resposta = None
    for _ in range(a.n):
        t = time.perf_counter()
        r = requests.post(f"{base}/faces/recognize", files={"file": ("f.jpg", raw, "image/jpeg")}, timeout=120)
        tempos.append((time.perf_counter() - t) * 1000)
        resposta = r
        r.raise_for_status()

    print("\nresposta:")
    print(json.dumps(resposta.json(), indent=2, ensure_ascii=False))
    if a.n > 1:
        print(
            f"\nlatência ponta a ponta ({a.n}x): mediana {statistics.median(tempos):.0f} ms | "
            f"mín {min(tempos):.0f} ms | máx {max(tempos):.0f} ms"
        )


if __name__ == "__main__":
    main()
