"""
diagnostico_2014.py — investiga o pico da razão base/declarado em ~2014.

Rode da raiz do projeto:  python diagnostico_2014.py

Pergunta central: o pico veio porque a BASE subiu ou porque o DECLARADO caiu?
Imprime, para 2013-2015:
  1. nº de transações por mês (descarta a hipótese "poucos dados")
  2. mediana de base, mediana de declarado e a razão, por mês
  3. distribuição da razão em 2014 (quartis) vs. um ano "normal" (2018)
  4. amostra das transações mais extremas de 2014
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import ITBI_RAW

# nomes de coluna no RAW (iguais aos do gerar_graficos.py)
COL_BASE = "Valor Base Calculo"
COL_DECLARADO = "Valor Declarado"
COL_DATA = "Data Quitacao Transacao"
COL_AREA = "Area Construida Adquirida"


def _num_br(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s = s.astype(str).str.strip().str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def main():
    raw = pd.read_csv(ITBI_RAW)
    print(f"Total de linhas no raw: {len(raw):,}\n")

    df = pd.DataFrame({
        "base": _num_br(raw[COL_BASE]),
        "decl": _num_br(raw[COL_DECLARADO]),
        "mes": pd.to_datetime(raw[COL_DATA], errors="coerce", dayfirst=True).dt.to_period("M"),
    })
    df = df[(df["base"] > 0) & (df["decl"] > 0)].dropna(subset=["mes"]).copy()
    df["razao"] = df["base"] / df["decl"]
    df["ano"] = df["mes"].dt.year

    # 1 + 2: por mês em 2013-2015
    jan = df[df["ano"].between(2013, 2015)]
    g = jan.groupby("mes")
    tabela = pd.DataFrame({
        "n": g.size(),
        "base_mediana": g["base"].median(),
        "decl_mediana": g["decl"].median(),
        "razao_mediana": g["razao"].median(),
    })
    pd.set_option("display.width", 120, "display.max_rows", 60)
    print("=== Por mês (2013-2015) ===")
    print(tabela.round(0).to_string())
    print()

    # 3: distribuição da razão — 2014 vs 2018
    print("=== Distribuição da razão base/declarado ===")
    for ano in (2014, 2018):
        r = df[df["ano"] == ano]["razao"]
        print(f"{ano}: n={len(r):,} | "
              f"p25={r.quantile(.25):.2f} | mediana={r.median():.2f} | "
              f"p75={r.quantile(.75):.2f} | p95={r.quantile(.95):.2f}")
    print()

    # 4: amostra das transações mais extremas de 2014
    print("=== 15 transações de 2014 com maior razão ===")
    top = df[df["ano"] == 2014].nlargest(15, "razao")
    print(top[["mes", "base", "decl", "razao"]].round(0).to_string(index=False))


if __name__ == "__main__":
    main()