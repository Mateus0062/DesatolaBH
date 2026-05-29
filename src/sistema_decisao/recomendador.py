# src/sistema_decisao/recomendador_v2.py
# ITEM 8 (parte 2) — Recomendador baseado em BANDA, não em ponto.
# Regra: só chama de "caro"/"barato" quando o preço pedido sai do intervalo
# calibrado [inferior, superior]. Dentro da banda = indistinguível do ruído do
# modelo, então a recomendação honesta é "alinhado / dentro da incerteza".

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_FINAL, TARGET
from src.sistema_decisao.intervalos import (treinar_base_log,
                                            IntervaloConformal)


class RecomendadorV2:
    def __init__(self, intervalo_conformal):
        self.conf = intervalo_conformal

    def analisar(self, df_imovel, preco_pedido):
        if isinstance(df_imovel, dict):
            df_imovel = pd.DataFrame([df_imovel])
        banda = self.conf.prever(df_imovel).iloc[0]
        inf, prev, sup = banda["inferior"], banda["previsto"], banda["superior"]
        largura_rel = (sup - inf) / prev * 100

        if preco_pedido > sup:
            acao, prio = "REDUZIR PREÇO", "ALTA"
            just = (f"Preço acima do teto da faixa estimada "
                    f"(R$ {inf:,.0f}–{sup:,.0f}). Mesmo considerando a "
                    f"incerteza do modelo (±{largura_rel/2:.0f}%), está caro.")
            sugerido = sup
        elif preco_pedido < inf:
            acao, prio = "POSSÍVEL OPORTUNIDADE / SUBDECLARAÇÃO", "MÉDIA"
            just = (f"Preço abaixo do piso da faixa estimada "
                    f"(R$ {inf:,.0f}–{sup:,.0f}). Pode ser barganha real ou "
                    f"subdeclaração — investigar.")
            sugerido = inf
        else:
            acao, prio = "DENTRO DA FAIXA DE MERCADO", "BAIXA"
            just = (f"Preço dentro do intervalo estimado "
                    f"(R$ {inf:,.0f}–{sup:,.0f}). A diferença para o ponto "
                    f"central está dentro da incerteza do modelo — sem ação.")
            sugerido = preco_pedido

        return {
            "preco_pedido": preco_pedido,
            "previsto": prev,
            "faixa_inferior": inf,
            "faixa_superior": sup,
            "largura_rel_%": largura_rel,
            "acao": acao,
            "prioridade": prio,
            "justificativa": just,
            "preco_sugerido": sugerido,
        }

    def relatorio(self, a):
        print("\n" + "=" * 70)
        print("ANÁLISE DO IMÓVEL (com banda de incerteza)")
        print("=" * 70)
        print(f"  Preço pedido : R$ {a['preco_pedido']:>13,.2f}")
        print(f"  Estimado     : R$ {a['previsto']:>13,.2f}")
        print(f"  Faixa        : R$ {a['faixa_inferior']:>13,.2f} – "
              f"R$ {a['faixa_superior']:,.2f}  (±{a['largura_rel_%']/2:.0f}%)")
        print(f"  Recomendação : {a['acao']}  [{a['prioridade']}]")
        print(f"  {a['justificativa']}")


if __name__ == "__main__":
    df = pd.read_csv(ITBI_FINAL)
    tr = df[df["ano_transacao"] <= 2021]
    cal = df[df["ano_transacao"] == 2022]
    base, cols = treinar_base_log(tr)
    conf = IntervaloConformal(base, cols).calibrar(cal)
    rec = RecomendadorV2(conf)

    exemplo = df[df["ano_transacao"] == 2024].iloc[[0]]
    prev = conf.prever(exemplo).iloc[0]["previsto"]
    for fator in (1.30, 1.00, 0.70):
        a = rec.analisar(exemplo, prev * fator)
        rec.relatorio(a)