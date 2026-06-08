"""
piso_ruido.py — Estima o PISO de erro irredutível do alvo via transações
quase-idênticas (mesmo bairro, tipo, padrão, ano e faixa de área).

IDEIA
-----
Duas transações idênticas do ponto de vista das features de agrupamento ainda
têm preços/m² diferentes. Essa dispersão intra-grupo é uma estimativa do erro
que NENHUM modelo conseguiria eliminar usando essas características — porque ele
não tem como distinguir duas unidades iguais que foram declaradas por valores
diferentes. Se a dispersão intra-grupo ≈ erro de teste do modelo (~14% backtest
/ ~18% holdout), o modelo está perto do limite dos dados — e é por isso que
Random Forest, XGBoost e LightGBM empatam.

COMO RODAR
----------
Coloque este arquivo na RAIZ do projeto (ao lado do config.py) e rode:
    python piso_ruido.py

Mexa em MIN_GRUPO / BIN_AREA_M2 para ver a sensibilidade (grupos mais finos =
estimativa mais apertada, porém menos grupos).
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import ITBI_FINAL, TARGET, classificar_regime

# ── parâmetros ──────────────────────────────────────────────────────────────
MIN_GRUPO = 3          # mínimo de transações por grupo para medir dispersão
BIN_AREA_M2 = 10       # largura do bucket de área (m²); ±5 m² em torno do centro
SO_REGIME_NOVO = True  # True = só regime novo (onde mora o ~18%); False = tudo


def carregar():
    df = pd.read_csv(ITBI_FINAL)
    df['data_transacao'] = pd.to_datetime(df['data_transacao'])
    if SO_REGIME_NOVO:
        regime = classificar_regime(df['data_transacao'])
        df = df[np.asarray(regime) == 'novo'].copy()
    df['pm2'] = df[TARGET] / df['area_construida_m2']
    df = df[df['pm2'] > 0].copy()
    return df


def montar_grupos(df):
    df = df.copy()
    # bucket de área: arredonda para o múltiplo mais próximo de BIN_AREA_M2
    df['area_bucket'] = (df['area_construida_m2'] / BIN_AREA_M2).round() * BIN_AREA_M2
    chaves = ['bairro', 'tipo_construtivo', 'padrao_acabamento_num',
              'ano_transacao', 'area_bucket']
    tam = df.groupby(chaves)['pm2'].transform('size')
    df_multi = df[tam >= MIN_GRUPO].copy()
    df_multi['grupo'] = df_multi[chaves].astype(str).agg(' | '.join, axis=1)
    return df_multi


def medir_piso(df_multi):
    # melhor predição possível para cada grupo = a mediana de R$/m² do grupo
    med = df_multi.groupby('grupo')['pm2'].transform('median')
    erro_pct = (df_multi['pm2'] - med).abs() / df_multi['pm2'] * 100

    n_grupos = df_multi['grupo'].nunique()
    n_trans = len(df_multi)
    piso_mape = erro_pct.mean()
    piso_mdape = erro_pct.median()
    p90 = erro_pct.quantile(0.90)

    print("=" * 72)
    print("PISO DE ERRO IRREDUTÍVEL — via transações quase-idênticas")
    print("=" * 72)
    print(f"Recorte: {'REGIME NOVO (>= 2023)' if SO_REGIME_NOVO else 'série inteira'}")
    print(f"Chave de grupo: bairro + tipo + padrão + ano + área (±{BIN_AREA_M2/2:.0f} m²)")
    print(f"Grupos com >= {MIN_GRUPO} transações: {n_grupos:,}")
    print(f"Transações dentro desses grupos: {n_trans:,}")
    print("-" * 72)
    print(f"  Piso MAPE  (erro médio vs mediana do grupo): {piso_mape:5.2f}%")
    print(f"  Piso MdAPE (erro mediano):                   {piso_mdape:5.2f}%")
    print(f"  P90 do erro intra-grupo:                     {p90:5.2f}%")
    print("=" * 72)
    print("LEITURA")
    print("-" * 72)
    print("Isto é uma ESTIMATIVA do piso, não um limite exato:")
    print(" • Tende a SUPERestimar o piso: os grupos ainda guardam sinal que o")
    print("   modelo capta por features extras (knn, dist_centro, comparáveis).")
    print(" • Tende a SUBestimar: usa a mediana do próprio grupo como 'oráculo',")
    print("   que um modelo out-of-sample não conhece de antemão.")
    print("Na prática, é um bom proxy. Se o erro de teste do modelo (~14%")
    print("backtest / ~18% holdout) está perto deste piso, o teto é dos DADOS,")
    print("não do algoritmo — e a escolha de modelo é de segunda ordem.")
    print("=" * 72)
    return piso_mape, piso_mdape


if __name__ == '__main__':
    df = carregar()
    df_multi = montar_grupos(df)
    if len(df_multi) == 0:
        print("Nenhum grupo com transações suficientes. Afrouxe MIN_GRUPO ou "
              "aumente BIN_AREA_M2 e rode de novo.")
        sys.exit(0)
    medir_piso(df_multi)