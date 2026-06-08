import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (ITBI_FINAL, OUTPUTS_TABLES, OUTPUTS_MODELS_TRAIN2, TARGET)
from src.modeling.evaluate import fn_baseline, fn_lightgbm
from src.sistema_decisao.recomendador import RecomendadorBanda

RESULTADOS_DIR = OUTPUTS_TABLES / 'resultados_gerar_grafico'
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

ANOS_CORTE = list(range(2011, 2024))

# ── fig1(b) + fig3b: predições out-of-sample linha a linha, série inteira ─────
def predicoes_rolling(df, anos=ANOS_CORTE):
    df = df.copy()
    df['_data'] = pd.to_datetime(df['data_transacao'])
    blocos = []
    for Y in anos:
        tr = df[df['ano_transacao'] <= Y]
        te = df[df['ano_transacao'] == Y + 1]
        if len(te) < 100 or len(tr) < 500:
            continue
        print(f"    origem treino<={Y} | teste {Y + 1} (n={len(te):,})...")
        pred_lgbm = fn_lightgbm(tr, te)
        pred_base = fn_baseline(tr, te)
        blocos.append(pd.DataFrame({
            'data_transacao': te['_data'].values,
            'real': te[TARGET].values,
            'pred': pred_lgbm,
            'pred_baseline': pred_base,
        }))
    return pd.concat(blocos, ignore_index=True)


def exportar_predicoes_e_trimestre(df):
    print("\n[1/3] Predições rolling-origin (fig1b + fig3b)...")
    p = predicoes_rolling(df)

    # fig1(b): a série inteira out-of-sample (é isso que faz o salto aparecer)
    p[['data_transacao', 'real', 'pred']].to_csv(
        RESULTADOS_DIR / 'predicoes_backtest.csv', index=False)
    print(f"  salvo: predicoes_backtest.csv ({len(p):,} linhas)")

    # fig3b: MAPE por trimestre, baseline vs LightGBM
    p['trimestre'] = pd.to_datetime(p['data_transacao']).dt.to_period('Q').dt.to_timestamp()
    p['ape_lgbm'] = np.abs(p['pred'] - p['real']) / np.maximum(p['real'], 1) * 100
    p['ape_base'] = np.abs(p['pred_baseline'] - p['real']) / np.maximum(p['real'], 1) * 100
    trim = (p.groupby('trimestre')
            .agg(lightgbm=('ape_lgbm', 'mean'),
                 baseline=('ape_base', 'mean'),
                 n=('ape_lgbm', 'size'))
            .reset_index())
    trim.to_csv(RESULTADOS_DIR / 'erro_por_trimestre.csv', index=False)
    print(f"  salvo: erro_por_trimestre.csv ({len(trim)} trimestres)")


# ── fig3a: a partir do comparativo que o evaluate.py já salvou (rápido) ───────
def exportar_backtest_modelos():
    print("\n[2/3] Backtest por modelo (fig3a) — a partir do comparativo do evaluate.py...")
    origem = OUTPUTS_TABLES / 'comparativo_regime_novo.csv'
    if not origem.exists():
        print(f"  [aviso] {origem} não existe. Rode o evaluate.py primeiro. Pulando fig3a.")
        return
    comp = pd.read_csv(origem)
    partes = comp['MAPE'].astype(str).str.split('±', expand=True)   # "14.36 ± 0.48"
    out = pd.DataFrame({
        'modelo': comp['Modelo'],
        'mape_medio': pd.to_numeric(partes[0].str.strip(), errors='coerce'),
        'mape_std': pd.to_numeric(partes[1].str.strip(), errors='coerce') if partes.shape[1] > 1 else 0.0,
    })
    out.to_csv(RESULTADOS_DIR / 'backtest_modelos.csv', index=False)
    print(f"  salvo: backtest_modelos.csv\n{out.to_string(index=False)}")


# ── fig5b: bandas do recomendador em 3 imóveis-exemplo ────────────────────────
def exportar_exemplos_recomendador(df):
    print("\n[3/3] Exemplos do recomendador (fig5b)...")
    caminho_rec = OUTPUTS_MODELS_TRAIN2 / 'recomendador_banda.pkl'
    if not caminho_rec.exists():
        print(f"  [aviso] {caminho_rec} não existe. Rode o recomendador.py primeiro. Pulando fig5b.")
        return
    rec = RecomendadorBanda.carregar(caminho_rec)

    df_2024 = df[df['ano_transacao'] == 2024].sort_values(TARGET)
    n = len(df_2024)
    # três imóveis cobrindo valor baixo/médio/alto, para a figura ser legível
    idxs = [df_2024.index[n // 5], df_2024.index[n // 2], df_2024.index[4 * n // 5]]
    faixa = rec.prever_faixa(df_2024.loc[idxs])

    # um cenário de cada situação. 'dentro' usa pedido 25% acima do central, de
    # propósito, para a figura ilustrar o caso "acima do p50 mas dentro da banda".
    cenarios = [
        ('Imóvel A (pedido +25%, ainda dentro)', 'dentro'),
        ('Imóvel B (sobrepreço)', 'caro'),
        ('Imóvel C (subdeclarado)', 'barato'),
    ]
    linhas = []
    for (rotulo, alvo), (idx, f) in zip(cenarios, faixa.iterrows()):
        inf, p50, sup = f['inferior'], f['previsto'], f['superior']
        if alvo == 'caro':
            pedido = sup * 1.12
        elif alvo == 'barato':
            pedido = inf * 0.85
        else:
            pedido = p50 * 1.25
        situ = 'caro' if pedido > sup else 'barato' if pedido < inf else 'dentro'
        linhas.append({'rotulo': rotulo, 'p10': inf, 'p50': p50, 'p90': sup,
                       'pedido': pedido, 'situacao': situ})
    pd.DataFrame(linhas).to_csv(RESULTADOS_DIR / 'exemplos_recomendador.csv', index=False)
    print(f"  salvo: exemplos_recomendador.csv\n{pd.DataFrame(linhas).to_string(index=False)}")


if __name__ == '__main__':
    df = pd.read_csv(ITBI_FINAL)
    exportar_predicoes_e_trimestre(df)
    exportar_backtest_modelos()
    exportar_exemplos_recomendador(df)
    print(f"\nPronto. CSVs em: {RESULTADOS_DIR}")
    print("Agora rode o gerar_graficos.py (com MODELO_LGBM = OUTPUTS_MODELS e "
          "ANO_TESTE_INICIO = 2023) — todas as figuras devem sair.")