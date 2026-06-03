import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (ITBI_FINAL, OUTPUTS_TABLES, OUTPUTS_MODELS_TRAIN2, TARGET)
from src.modeling.evaluate import (backtest_rolling, fn_baseline, fn_xgboost,
                                   fn_lightgbm, fn_ensemble)
from src.sistema_decisao.recomendador import RecomendadorBanda

# ── parâmetros ────────────────────────────────────────────────────────────────
OUT = OUTPUTS_TABLES / 'resultados_gerar_grafico'
OUT.mkdir(parents=True, exist_ok=True)
ANOS = list(range(2018, 2024))           # mesmos anos do seu evaluate.py
N_EXEMPLOS = 5                            # imóveis na figura da banda

# fig3b: True -> trimestres do REGIME NOVO (curva curta, pós jul/2023).
#        False -> série inteira ANUAL (mostra o salto de regime; rótulo vira ano).
TRIMESTRE_REGIME_NOVO = True

# pedido = previsto * fator, para ilustrar dentro/caro/barato (a situação real
# é decidida pelo SEU recomendador.analisar, não forçada aqui).
FATORES_PEDIDO = [1.00, 1.50, 0.60, 1.15, 0.85]


def _std(s):
    v = s.std()
    return float(v) if pd.notna(v) else 0.0


# ── fig3a ─────────────────────────────────────────────────────────────────────
def exportar_backtest_modelos(df):
    modelos = {'Baseline R$/m²': fn_baseline, 'XGBoost': fn_xgboost,
               'LightGBM': fn_lightgbm, 'Ensemble': fn_ensemble}
    linhas = []
    for nome, fn in modelos.items():
        res = backtest_rolling(df, fn, ANOS, nome,
                               apenas_regime_novo=True, silencioso=True)
        if len(res):
            linhas.append({'modelo': nome,
                           'mape_medio': float(res['MAPE'].mean()),
                           'mape_std': _std(res['MAPE'])})
    caminho = OUT / 'backtest_modelos.csv'
    pd.DataFrame(linhas).to_csv(caminho, index=False)
    print(f"  salvo: {caminho}")


# ── fig3b ─────────────────────────────────────────────────────────────────────
def exportar_erro_por_trimestre(df):
    rb = backtest_rolling(df, fn_baseline, ANOS, 'baseline',
                          apenas_regime_novo=TRIMESTRE_REGIME_NOVO, silencioso=True)
    rl = backtest_rolling(df, fn_lightgbm, ANOS, 'lightgbm',
                          apenas_regime_novo=TRIMESTRE_REGIME_NOVO, silencioso=True)
    b = rb[['testa', 'MAPE']].rename(columns={'MAPE': 'baseline'})
    l = rl[['testa', 'MAPE']].rename(columns={'MAPE': 'lightgbm'})
    m = b.merge(l, on='testa', how='outer')
    if TRIMESTRE_REGIME_NOVO:
        m['trimestre'] = pd.PeriodIndex(m['testa'], freq='Q').to_timestamp()
    else:
        m['trimestre'] = pd.to_datetime(m['testa'].astype(str) + '-01-01', errors='coerce')
    m = m.sort_values('trimestre')
    caminho = OUT / 'erro_por_trimestre.csv'
    m[['trimestre', 'baseline', 'lightgbm']].to_csv(caminho, index=False)
    print(f"  salvo: {caminho}")


# ── fig5b ─────────────────────────────────────────────────────────────────────
def _carregar_ou_treinar_recomendador(df):
    pkl = OUTPUTS_MODELS_TRAIN2 / 'recomendador_banda.pkl'
    if pkl.exists():
        print(f"  recomendador carregado de {pkl.name}")
        return RecomendadorBanda.carregar(pkl)
    print("  recomendador_banda.pkl não encontrado — treinando (treino<=2023, calib=1ª metade 2024)...")
    df = df.copy()
    df['_data'] = pd.to_datetime(df['data_transacao'])
    df_2024 = df[df['ano_transacao'] == 2024].sort_values('_data')
    corte = len(df_2024) // 2
    return RecomendadorBanda().treinar(df[df['ano_transacao'] <= 2023],
                                       df_2024.iloc[:corte])


def exportar_exemplos_recomendador(df):
    rec = _carregar_ou_treinar_recomendador(df)
    teste = df[df['ano_transacao'] == 2024].copy()
    exemplos = teste.sample(min(N_EXEMPLOS, len(teste)), random_state=1)
    faixa = rec.prever_faixa(exemplos)

    mapa = {'REDUZIR PREÇO': 'caro',
            'POSSÍVEL OPORTUNIDADE / SUBDECLARAÇÃO': 'barato',
            'DENTRO DA FAIXA DE MERCADO': 'dentro'}

    fatores = (FATORES_PEDIDO * ((len(exemplos) // len(FATORES_PEDIDO)) + 1))[:len(exemplos)]
    linhas = []
    for (idx, row), fator in zip(exemplos.iterrows(), fatores):
        prev = float(faixa.loc[idx, 'previsto'])
        pedido = prev * fator
        a = rec.analisar(exemplos.loc[[idx]], pedido)
        bairro = str(row.get('bairro', '?')).title()
        tipo = str(row.get('tipo_construtivo', ''))
        linhas.append({
            'rotulo': f"{tipo} {bairro}".strip(),
            'p10': float(faixa.loc[idx, 'inferior']),
            'p50': prev,
            'p90': float(faixa.loc[idx, 'superior']),
            'pedido': pedido,
            'situacao': mapa.get(a['acao'], 'dentro'),
        })
    caminho = OUT / 'exemplos_recomendador.csv'
    pd.DataFrame(linhas).to_csv(caminho, index=False)
    print(f"  salvo: {caminho}")


def main():
    print(f"Carregando dataset final: {ITBI_FINAL}")
    df = pd.read_csv(ITBI_FINAL)

    print("\n[1/3] Backtest por modelo (regime novo)...")
    exportar_backtest_modelos(df)

    print("\n[2/3] Erro por trimestre (baseline vs LightGBM)...")
    exportar_erro_por_trimestre(df)

    print("\n[3/3] Exemplos do recomendador (banda p10/p50/p90)...")
    exportar_exemplos_recomendador(df)

    print(f"\nPronto. CSVs em {OUT}")
    print("Agora rode: python src/modeling/gerar_graficos.py")


if __name__ == '__main__':
    main()