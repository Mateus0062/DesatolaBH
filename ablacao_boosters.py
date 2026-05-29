"""
ablacao_boosters.py — Por que os boosters ficam atrás da RF?

Isola, UMA VARIÁVEL POR VEZ, o efeito de:
  (A) monotonia ligada vs desligada          <- principal suspeita
  (B) n_estimators fixo (300 / 600 / 1000)   <- a sua hipótese
  (C) early stopping na cauda recente vs n_estimators fixo

Cada variante roda pelo MESMO backtest do regime novo (média ± desvio entre
trimestres), para a comparação ser honesta — não um teste único. Faz isso para
XGBoost E LightGBM. No fim, imprime uma tabela por modelo.

Rode da raiz do projeto:  python -m src.modeling.ablacao_boosters
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (ITBI_FINAL, TARGET, COLUNAS_EXCLUIR_MODELO,
                    HIPERPARAMETROS, MONOTONIC_CONSTRAINTS)
from src.modeling.evaluate import backtest_rolling

EARLY_STOPPING_ROUNDS = 50
N_ESTIMATORS_TETO = 4000
FRAC_EARLY_STOPPING = 0.12


# ─── helpers (espelham o train.py, sem alvo em R$/m² para simplificar) ──────
def _Xy(df):
    y = df[TARGET].copy()
    aux = ['_data', '_periodo']
    X = df.drop(columns=[c for c in COLUNAS_EXCLUIR_MODELO + [TARGET] + aux
                         if c in df.columns])
    return X, y


def _vetor_monotonia(colunas, framework):
    cons = [MONOTONIC_CONSTRAINTS.get(c, 0) for c in colunas]
    if framework == 'xgboost':
        return '(' + ','.join(str(x) for x in cons) + ')'
    return cons


def _split_cauda(X, y, datas, frac=FRAC_EARLY_STOPPING):
    ordem = np.argsort(datas.values, kind='stable')
    k = max(int(len(ordem) * frac), 1)
    return (X.iloc[ordem[:-k]], y.iloc[ordem[:-k]],
            X.iloc[ordem[-k:]], y.iloc[ordem[-k:]])


def _novo_modelo(framework, n_estimators, usar_monotonia, colunas):
    hp = dict(HIPERPARAMETROS[framework])
    hp.pop('n_estimators', None)
    mono = _vetor_monotonia(colunas, framework) if usar_monotonia else None
    if framework == 'xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(**hp, n_estimators=n_estimators,
                            monotone_constraints=mono,
                            random_state=42, n_jobs=-1, verbosity=0)
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**hp, n_estimators=n_estimators,
                         monotone_constraints=mono,
                         random_state=42, n_jobs=-1, verbose=-1)


# ─── fábricas de "treina-e-prevê" para cada variante ────────────────────────
def make_fn(framework, usar_monotonia=True, n_fixo=None,
            usar_cauda=True):
    """
    usar_cauda=True  -> early stopping na cauda recente, depois re-treina no full.
    usar_cauda=False -> treina direto com n_fixo (sem early stopping).
    n_fixo definido  -> ignora a cauda e usa esse n_estimators direto.
    """
    from xgboost import XGBRegressor  # noqa
    from lightgbm import LGBMRegressor, early_stopping as lgb_es  # noqa

    def fn(df_tr, df_te):
        X_tr, y_tr = _Xy(df_tr)
        X_te, _ = _Xy(df_te)
        X_te = X_te[X_tr.columns]
        cols = list(X_tr.columns)

        # caminho 1: n_estimators fixo, sem early stopping
        if n_fixo is not None or not usar_cauda:
            n = n_fixo if n_fixo is not None else 600
            mdl = _novo_modelo(framework, n, usar_monotonia, cols)
            mdl.fit(X_tr, np.log1p(y_tr))
            return np.expm1(mdl.predict(X_te))

        # caminho 2: early stopping na cauda recente -> best_n -> refit no full
        datas = pd.to_datetime(df_tr['data_transacao'])
        Xf, yf, Xe, ye = _split_cauda(X_tr, y_tr, datas)
        es = _novo_modelo(framework, N_ESTIMATORS_TETO, usar_monotonia, cols)
        if framework == 'xgboost':
            es.set_params(early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            es.fit(Xf, np.log1p(yf), eval_set=[(Xe, np.log1p(ye))], verbose=False)
            best_n = es.best_iteration + 1
        else:
            from lightgbm import early_stopping as lgb_es2
            es.fit(Xf, np.log1p(yf), eval_set=[(Xe, np.log1p(ye))],
                   callbacks=[lgb_es2(EARLY_STOPPING_ROUNDS, verbose=False)])
            best_n = es.best_iteration_
        final = _novo_modelo(framework, best_n, usar_monotonia, cols)
        final.fit(X_tr, np.log1p(y_tr))
        return np.expm1(final.predict(X_te))

    return fn


def rodar_ablacao(framework):
    df = pd.read_csv(ITBI_FINAL)
    anos = list(range(2018, 2024))

    variantes = {
        'atual (mono + early stop)':      make_fn(framework, True,  None, True),
        'SEM monotonia (+ early stop)':   make_fn(framework, False, None, True),
        'n_estimators=300 (mono)':        make_fn(framework, True,  300,  False),
        'n_estimators=600 (mono)':        make_fn(framework, True,  600,  False),
        'n_estimators=1000 (mono)':       make_fn(framework, True,  1000, False),
        'n=600 + SEM monotonia':          make_fn(framework, False, 600,  False),
    }

    print("\n" + "#" * 80)
    print(f"#  ABLAÇÃO — {framework.upper()} — backtest REGIME NOVO (média ± desvio)")
    print("#" * 80)

    linhas = []
    for nome, fn in variantes.items():
        res = backtest_rolling(df, fn, anos, f'{framework}:{nome}',
                               apenas_regime_novo=True, silencioso=True)
        if len(res):
            linhas.append({
                'Variante': nome,
                'MAPE': f"{res['MAPE'].mean():.2f} ± {res['MAPE'].std():.2f}",
                'MdAPE': f"{res['MdAPE'].mean():.2f} ± {res['MdAPE'].std():.2f}",
                'RMSLE': f"{res['RMSLE'].mean():.3f}",
                'R²': f"{res['R²'].mean():.3f}",
                '_ord': res['MdAPE'].mean(),
            })
    tab = pd.DataFrame(linhas).sort_values('_ord').drop(columns='_ord')
    print("\n" + tab.to_string(index=False))
    print(f"\n  → Melhor variante de {framework} por MdAPE: {tab.iloc[0]['Variante']}")
    return tab


if __name__ == '__main__':
    tab_xgb = rodar_ablacao('xgboost')
    tab_lgb = rodar_ablacao('lightgbm')

    from config import OUTPUTS_TABLES
    tab_xgb.to_csv(OUTPUTS_TABLES / 'ablacao_xgboost.csv', index=False)
    tab_lgb.to_csv(OUTPUTS_TABLES / 'ablacao_lightgbm.csv', index=False)
    print("\nTabelas de ablação salvas em outputs/tables/.")
    print("Leitura: se 'SEM monotonia' melhora muito, a monotonia é o custo.")