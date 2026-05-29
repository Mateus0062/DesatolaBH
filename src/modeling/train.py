import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    ITBI_FINAL,
    OUTPUTS_MODELS, OUTPUTS_TABLES, HIPERPARAMETROS,
    TARGET, COLUNAS_EXCLUIR_MODELO,
    MONOTONIC_CONSTRAINTS, PREVER_PRECO_M2, TREINAR_APENAS_REGIME_NOVO,
    classificar_regime,
)

EARLY_STOPPING_ROUNDS = 50
N_ESTIMATORS_MAX = 4000
# Fração final (por data) do treino usada só para early stopping dos boosters.
# Depois do early stopping, o booster é RE-TREINADO no treino inteiro.
FRAC_EARLY_STOPPING = 0.12


# ─── ITEM 5: alvo opcional em R$/m² ─────────────────────────────────────────
def _y_para_modelo(y, area):
    if PREVER_PRECO_M2:
        return np.log1p(np.asarray(y, float) / np.asarray(area, float))
    return np.log1p(np.asarray(y, float))


def _pred_para_reais(pred_log, area):
    val = np.expm1(pred_log)
    if PREVER_PRECO_M2:
        return val * np.asarray(area, float)
    return val


# ─── ITEM 5: vetor de monotonicidade alinhado às colunas ────────────────────
def _vetor_monotonia(colunas, framework):
    cons = []
    for c in colunas:
        s = MONOTONIC_CONSTRAINTS.get(c, 0)
        if PREVER_PRECO_M2 and c in ('area_construida_m2', 'area_total_m2',
                                     'area_terreno_m2'):
            s = 0
        cons.append(s)
    if framework == 'xgboost':
        return '(' + ','.join(str(x) for x in cons) + ')'
    return cons


def preparar_dados(df):
    print("\n[1/6] Preparando dados...")
    y = df[TARGET].copy()
    X = df.drop(columns=[c for c in COLUNAS_EXCLUIR_MODELO + [TARGET]
                         if c in df.columns])

    strings_restantes = X.select_dtypes(include='object').columns.tolist()
    if strings_restantes:
        raise ValueError(
            f"Colunas string não tratadas chegaram ao modelo: {strings_restantes}.")
    nans = X.columns[X.isna().any()].tolist()
    if nans:
        raise ValueError(f"Features com NaN: {nans}.")

    print(f"Target: {TARGET}{' (modelado em R$/m²)' if PREVER_PRECO_M2 else ''}")
    print(f"Features: {X.shape[1]} colunas, {X.shape[0]:,} linhas")
    return X, y


def dividir_dados(X, y, df_original):
    """
    Split novo: TODOS os modelos treinam no MESMO conjunto (treino_full), para a
    comparação RF × boosters ser justa. A 'validação' some como camada de
    relatório — quem estima generalização é o evaluate.py (backtest). Os boosters
    usam uma CAUDA recente do treino só para early stopping (depois re-treinam
    no treino inteiro).
    """
    print("\n[2/6] Dividindo dados (divisão temporal)...")
    anos = df_original['ano_transacao'].values

    if TREINAR_APENAS_REGIME_NOVO:
        # ITEM 1: tudo dentro do regime novo. treino_full = 2023; teste = 2024.
        regime = classificar_regime(df_original['data_transacao'])
        mask_novo = regime == 'novo'
        mask_train_full = mask_novo & (anos <= 2023)
        mask_test = mask_novo & (anos >= 2024)
        print("  [regime novo] treino_full = 2023 | teste >= 2024")
    else:
        mask_train_full = anos <= 2022       # RF e boosters treinam aqui
        mask_test = anos >= 2023             # holdout do artigo
        print("  treino_full = 2008-2022 | teste >= 2023")

    X_train_full, y_train_full = X[mask_train_full], y[mask_train_full]
    X_test, y_test = X[mask_test], y[mask_test]

    # datas do treino_full, para recortar a cauda de early stopping por tempo
    datas_tf = pd.to_datetime(df_original.loc[mask_train_full, 'data_transacao'])

    total = len(X)
    print(f"Treino (todos): {len(X_train_full):>7,} "
          f"({len(X_train_full)/total*100:.1f}%)")
    print(f"Teste:          {len(X_test):>7,} ({len(X_test)/total*100:.1f}%)")
    return X_train_full, y_train_full, X_test, y_test, datas_tf


def _split_early_stopping(X_tf, y_tf, datas_tf, frac=FRAC_EARLY_STOPPING):
    """Separa a cauda mais recente (por data) do treino para early stopping."""
    ordem = np.argsort(datas_tf.values, kind='stable')
    n_tail = max(int(len(ordem) * frac), 1)
    idx_tail = ordem[-n_tail:]
    idx_fit = ordem[:-n_tail]
    return (X_tf.iloc[idx_fit], y_tf.iloc[idx_fit],
            X_tf.iloc[idx_tail], y_tf.iloc[idx_tail])


def calcular_metricas(y_true, y_pred, nome_modelo):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    erro_pct = np.abs((y_true - y_pred) / y_true) * 100
    return {
        'Modelo': nome_modelo,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': np.mean(erro_pct),
        'Mediana_Erro_%': np.median(erro_pct),
        'R²': r2_score(y_true, y_pred),
    }


def _reportar(metricas, etiqueta):
    print(f" [{etiqueta}] "
          f"MAE R$ {metricas['MAE']:,.2f} | RMSE R$ {metricas['RMSE']:,.2f} | "
          f"MAPE {metricas['MAPE']:.2f}% | Mediana {metricas['Mediana_Erro_%']:.2f}% | "
          f"R² {metricas['R²']:.4f}")


def treinar_random_forest(X_train_full, y_train_full, X_test, y_test):
    print("\n[3/6] Treinando Random Forest (treino 2008-2022)...")
    area_full = X_train_full['area_construida_m2'].values
    modelo = RandomForestRegressor(
        **HIPERPARAMETROS['random_forest'],
        random_state=42, n_jobs=-1, verbose=0,
    )
    modelo.fit(X_train_full, _y_para_modelo(y_train_full, area_full))

    y_pred_train = _pred_para_reais(modelo.predict(X_train_full), area_full)
    y_pred_test = _pred_para_reais(modelo.predict(X_test),
                                   X_test['area_construida_m2'].values)
    m_train = calcular_metricas(y_train_full, y_pred_train, 'Random Forest (Train)')
    m_test = calcular_metricas(y_test, y_pred_test, 'Random Forest')
    _reportar(m_train, 'train'); _reportar(m_test, 'TESTE')
    return modelo, m_train, m_test


def _treinar_booster(framework, X_tf, y_tf, datas_tf, X_test, y_test):
    """
    Passo 1: acha best_iteration via early stopping numa cauda recente do treino.
    Passo 2: RE-TREINA no treino_full inteiro (2008-2022) com esse n_estimators.
    Assim o booster treina nos MESMOS dados da RF (comparação justa).
    """
    X_fit, y_fit, X_es, y_es = _split_early_stopping(X_tf, y_tf, datas_tf)
    area_fit = X_fit['area_construida_m2'].values
    area_es = X_es['area_construida_m2'].values
    area_tf = X_tf['area_construida_m2'].values
    mono = _vetor_monotonia(list(X_tf.columns), framework)

    hp = dict(HIPERPARAMETROS[framework])
    hp.pop('n_estimators', None)

    # ── passo 1: early stopping na cauda ──
    if framework == 'xgboost':
        m_es = XGBRegressor(**hp, n_estimators=N_ESTIMATORS_MAX,
                            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                            monotone_constraints=mono,
                            random_state=42, n_jobs=-1, verbosity=0)
        m_es.fit(X_fit, _y_para_modelo(y_fit, area_fit),
                 eval_set=[(X_es, _y_para_modelo(y_es, area_es))], verbose=False)
        melhor_n = m_es.best_iteration + 1
    else:  # lightgbm
        m_es = LGBMRegressor(**hp, n_estimators=N_ESTIMATORS_MAX,
                             monotone_constraints=mono,
                             random_state=42, n_jobs=-1, verbose=-1)
        m_es.fit(X_fit, _y_para_modelo(y_fit, area_fit),
                 eval_set=[(X_es, _y_para_modelo(y_es, area_es))],
                 callbacks=[lgb_early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        melhor_n = m_es.best_iteration_
    print(f"  early stopping (cauda recente do treino) -> {melhor_n} árvores")

    # ── passo 2: re-treina no treino_full inteiro com melhor_n ──
    if framework == 'xgboost':
        modelo = XGBRegressor(**hp, n_estimators=melhor_n,
                              monotone_constraints=mono,
                              random_state=42, n_jobs=-1, verbosity=0)
    else:
        modelo = LGBMRegressor(**hp, n_estimators=melhor_n,
                               monotone_constraints=mono,
                               random_state=42, n_jobs=-1, verbose=-1)
    modelo.fit(X_tf, _y_para_modelo(y_tf, area_tf))

    y_pred_train = _pred_para_reais(modelo.predict(X_tf), area_tf)
    y_pred_test = _pred_para_reais(modelo.predict(X_test),
                                   X_test['area_construida_m2'].values)
    nome = 'XGBoost' if framework == 'xgboost' else 'LightGBM'
    m_train = calcular_metricas(y_tf, y_pred_train, f'{nome} (Train)')
    m_test = calcular_metricas(y_test, y_pred_test, nome)
    _reportar(m_train, 'train'); _reportar(m_test, 'TESTE')
    return modelo, m_train, m_test


def salvar_modelos_e_resultados(modelos, resultados_test, resultados_train):
    print("\n[6/6] Salvando modelos e resultados...")
    for nome, modelo in modelos.items():
        caminho = OUTPUTS_MODELS / f'{nome.lower().replace(" ", "_")}.pkl'
        with open(caminho, 'wb') as f:
            pickle.dump(modelo, f)
        print(f"Modelo salvo: {caminho.name}")

    df_test = pd.DataFrame(resultados_test)
    df_test.to_csv(OUTPUTS_TABLES / 'resultados_modelos.csv', index=False)
    pd.DataFrame(resultados_train).to_csv(
        OUTPUTS_TABLES / 'resultados_modelos_train.csv', index=False)
    print("Tabelas (teste, train) salvas.")
    return df_test


def pipeline_completo(path_input=None):
    if path_input is None:
        path_input = ITBI_FINAL
    print("\nPIPELINE DE TREINAMENTO DE MODELOS")
    print(f"\nCarregando dados de: {path_input}")
    df = pd.read_csv(path_input)
    print(f"{len(df):,} linhas carregadas")

    X, y = preparar_dados(df)
    X_tf, y_tf, X_test, y_test, datas_tf = dividir_dados(X, y, df)

    rf_model, rf_train, rf_test = treinar_random_forest(X_tf, y_tf, X_test, y_test)

    print("\n[4/6] Treinando XGBoost (early stopping + re-treino no treino full)...")
    xgb_model, xgb_train, xgb_test = _treinar_booster(
        'xgboost', X_tf, y_tf, datas_tf, X_test, y_test)

    print("\n[5/6] Treinando LightGBM (early stopping + re-treino no treino full)...")
    lgb_model, lgb_train, lgb_test = _treinar_booster(
        'lightgbm', X_tf, y_tf, datas_tf, X_test, y_test)

    # ENSEMBLE (média XGB+LGBM em R$)
    area_test = X_test['area_construida_m2'].values
    pred_ens = (_pred_para_reais(xgb_model.predict(X_test), area_test) +
                _pred_para_reais(lgb_model.predict(X_test), area_test)) / 2.0
    ens_test = calcular_metricas(y_test, pred_ens, 'Ensemble (XGB+LGBM)')
    _reportar(ens_test, 'TESTE')

    modelos = {'Random Forest': rf_model, 'XGBoost': xgb_model, 'LightGBM': lgb_model}
    resultados_test = [rf_test, xgb_test, lgb_test, ens_test]
    resultados_train = [rf_train, xgb_train, lgb_train]

    df_test = salvar_modelos_e_resultados(modelos, resultados_test, resultados_train)

    pd.set_option('display.float_format', lambda v: f'{v:,.2f}')
    print("\n" + "=" * 80)
    print("RESULTADOS NO TESTE — métricas do artigo")
    print("=" * 80)
    print(df_test.to_string(index=False))

    campeao = min(resultados_test, key=lambda m: m['Mediana_Erro_%'])
    print("\n" + "=" * 80)
    print(f"CAMPEÃO (menor mediana de erro): {campeao['Modelo']} "
          f"— MAPE {campeao['MAPE']:.2f}%, Mediana {campeao['Mediana_Erro_%']:.2f}%, "
          f"R² {campeao['R²']:.4f}")
    print("=" * 80)
    print("\nLembrete: o número HONESTO de generalização é o backtest do "
          "evaluate.py. Este teste 2023-2024 é o holdout único do artigo.")
    print("TREINAMENTO CONCLUÍDO!")
    return modelos, df_test, (X_test, y_test)


if __name__ == '__main__':
    modelos, resultados, dados_teste = pipeline_completo()