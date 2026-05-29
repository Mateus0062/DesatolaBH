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
N_ESTIMATORS_MAX = 2000

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
        # em R$/m² a relação com área deixa de ser monotônica
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
    print("\n[2/6] Dividindo dados (divisão temporal)...")
    anos = df_original['ano_transacao'].values

    if TREINAR_APENAS_REGIME_NOVO:
        # ITEM 1: split DENTRO do regime novo (>=2023). 2023 = treino+val,
        # 2024 = teste. Evita misturar o regime antigo (alvo de outra natureza).
        regime = classificar_regime(df_original['data_transacao'])
        mask_novo = regime == 'novo'
        mask_train = mask_novo & (anos <= 2023)
        mask_val = mask_novo & (anos == 2023)   # val = parte recente de 2023
        # refinamento simples: val = 2º semestre de 2023
        mes = df_original['mes_transacao'].values
        mask_val = mask_novo & (anos == 2023) & (mes >= 7)
        mask_train = mask_novo & ((anos < 2023) | ((anos == 2023) & (mes < 7)))
        mask_test = mask_novo & (anos >= 2024)
        print("  [regime novo] treino<=2023H1 | val=2023H2 | teste>=2024")
    else:
        mask_train = anos <= 2021
        mask_val = anos == 2022
        mask_test = anos >= 2023

    X_train, y_train = X[mask_train], y[mask_train]
    X_val, y_val = X[mask_val], y[mask_val]
    X_test, y_test = X[mask_test], y[mask_test]
    X_train_full = X[mask_train | mask_val]
    y_train_full = y[mask_train | mask_val]

    total = len(X)
    print(f"Treino:    {len(X_train):>7,} ({len(X_train)/total*100:.1f}%)")
    print(f"Validação: {len(X_val):>7,} ({len(X_val)/total*100:.1f}%)")
    print(f"Teste:     {len(X_test):>7,} ({len(X_test)/total*100:.1f}%)")
    print(f"\nRF usará treino+val: {len(X_train_full):,} linhas")
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            X_train_full, y_train_full)

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


def treinar_random_forest(X_train_full, y_train_full, X_val, y_val, X_test, y_test):
    print("\n[3/6] Treinando Random Forest...")
    area_full = X_train_full['area_construida_m2'].values
    modelo = RandomForestRegressor(
        **HIPERPARAMETROS['random_forest'],
        random_state=42, n_jobs=-1, verbose=0,
    )
    modelo.fit(X_train_full, _y_para_modelo(y_train_full, area_full))

    y_pred_train = _pred_para_reais(modelo.predict(X_train_full), area_full)
    y_pred_val = _pred_para_reais(modelo.predict(X_val), X_val['area_construida_m2'].values)
    y_pred_test = _pred_para_reais(modelo.predict(X_test), X_test['area_construida_m2'].values)

    m_train = calcular_metricas(y_train_full, y_pred_train, 'Random Forest (Train)')
    m_val = calcular_metricas(y_val, y_pred_val, 'Random Forest (Val)')
    m_test = calcular_metricas(y_test, y_pred_test, 'Random Forest')
    _reportar(m_train, 'train'); _reportar(m_val, 'val '); _reportar(m_test, 'TESTE')
    return modelo, m_train, m_val, m_test

def treinar_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n[4/6] Treinando XGBoost (monotonia + regularizado)...")
    hp = dict(HIPERPARAMETROS['xgboost'])
    hp.pop('n_estimators', None)
    area_tr = X_train['area_construida_m2'].values
    modelo = XGBRegressor(
        **hp, n_estimators=N_ESTIMATORS_MAX,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        monotone_constraints=_vetor_monotonia(list(X_train.columns), 'xgboost'),
        random_state=42, n_jobs=-1, verbosity=0,
    )
    modelo.fit(
        X_train, _y_para_modelo(y_train, area_tr),
        eval_set=[(X_val, _y_para_modelo(y_val, X_val['area_construida_m2'].values))],
        verbose=False,
    )
    print(f"Árvores usadas (early stopping): {modelo.best_iteration + 1}")

    y_pred_train = _pred_para_reais(modelo.predict(X_train), area_tr)
    y_pred_val = _pred_para_reais(modelo.predict(X_val), X_val['area_construida_m2'].values)
    y_pred_test = _pred_para_reais(modelo.predict(X_test), X_test['area_construida_m2'].values)

    m_train = calcular_metricas(y_train, y_pred_train, 'XGBoost (Train)')
    m_val = calcular_metricas(y_val, y_pred_val, 'XGBoost (Val)')
    m_test = calcular_metricas(y_test, y_pred_test, 'XGBoost')
    _reportar(m_train, 'train'); _reportar(m_val, 'val'); _reportar(m_test, 'TESTE')
    return modelo, m_train, m_val, m_test

def treinar_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\n[5/6] Treinando LightGBM (monotonia + regularizado)...")
    hp = dict(HIPERPARAMETROS['lightgbm'])
    hp.pop('n_estimators', None)
    area_tr = X_train['area_construida_m2'].values
    modelo = LGBMRegressor(
        **hp, n_estimators=N_ESTIMATORS_MAX,
        monotone_constraints=_vetor_monotonia(list(X_train.columns), 'lightgbm'),
        random_state=42, n_jobs=-1, verbose=-1,
    )
    modelo.fit(
        X_train, _y_para_modelo(y_train, area_tr),
        eval_set=[(X_val, _y_para_modelo(y_val, X_val['area_construida_m2'].values))],
        callbacks=[lgb_early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    print(f" Árvores usadas (early stopping): {modelo.best_iteration_}")

    y_pred_train = _pred_para_reais(modelo.predict(X_train), area_tr)
    y_pred_val = _pred_para_reais(modelo.predict(X_val), X_val['area_construida_m2'].values)
    y_pred_test = _pred_para_reais(modelo.predict(X_test), X_test['area_construida_m2'].values)

    m_train = calcular_metricas(y_train, y_pred_train, 'LightGBM (Train)')
    m_val = calcular_metricas(y_val, y_pred_val, 'LightGBM (Val)')
    m_test = calcular_metricas(y_test, y_pred_test, 'LightGBM')
    _reportar(m_train, 'train'); _reportar(m_val, 'val '); _reportar(m_test, 'TESTE')
    return modelo, m_train, m_val, m_test

def salvar_modelos_e_resultados(modelos, resultados_test, resultados_val,
                                resultados_train):
    print("\n[6/6] Salvando modelos e resultados...")
    for nome, modelo in modelos.items():
        caminho = OUTPUTS_MODELS / f'{nome.lower().replace(" ", "_")}.pkl'
        with open(caminho, 'wb') as f:
            pickle.dump(modelo, f)
        print(f"Modelo salvo: {caminho.name}")

    df_test = pd.DataFrame(resultados_test)
    df_test.to_csv(OUTPUTS_TABLES / 'resultados_modelos.csv', index=False)
    pd.DataFrame(resultados_val).to_csv(
        OUTPUTS_TABLES / 'resultados_modelos_val.csv', index=False)
    pd.DataFrame(resultados_train).to_csv(
        OUTPUTS_TABLES / 'resultados_modelos_train.csv', index=False)
    print("Tabelas (teste, val, train) salvas.")
    return df_test

def pipeline_completo(path_input=None):
    if path_input is None:
        path_input = ITBI_FINAL
    print("\nPIPELINE DE TREINAMENTO DE MODELOS")
    print(f"\nCarregando dados de: {path_input}")
    df = pd.read_csv(path_input)
    print(f"{len(df):,} linhas carregadas")

    X, y = preparar_dados(df)
    (X_train, y_train, X_val, y_val, X_test, y_test,
     X_train_full, y_train_full) = dividir_dados(X, y, df)

    rf_model, rf_train, rf_val, rf_test = treinar_random_forest(
        X_train_full, y_train_full, X_val, y_val, X_test, y_test)
    xgb_model, xgb_train, xgb_val, xgb_test = treinar_xgboost(
        X_train, y_train, X_val, y_val, X_test, y_test)
    lgb_model, lgb_train, lgb_val, lgb_test = treinar_lightgbm(
        X_train, y_train, X_val, y_val, X_test, y_test)

    # ITEM 5: ENSEMBLE (média XGB+LGBM em R$). Modelos empatados -> média estável.
    area_test = X_test['area_construida_m2'].values
    pred_ens = (_pred_para_reais(xgb_model.predict(X_test), area_test) +
                _pred_para_reais(lgb_model.predict(X_test), area_test)) / 2.0
    ens_test = calcular_metricas(y_test, pred_ens, 'Ensemble (XGB+LGBM)')
    _reportar(ens_test, 'TESTE')

    modelos = {'Random Forest': rf_model, 'XGBoost': xgb_model, 'LightGBM': lgb_model}
    resultados_test = [rf_test, xgb_test, lgb_test, ens_test]
    resultados_val = [rf_val, xgb_val, lgb_val]
    resultados_train = [rf_train, xgb_train, lgb_train]

    df_test = salvar_modelos_e_resultados(
        modelos, resultados_test, resultados_val, resultados_train)

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
    print("\nTREINAMENTO CONCLUÍDO!")
    return modelos, df_test, (X_test, y_test)

if __name__ == '__main__':
    modelos, resultados, dados_teste = pipeline_completo()