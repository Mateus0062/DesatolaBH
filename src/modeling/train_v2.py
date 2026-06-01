import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_FINAL, OUTPUTS_MODELS, OUTPUTS_TABLES, TARGET

# Reaproveita TUDO do train.py (mesmo modelo, mesmas travas, mesma monotonia).
from src.modeling.train import (
    preparar_dados, _split_early_stopping, treinar_random_forest,
    _treinar_booster, _pred_para_reais, calcular_metricas, _reportar,
)

# Saídas próprias — não tocam nos arquivos do train.py.
OUT_MODELS_V2 = OUTPUTS_MODELS / 'treino_ate_2023'
OUT_TABLES_V2 = OUTPUTS_TABLES / 'treino_ate_2023'
OUT_MODELS_V2.mkdir(parents=True, exist_ok=True)
OUT_TABLES_V2.mkdir(parents=True, exist_ok=True)


def dividir_dados_v2(X, y, df_original):
    print("\n[2/6] Dividindo dados (cenário operação: treino<=2023, teste=2024)...")
    anos = df_original['ano_transacao'].values
    mask_train_full = anos <= 2023
    mask_test = anos == 2024

    X_tf, y_tf = X[mask_train_full], y[mask_train_full]
    X_test, y_test = X[mask_test], y[mask_test]
    datas_tf = pd.to_datetime(df_original.loc[mask_train_full, 'data_transacao'])

    total = len(X)
    print(f"Treino (todos, 2008-2023): {len(X_tf):>7,} "
          f"({len(X_tf)/total*100:.1f}%)")
    print(f"Teste (2024):              {len(X_test):>7,} "
          f"({len(X_test)/total*100:.1f}%)")
    if len(X_test) < 100:
        raise ValueError("Pouquíssimas linhas de 2024 no dataset — confira o "
                         "filtro/recorte temporal antes de usar este cenário.")
    return X_tf, y_tf, X_test, y_test, datas_tf


def salvar_v2(modelos, resultados_test, resultados_train):
    print("\n[6/6] Salvando modelos e resultados (subpasta treino_ate_2023)...")
    for nome, modelo in modelos.items():
        caminho = OUT_MODELS_V2 / f'{nome.lower().replace(" ", "_")}.pkl'
        with open(caminho, 'wb') as f:
            pickle.dump(modelo, f)
        print(f"Modelo salvo: {caminho.relative_to(OUTPUTS_MODELS)}")
    df_test = pd.DataFrame(resultados_test)
    df_test.to_csv(OUT_TABLES_V2 / 'resultados_modelos.csv', index=False)
    pd.DataFrame(resultados_train).to_csv(
        OUT_TABLES_V2 / 'resultados_modelos_train.csv', index=False)
    print("Tabelas (teste, train) salvas em treino_ate_2023/.")
    return df_test


def pipeline_v2(path_input=None):
    if path_input is None:
        path_input = ITBI_FINAL
    print("\nPIPELINE DE TREINAMENTO — CENÁRIO OPERAÇÃO (treino<=2023, teste 2024)")
    print(f"\nCarregando dados de: {path_input}")
    df = pd.read_csv(path_input)
    print(f"{len(df):,} linhas carregadas")

    X, y = preparar_dados(df)
    X_tf, y_tf, X_test, y_test, datas_tf = dividir_dados_v2(X, y, df)

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
    print("\n[ENSEMBLE]")
    _reportar(ens_test, 'TESTE')

    modelos = {'Random Forest': rf_model, 'XGBoost': xgb_model, 'LightGBM': lgb_model}
    resultados_test = [rf_test, xgb_test, lgb_test, ens_test]
    resultados_train = [rf_train, xgb_train, lgb_train]

    df_test = salvar_v2(modelos, resultados_test, resultados_train)

    pd.set_option('display.float_format', lambda v: f'{v:,.2f}')
    print("\n" + "=" * 80)
    print("RESULTADOS NO TESTE 2024 — cenário OPERAÇÃO (modelo já viu o regime novo)")
    print("=" * 80)
    print(df_test.to_string(index=False))

    campeao = min(resultados_test, key=lambda m: m['Mediana_Erro_%'])
    print("\n" + "=" * 80)
    print(f"CAMPEÃO (menor mediana de erro): {campeao['Modelo']} "
          f"— MAPE {campeao['MAPE']:.2f}%, Mediana {campeao['Mediana_Erro_%']:.2f}%, "
          f"R² {campeao['R²']:.4f}")
    print("=" * 80)
    print("\nEste é o cenário de OPERAÇÃO (use estes .pkl para produção).")
    print("Para o número de extrapolação honesta (treino<=2022, teste 2023-2024),")
    print("use o train.py. Para a régua com barra de erro, use o evaluate.py.")
    print("TREINAMENTO V2 CONCLUÍDO!")
    return modelos, df_test, (X_test, y_test)


if __name__ == '__main__':
    modelos, resultados, dados_teste = pipeline_v2()