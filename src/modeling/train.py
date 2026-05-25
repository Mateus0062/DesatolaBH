import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    ITBI_FINAL,
    OUTPUTS_MODELS, OUTPUTS_TABLES
)

def preparar_dados(df):
    from config import TARGET, COLUNAS_EXCLUIR_MODELO

    print("\n[1/6] Preparando dados...")

    y = df[TARGET].copy()
    X = df.drop(columns=[c for c in COLUNAS_EXCLUIR_MODELO + [TARGET]
                         if c in df.columns])

    # Trava de segurança: nenhuma string pode passar daqui.
    strings_restantes = X.select_dtypes(include='object').columns.tolist()
    if strings_restantes:
        raise ValueError(
            f"Colunas string não tratadas chegaram ao modelo: "
            f"{strings_restantes}. Adicione-as a COLUNAS_EXCLUIR_MODELO "
            f"no config.py ou codifique-as numericamente."
        )

    # Trava 2: nenhum NaN silencioso.
    nans = X.columns[X.isna().any()].tolist()
    if nans:
        raise ValueError(f"Features com NaN: {nans}. Trate antes de treinar.")

    print(f"  ✓ Target: {TARGET}")
    print(f"  ✓ Features: {X.shape[1]} colunas, {X.shape[0]:,} linhas")
    return X, y

def dividir_dados(X, y, df_original):
    print("\n[2/6] Dividindo dados (divisão temporal)...")

    # Adicionar coluna ano_transacao temporariamente para filtrar
    anos = df_original['ano_transacao'].values

    # Criar máscaras temporais
    mask_train = anos <= 2021
    mask_val = anos == 2022
    mask_test = anos >= 2023

    # Dividir
    X_train = X[mask_train]
    y_train = y[mask_train]

    X_val = X[mask_val]
    y_val = y[mask_val]

    X_test = X[mask_test]
    y_test = y[mask_test]

    # Estatísticas
    total = len(X)
    print(f"Treino (2008-2021):  {len(X_train):>7,} linhas ({len(X_train) / total * 100:.1f}%)")
    print(f"Validação (2022):    {len(X_val):>7,} linhas ({len(X_val) / total * 100:.1f}%)")
    print(f"Teste (2023-2024):   {len(X_test):>7,} linhas ({len(X_test) / total * 100:.1f}%)")

    print(f"\nDistribuição temporal:")
    print(f"Treino:    {anos[mask_train].min():.0f} - {anos[mask_train].max():.0f}")
    print(f"Validação: {anos[mask_val].min():.0f} - {anos[mask_val].max():.0f}")
    print(f"Teste:     {anos[mask_test].min():.0f} - {anos[mask_test].max():.0f}")


    return X_train, X_val, X_test, y_train, y_val, y_test

def calcular_metricas(y_true, y_pred, nome_modelo):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return {
        'Modelo': nome_modelo,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R²': r2
    }

def treinar_random_forest(X_train, y_train, X_val, y_val):
    print("\n[3/6] Treinando Random Forest...")

    modelo = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    modelo.fit(X_train, y_train)

    # Predições
    y_pred_train = modelo.predict(X_train)
    y_pred_val = modelo.predict(X_val)

    # Métricas
    metricas_train = calcular_metricas(y_train, y_pred_train, 'Random Forest (Train)')
    metricas_val = calcular_metricas(y_val, y_pred_val, 'Random Forest (Val)')

    print(f"MAE (val): R$ {metricas_val['MAE']:,.2f}")
    print(f"RMSE (val): R$ {metricas_val['RMSE']:,.2f}")
    print(f"MAPE (val): {metricas_val['MAPE']:.2f}%")
    print(f"R² (val): {metricas_val['R²']:.4f}")

    return modelo, metricas_train, metricas_val

def treinar_xgboost(X_train, y_train, X_val, y_val):
    print("\n[4/6] Treinando XGBoost...")

    modelo = XGBRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )

    modelo.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predições
    y_pred_train = modelo.predict(X_train)
    y_pred_val = modelo.predict(X_val)

    # Métricas
    metricas_train = calcular_metricas(y_train, y_pred_train, 'XGBoost (Train)')
    metricas_val = calcular_metricas(y_val, y_pred_val, 'XGBoost (Val)')

    print(f"MAE (val):R$ {metricas_val['MAE']:,.2f}")
    print(f"RMSE (val): R$ {metricas_val['RMSE']:,.2f}")
    print(f"MAPE (val): {metricas_val['MAPE']:.2f}%")
    print(f"R² (val): {metricas_val['R²']:.4f}")

    return modelo, metricas_train, metricas_val

def treinar_lightgbm(X_train, y_train, X_val, y_val):
    print("\n[5/6] Treinando LightGBM...")

    modelo = LGBMRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    modelo.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=None  # silenciar output
    )

    # Predições
    y_pred_train = modelo.predict(X_train)
    y_pred_val = modelo.predict(X_val)

    # Métricas
    metricas_train = calcular_metricas(y_train, y_pred_train, 'LightGBM (Train)')
    metricas_val = calcular_metricas(y_val, y_pred_val, 'LightGBM (Val)')

    print(f"MAE (val): R$ {metricas_val['MAE']:,.2f}")
    print(f"RMSE (val): R$ {metricas_val['RMSE']:,.2f}")
    print(f"MAPE (val): {metricas_val['MAPE']:.2f}%")
    print(f"R² (val): {metricas_val['R²']:.4f}")

    return modelo, metricas_train, metricas_val

def salvar_modelos_e_resultados(modelos, resultados):
    print("\n[6/6] Salvando modelos e resultados...")

    # Salvar modelos
    for nome, modelo in modelos.items():
        caminho = OUTPUTS_MODELS / f'{nome.lower().replace(" ", "_")}.pkl'
        with open(caminho, 'wb') as f:
            pickle.dump(modelo, f)
        print(f"Modelo salvo: {caminho.name}")

    # Salvar tabela de resultados
    df_resultados = pd.DataFrame(resultados)
    caminho_tabela = OUTPUTS_TABLES / 'resultados_modelos.csv'
    df_resultados.to_csv(caminho_tabela, index=False)
    print(f"Tabela salva: {caminho_tabela.name}")

    return df_resultados

def pipeline_completo(path_input=None):
    # Carregar dados
    if path_input is None:
        path_input = ITBI_FINAL

    print("\nPIPELINE DE TREINAMENTO DE MODELOS")
    print(f"\nCarregando dados de: {path_input}")

    df = pd.read_csv(path_input)
    print(f"{len(df):,} linhas carregadas")

    # Preparar dados
    X, y = preparar_dados(df)

    # Dividir dados
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_dados(X, y, df)

    # Treinar modelos
    rf_model, rf_train, rf_val = treinar_random_forest(X_train, y_train, X_val, y_val)
    xgb_model, xgb_train, xgb_val = treinar_xgboost(X_train, y_train, X_val, y_val)
    lgb_model, lgb_train, lgb_val = treinar_lightgbm(X_train, y_train, X_val, y_val)

    # Compilar resultados
    modelos = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'LightGBM': lgb_model
    }

    resultados = [rf_val, xgb_val, lgb_val]

    # Salvar
    df_resultados = salvar_modelos_e_resultados(modelos, resultados)

    # Exibir resumo
    print("\nRESUMO DOS RESULTADOS (VALIDAÇÃO)")
    print(df_resultados.to_string(index=False))

    print("\nTREINAMENTO CONCLUÍDO!")
    print(f"\nModelos salvos em: {OUTPUTS_MODELS}")
    print(f"Resultados salvos em: {OUTPUTS_TABLES}")

    return modelos, df_resultados, (X_test, y_test)

if __name__ == '__main__':
    modelos, resultados, dados_teste = pipeline_completo()