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
    OUTPUTS_MODELS, OUTPUTS_TABLES, HIPERPARAMETROS
)

# Rondas de paciência para early stopping (boosters).
EARLY_STOPPING_ROUNDS = 50

# Teto generoso para n_estimators dos boosters quando há early stopping;
# o modelo vai parar antes naturalmente se a validação não melhorar.
N_ESTIMATORS_MAX = 2000


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
            f"{strings_restantes}."
        )

    # Trava 2: nenhum NaN silencioso.
    nans = X.columns[X.isna().any()].tolist()
    if nans:
        raise ValueError(f"Features com NaN: {nans}.")

    print(f"Target: {TARGET}")
    print(f"Features: {X.shape[1]} colunas, {X.shape[0]:,} linhas")
    return X, y


def dividir_dados(X, y, df_original):
    """
    Divisão temporal em três blocos, usada de forma diferente por modelo:

      - Treino (2008-2021): usado por todos os modelos.
      - Validação (2022): usado APENAS pelos boosters, como eval_set para
        early stopping. O RF (sem early stopping nativo) é treinado em
        treino+validação juntos para não desperdiçar dados.
      - Teste (2023-2024): intocado, avaliação final de todos os modelos.

    Essa assimetria é declarada explicitamente no relatório.
    """
    print("\n[2/6] Dividindo dados (divisão temporal)...")

    anos = df_original['ano_transacao'].values

    mask_train = anos <= 2021
    mask_val = anos == 2022
    mask_test = anos >= 2023

    X_train, y_train = X[mask_train], y[mask_train]
    X_val, y_val = X[mask_val], y[mask_val]
    X_test, y_test = X[mask_test], y[mask_test]

    # Treino+val unido para o RF.
    X_train_full = X[mask_train | mask_val]
    y_train_full = y[mask_train | mask_val]

    total = len(X)
    print(f"Treino (2008-2021): {len(X_train):>7,} linhas "
          f"({len(X_train) / total * 100:.1f}%)")
    print(f"Validação (2022):   {len(X_val):>7,} linhas "
          f"({len(X_val) / total * 100:.1f}%)")
    print(f"Teste (2023-2024):  {len(X_test):>7,} linhas "
          f"({len(X_test) / total * 100:.1f}%)")
    print(f"\nRF usará treino+val juntos: {len(X_train_full):,} linhas")
    print(f"Boosters usarão treino + early stopping na validação")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            X_train_full, y_train_full)


def calcular_metricas(y_true, y_pred, nome_modelo):
    """
    Métricas SEMPRE em reais. Inclui média (MAPE) e mediana do erro
    percentual: o MAPE é puxado por poucos imóveis baratos com erro alto,
    a mediana revela o erro percentual 'típico'.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    erro_pct = np.abs((y_true - y_pred) / y_true) * 100

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(erro_pct)
    mediana_pct = np.median(erro_pct)
    r2 = r2_score(y_true, y_pred)

    return {
        'Modelo': nome_modelo,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Mediana_Erro_%': mediana_pct,
        'R²': r2,
    }


def _reportar(metricas, etiqueta):
    print(f"  [{etiqueta}] "
          f"MAE R$ {metricas['MAE']:,.2f} | "
          f"RMSE R$ {metricas['RMSE']:,.2f} | "
          f"MAPE {metricas['MAPE']:.2f}% | "
          f"Mediana {metricas['Mediana_Erro_%']:.2f}% | "
          f"R² {metricas['R²']:.4f}")


def treinar_random_forest(X_train_full, y_train_full, X_val, y_val, X_test, y_test):
    """
    RF não tem early stopping nativo. Treina em treino+validação (2008-2022)
    para não desperdiçar dados. Reporta validação e teste para inspeção.
    """
    print("\n[3/6] Treinando Random Forest (treino+val 2008-2022)...")

    modelo = RandomForestRegressor(
        **HIPERPARAMETROS['random_forest'],
        random_state=42, n_jobs=-1, verbose=0,
    )

    modelo.fit(X_train_full, np.log1p(y_train_full))

    y_pred_train = np.expm1(modelo.predict(X_train_full))
    y_pred_val = np.expm1(modelo.predict(X_val))
    y_pred_test = np.expm1(modelo.predict(X_test))

    m_train = calcular_metricas(y_train_full, y_pred_train, 'Random Forest (Train)')
    m_val = calcular_metricas(y_val, y_pred_val, 'Random Forest (Val)')
    m_test = calcular_metricas(y_test, y_pred_test, 'Random Forest')

    _reportar(m_train, 'train')
    _reportar(m_val, 'val  ')
    _reportar(m_test, 'TESTE')

    return modelo, m_train, m_val, m_test


def treinar_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    XGBoost com early stopping em 2022. Hiperparâmetros do tuning são
    aplicados, com n_estimators expandido para 2000 (teto) — o early
    stopping corta antes naturalmente.
    """
    print(f"\n[4/6] Treinando XGBoost (treino 2008-2021, early stopping em 2022)...")

    hp = dict(HIPERPARAMETROS['xgboost'])
    hp.pop('n_estimators', None)  # Vamos sobrescrever com o teto.

    modelo = XGBRegressor(
        **hp,
        n_estimators=N_ESTIMATORS_MAX,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        random_state=42, n_jobs=-1, verbosity=0,
    )

    # eval_set em log: validação na mesma escala do treino.
    modelo.fit(
        X_train, np.log1p(y_train),
        eval_set=[(X_val, np.log1p(y_val))],
        verbose=False,
    )

    print(f"  Árvores efetivamente usadas (early stopping cortou em): "
          f"{modelo.best_iteration + 1}")

    y_pred_train = np.expm1(modelo.predict(X_train))
    y_pred_val = np.expm1(modelo.predict(X_val))
    y_pred_test = np.expm1(modelo.predict(X_test))

    m_train = calcular_metricas(y_train, y_pred_train, 'XGBoost (Train)')
    m_val = calcular_metricas(y_val, y_pred_val, 'XGBoost (Val)')
    m_test = calcular_metricas(y_test, y_pred_test, 'XGBoost')

    _reportar(m_train, 'train')
    _reportar(m_val, 'val  ')
    _reportar(m_test, 'TESTE')

    return modelo, m_train, m_val, m_test


def treinar_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    LightGBM com early stopping em 2022. n_estimators expandido para 2000;
    o callback early_stopping da lightgbm corta antes naturalmente.
    """
    print(f"\n[5/6] Treinando LightGBM (treino 2008-2021, early stopping em 2022)...")

    hp = dict(HIPERPARAMETROS['lightgbm'])
    hp.pop('n_estimators', None)

    modelo = LGBMRegressor(
        **hp,
        n_estimators=N_ESTIMATORS_MAX,
        random_state=42, n_jobs=-1, verbose=-1,
    )

    modelo.fit(
        X_train, np.log1p(y_train),
        eval_set=[(X_val, np.log1p(y_val))],
        callbacks=[lgb_early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    print(f"  Árvores efetivamente usadas (early stopping cortou em): "
          f"{modelo.best_iteration_}")

    y_pred_train = np.expm1(modelo.predict(X_train))
    y_pred_val = np.expm1(modelo.predict(X_val))
    y_pred_test = np.expm1(modelo.predict(X_test))

    m_train = calcular_metricas(y_train, y_pred_train, 'LightGBM (Train)')
    m_val = calcular_metricas(y_val, y_pred_val, 'LightGBM (Val)')
    m_test = calcular_metricas(y_test, y_pred_test, 'LightGBM')

    _reportar(m_train, 'train')
    _reportar(m_val, 'val  ')
    _reportar(m_test, 'TESTE')

    return modelo, m_train, m_val, m_test


def salvar_modelos_e_resultados(modelos, resultados_test, resultados_val,
                                resultados_train):
    print("\n[6/6] Salvando modelos e resultados...")

    for nome, modelo in modelos.items():
        caminho = OUTPUTS_MODELS / f'{nome.lower().replace(" ", "_")}.pkl'
        with open(caminho, 'wb') as f:
            pickle.dump(modelo, f)
        print(f"Modelo salvo: {caminho.name}")

    # Tabela principal: métricas de TESTE.
    df_test = pd.DataFrame(resultados_test)
    df_test.to_csv(OUTPUTS_TABLES / 'resultados_modelos.csv', index=False)
    print(f"Tabela (teste) salva: resultados_modelos.csv")

    # Tabelas auxiliares: validação e treino.
    pd.DataFrame(resultados_val).to_csv(
        OUTPUTS_TABLES / 'resultados_modelos_val.csv', index=False)
    pd.DataFrame(resultados_train).to_csv(
        OUTPUTS_TABLES / 'resultados_modelos_train.csv', index=False)
    print(f"Tabelas (val, train) salvas.")

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

    modelos = {
        'Random Forest': rf_model,
        'XGBoost': xgb_model,
        'LightGBM': lgb_model,
    }

    resultados_test = [rf_test, xgb_test, lgb_test]
    resultados_val = [rf_val, xgb_val, lgb_val]
    resultados_train = [rf_train, xgb_train, lgb_train]

    df_test = salvar_modelos_e_resultados(
        modelos, resultados_test, resultados_val, resultados_train)

    pd.set_option('display.float_format', lambda v: f'{v:,.2f}')

    print("\n" + "=" * 80)
    print("RESULTADOS NO TESTE (2023-2024) — métricas do artigo")
    print("=" * 80)
    print(df_test.to_string(index=False))

    print("\n" + "=" * 80)
    print("RESULTADOS NA VALIDAÇÃO (2022) — sanidade pré-teste")
    print("=" * 80)
    print(pd.DataFrame(resultados_val).to_string(index=False))

    print("\n" + "=" * 80)
    print("RESULTADOS NO TREINO — checagem de overfitting")
    print("=" * 80)
    print(pd.DataFrame(resultados_train).to_string(index=False))

    campeao = min(resultados_test, key=lambda m: m['MAPE'])
    print("\n" + "=" * 80)
    print(f"CAMPEÃO (menor MAPE de teste): {campeao['Modelo']} "
          f"— MAPE {campeao['MAPE']:.2f}%, "
          f"Mediana {campeao['Mediana_Erro_%']:.2f}%, "
          f"R² {campeao['R²']:.4f}")
    print("=" * 80)

    print("\nTREINAMENTO CONCLUÍDO!")
    print(f"\nModelos salvos em: {OUTPUTS_MODELS}")
    print(f"Resultados salvos em: {OUTPUTS_TABLES}")

    return modelos, df_test, (X_test, y_test)


if __name__ == '__main__':
    modelos, resultados, dados_teste = pipeline_completo()