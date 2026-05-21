import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import OUTPUTS_MODELS, OUTPUTS_TABLES


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

def avaliar_modelos_teste(X_test, y_test):
    print("=" * 80)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("=" * 80)
    print(f"\nLinhas de teste: {len(X_test):,}")

    # Carregar modelos
    modelos = {}
    for nome_arquivo in ['random_forest.pkl', 'xgboost.pkl', 'lightgbm.pkl']:
        caminho = OUTPUTS_MODELS / nome_arquivo
        with open(caminho, 'rb') as f:
            modelo = pickle.load(f)
        nome = nome_arquivo.replace('.pkl', '').replace('_', ' ').title()
        modelos[nome] = modelo
        print(f"  ✓ {nome} carregado")

    # Avaliar cada modelo
    resultados = []

    for nome, modelo in modelos.items():
        print(f"\nAvaliando {nome}...")
        y_pred = modelo.predict(X_test)

        metricas = calcular_metricas(y_test, y_pred, nome)
        resultados.append(metricas)

        print(f"  MAE:  R$ {metricas['MAE']:,.2f}")
        print(f"  RMSE: R$ {metricas['RMSE']:,.2f}")
        print(f"  MAPE: {metricas['MAPE']:.2f}%")
        print(f"  R²:   {metricas['R²']:.4f}")

    # Criar DataFrame
    df_resultados = pd.DataFrame(resultados)

    # Salvar
    caminho_saida = OUTPUTS_TABLES / 'resultados_teste.csv'
    df_resultados.to_csv(caminho_saida, index=False)

    print("\n" + "=" * 80)
    print("RESUMO - CONJUNTO DE TESTE")
    print("=" * 80)
    print(df_resultados.to_string(index=False))

    # Comparar com validação
    print("\n" + "=" * 80)
    print("COMPARAÇÃO: VALIDAÇÃO vs TESTE")
    print("=" * 80)

    df_val = pd.read_csv(OUTPUTS_TABLES / 'resultados_modelos.csv')

    for i, row_teste in df_resultados.iterrows():
        nome = row_teste['Modelo']
        filtro = df_val['Modelo'].str.contains(nome.split()[0], case=False, na=False)
        if filtro.any():
            row_val = df_val[filtro].iloc[0]
        else:
            print(f"  ⚠️  Modelo '{nome}' não encontrado nos resultados de validação")
            continue

        print(f"\n{nome}:")
        print(f"  MAE:  Val={row_val['MAE']:>10,.2f}  |  Test={row_teste['MAE']:>10,.2f}")
        print(f"  R²:   Val={row_val['R²']:>10.4f}  |  Test={row_teste['R²']:>10.4f}")

        # Detectar overfitting
        diff_r2 = row_val['R²'] - row_teste['R²']
        if diff_r2 > 0.05:
            print(f"  ⚠️  OVERFITTING DETECTADO (R² caiu {diff_r2:.4f})")
        elif diff_r2 > 0.02:
            print(f"  ⚠️  Leve overfitting (R² caiu {diff_r2:.4f})")
        else:
            print(f"  ✓  Generalização boa (diferença {diff_r2:.4f})")

    return df_resultados


if __name__ == '__main__':
    # Recarregar dados de teste
    print("Carregando dados de teste...")

    # Você vai precisar salvar X_test e y_test
    # Por enquanto, rode isso DEPOIS de rodar train.py na mesma sessão
    print("\n⚠️  Execute este script importando de train.py:")
    print("    from src.modeling.train import pipeline_completo")
    print("    modelos, resultados, (X_test, y_test) = pipeline_completo()")
    print("    from src.modeling.evaluate import avaliar_modelos_teste")
    print("    avaliar_modelos_teste(X_test, y_test)")