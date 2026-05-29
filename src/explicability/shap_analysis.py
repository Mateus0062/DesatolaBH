import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

import shap
import matplotlib
matplotlib.use('Agg')  # backend sem janela — só salva arquivos
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_FINAL, OUTPUTS_FIGURES, OUTPUTS_MODELS_TRAIN2
from src.modeling.train import preparar_dados

TAMANHO_AMOSTRA = 5000
RANDOM_STATE = 42
N_DEPENDENCE_PLOTS = 5  # quantas features de topo ganham dependence plot

# Modelo de OPERAÇÃO = train_v2 (treino 2008-2023). Seu holdout real é 2024,
# então é em 2024 que explicamos as previsões — dado que o modelo NÃO viu.
ANO_TESTE_OPERACAO = 2024


def carregar_modelo(nome_arquivo):
    caminho = OUTPUTS_MODELS_TRAIN2 / nome_arquivo
    with open(caminho, 'rb') as f:
        modelo = pickle.load(f)
    print(f"Modelo carregado: {caminho.name} (operação, treino<=2023)")
    return modelo


def preparar_amostra_teste():
    """Amostra do conjunto de teste do modelo de operação = transações de 2024
    (o que o train_v2 deixou de fora). Coerente com o modelo explicado."""
    print("\n[1/4] Reconstruindo conjunto de teste (2024, holdout da operação)...")

    df = pd.read_csv(ITBI_FINAL)
    X, y = preparar_dados(df)

    # Recorta 2024 pelo índice (X e df compartilham índice).
    mask_2024 = (df['ano_transacao'] == ANO_TESTE_OPERACAO).values
    X_test = X[mask_2024]
    print(f"Conjunto de teste (2024): {len(X_test):,} imóveis")

    n = min(TAMANHO_AMOSTRA, len(X_test))
    X_amostra = X_test.sample(n=n, random_state=RANDOM_STATE)
    print(f"Amostra para SHAP: {len(X_amostra):,} imóveis "
          f"(random_state={RANDOM_STATE})")
    return X_amostra


def calcular_shap(modelo, X_amostra):
    print("\n[2/4] Calculando valores SHAP (TreeExplainer)...")
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_amostra)
    print(f"Valores SHAP calculados: matriz {np.asarray(shap_values).shape}")
    return explainer, shap_values


def ranking_importancia(shap_values, X_amostra):
    print("\n[3/4] Ranking de importância de features...")
    importancia = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame({
        'feature': X_amostra.columns,
        'importancia_shap': importancia
    }).sort_values('importancia_shap', ascending=False).reset_index(drop=True)
    ranking['pct'] = ranking['importancia_shap'] / ranking['importancia_shap'].sum() * 100

    print("\nRanking completo (importância média |SHAP|):")
    print("  " + "-" * 60)
    for i, row in ranking.iterrows():
        print(f"{i+1:2d}. {row['feature']:32s} "
              f"{row['importancia_shap']:.4f}  ({row['pct']:5.1f}%)")
    print("  " + "-" * 60)

    top1_pct = ranking.iloc[0]['pct']
    print(f"\n[AUDITORIA] Feature mais importante: "
          f"'{ranking.iloc[0]['feature']}' com {top1_pct:.1f}% da importância total.")
    if top1_pct > 50:
        print(f"[AUDITORIA] ATENÇÃO: uma única feature concentra >50% da "
              f"importância. Verifique se não há leakage por essa feature.")
    else:
        print(f"[AUDITORIA] Concentração sem alarme — importância distribuída.")
    return ranking


def gerar_plots(explainer, shap_values, X_amostra, ranking):
    print("\n[4/4] Gerando figuras...")
    OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X_amostra, show=False)
    caminho_summary = OUTPUTS_FIGURES / 'shap_summary.png'
    plt.tight_layout()
    plt.savefig(caminho_summary, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Summary plot salvo: {caminho_summary.name}")

    plt.figure()
    shap.summary_plot(shap_values, X_amostra, plot_type='bar', show=False)
    caminho_bar = OUTPUTS_FIGURES / 'shap_importancia_barras.png'
    plt.tight_layout()
    plt.savefig(caminho_bar, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Gráfico de barras salvo: {caminho_bar.name}")

    top_features = ranking['feature'].head(N_DEPENDENCE_PLOTS).tolist()
    for feature in top_features:
        plt.figure()
        shap.dependence_plot(feature, shap_values, X_amostra, show=False)
        nome_arq = f'shap_dependence_{feature}.png'
        plt.tight_layout()
        plt.savefig(OUTPUTS_FIGURES / nome_arq, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Dependence plot salvo: {nome_arq}")


def main(nome_modelo='lightgbm.pkl'):
    print("=" * 80)
    print("ANÁLISE SHAP — EXPLICABILIDADE GLOBAL (modelo de operação)")
    print("=" * 80)
    print(f"\nModelo a explicar: {nome_modelo}")

    modelo = carregar_modelo(nome_modelo)
    X_amostra = preparar_amostra_teste()

    # Alinha a ordem das colunas à do modelo (segurança).
    if hasattr(modelo, 'feature_names_in_'):
        X_amostra = X_amostra[list(modelo.feature_names_in_)]

    explainer, shap_values = calcular_shap(modelo, X_amostra)
    ranking = ranking_importancia(shap_values, X_amostra)
    gerar_plots(explainer, shap_values, X_amostra, ranking)

    caminho_csv = OUTPUTS_FIGURES.parent / 'tables' / 'shap_ranking_importancia.csv'
    ranking.to_csv(caminho_csv, index=False, encoding='utf-8')

    print("\n" + "=" * 80)
    print("ANÁLISE SHAP CONCLUÍDA")
    print("=" * 80)
    print(f"\nFiguras em: {OUTPUTS_FIGURES}")
    print(f"Ranking (CSV) em: {caminho_csv}")


if __name__ == '__main__':
    main(nome_modelo='lightgbm.pkl')