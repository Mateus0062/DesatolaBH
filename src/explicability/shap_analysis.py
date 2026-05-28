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
from config import ITBI_FINAL, OUTPUTS_MODELS, OUTPUTS_FIGURES
from src.modeling.train import preparar_dados, dividir_dados

TAMANHO_AMOSTRA = 5000
RANDOM_STATE = 42
N_DEPENDENCE_PLOTS = 5  # quantas features de topo ganham dependence plot

def carregar_modelo(nome_arquivo):
    caminho = OUTPUTS_MODELS / nome_arquivo
    with open(caminho, 'rb') as f:
        modelo = pickle.load(f)
    print(f"  Modelo carregado: {caminho.name}")
    return modelo


def preparar_amostra_teste():
    print("\n[1/4] Reconstruindo conjunto de teste...")

    df = pd.read_csv(ITBI_FINAL)
    X, y = preparar_dados(df)
    X_train, y_train, X_val, y_val, X_test, y_test, _, _ = dividir_dados(X, y, df)

    print(f"Conjunto de teste: {len(X_test):,} imóveis")

    # Amostra para o SHAP. Se o teste for menor que a amostra pedida,
    # usa o teste inteiro.
    n = min(TAMANHO_AMOSTRA, len(X_test))
    X_amostra = X_test.sample(n=n, random_state=RANDOM_STATE)

    print(f"Amostra para SHAP: {len(X_amostra):,} imóveis "
          f"(random_state={RANDOM_STATE})")
    return X_amostra


def calcular_shap(modelo, X_amostra):
    print("\n[2/4] Calculando valores SHAP (TreeExplainer)...")

    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer.shap_values(X_amostra)

    print(f"Valores SHAP calculados: matriz {shap_values.shape}")
    return explainer, shap_values


def ranking_importancia(shap_values, X_amostra):
    print("\n[3/4] Ranking de importância de features...")

    importancia = np.abs(shap_values).mean(axis=0)
    ranking = pd.DataFrame({
        'feature': X_amostra.columns,
        'importancia_shap': importancia
    }).sort_values('importancia_shap', ascending=False).reset_index(drop=True)

    # Participação relativa — ajuda a ver concentração.
    ranking['pct'] = ranking['importancia_shap'] / ranking['importancia_shap'].sum() * 100

    print("\nRanking completo (importância média |SHAP|):")
    print("  " + "-" * 60)
    for i, row in ranking.iterrows():
        print(f"{i+1:2d}. {row['feature']:32s} "
              f"{row['importancia_shap']:.4f}  ({row['pct']:5.1f}%)")
    print("  " + "-" * 60)

    # Sinal de auditoria.
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

    # --- Summary plot (beeswarm): visão global ---
    plt.figure()
    shap.summary_plot(shap_values, X_amostra, show=False)
    caminho_summary = OUTPUTS_FIGURES / 'shap_summary.png'
    plt.tight_layout()
    plt.savefig(caminho_summary, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Summary plot salvo: {caminho_summary.name}")

    # --- Summary plot em barras: importância média ---
    plt.figure()
    shap.summary_plot(shap_values, X_amostra, plot_type='bar', show=False)
    caminho_bar = OUTPUTS_FIGURES / 'shap_importancia_barras.png'
    plt.tight_layout()
    plt.savefig(caminho_bar, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Gráfico de barras salvo: {caminho_bar.name}")

    # --- Dependence plots das N features mais importantes ---
    top_features = ranking['feature'].head(N_DEPENDENCE_PLOTS).tolist()
    for feature in top_features:
        plt.figure()
        shap.dependence_plot(feature, shap_values, X_amostra, show=False)
        nome_arq = f'shap_dependence_{feature}.png'
        plt.tight_layout()
        plt.savefig(OUTPUTS_FIGURES / nome_arq, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Dependence plot salvo: {nome_arq}")


def main(nome_modelo='xgboost.pkl'):
    print("=" * 80)
    print("ANÁLISE SHAP — EXPLICABILIDADE GLOBAL")
    print("=" * 80)
    print(f"\nModelo a explicar: {nome_modelo}")

    modelo = carregar_modelo(nome_modelo)
    X_amostra = preparar_amostra_teste()
    explainer, shap_values = calcular_shap(modelo, X_amostra)
    ranking = ranking_importancia(shap_values, X_amostra)
    gerar_plots(explainer, shap_values, X_amostra, ranking)

    # Salva o ranking como CSV para uso posterior (tabela do artigo).
    caminho_csv = OUTPUTS_FIGURES.parent / 'tables' / 'shap_ranking_importancia.csv'
    ranking.to_csv(caminho_csv, index=False, encoding='utf-8')

    print("\n" + "=" * 80)
    print("ANÁLISE SHAP CONCLUÍDA")
    print("=" * 80)
    print(f"\nFiguras em: {OUTPUTS_FIGURES}")
    print(f"Ranking (CSV) em: {caminho_csv}")


if __name__ == '__main__':
    main(nome_modelo='xgboost.pkl')