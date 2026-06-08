import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_FINAL, OUTPUTS_FIGURES, OUTPUTS_MODELS
from src.modeling.train import preparar_dados

N_FEATURES_MOSTRAR = 8   # quantas features de maior efeito detalhar por imóvel
ANO_TESTE_OPERACAO = 2023  # holdout do modelo de operação (train, treino<=2022)


def carregar_modelo(nome_arquivo):
    caminho = OUTPUTS_MODELS / nome_arquivo
    with open(caminho, 'rb') as f:
        modelo = pickle.load(f)
    print(f"  Modelo carregado: {caminho.name} (operação, treino<=2022)")
    return modelo


def preparar_teste():
    """Conjunto de teste do modelo de operação = transações de 2024 (holdout
    que o train_v2 deixou de fora). Mesmo recorte do shap_analysis.py."""
    df = pd.read_csv(ITBI_FINAL)
    X, y = preparar_dados(df)
    mask = (df['ano_transacao'] >= ANO_TESTE_OPERACAO).values
    return X[mask], y[mask]


def selecionar_imoveis(modelo, X_test, y_test):
    print("\n[1/4] Selecionando imóveis-exemplo...")

    y_pred = np.expm1(modelo.predict(X_test))
    erro_pct = np.abs((y_test.values - y_pred) / y_test.values) * 100

    aux = pd.DataFrame({
        'idx': X_test.index,
        'real': y_test.values,
        'previsto': y_pred,
        'erro_pct': erro_pct,
    }, index=X_test.index)

    # Típico: valor próximo à mediana E erro baixo.
    mediana = aux['real'].median()
    candidatos_tipicos = aux[aux['erro_pct'] < 8].copy()
    candidatos_tipicos['dist_mediana'] = (candidatos_tipicos['real'] - mediana).abs()
    idx_tipico = candidatos_tipicos.sort_values('dist_mediana').iloc[0]['idx']

    # Caro: maior valor real, com erro razoável (não um caso aberrante).
    candidatos_caros = aux[aux['erro_pct'] < 15].copy()
    idx_caro = candidatos_caros.sort_values('real', ascending=False).iloc[0]['idx']

    # Pior erro: top-1 de erro percentual.
    idx_pior = aux.sort_values('erro_pct', ascending=False).iloc[0]['idx']

    selecao = {
        'tipico': int(idx_tipico),
        'caro': int(idx_caro),
        'pior_erro': int(idx_pior),
    }
    for rotulo, idx in selecao.items():
        linha = aux.loc[idx]
        print(f"  {rotulo:10s}: real R$ {linha['real']:,.0f} | "
              f"previsto R$ {linha['previsto']:,.0f} | "
              f"erro {linha['erro_pct']:.1f}%")
    return selecao, aux


def explicar_imovel(explainer, modelo, X_test, idx, rotulo, info):
    """Gera a explicação local de um imóvel: força em log e leitura
    multiplicativa em reais."""
    print("\n" + "=" * 80)
    print(f"IMÓVEL: {rotulo.upper()}  (índice {idx})")
    print("=" * 80)

    x = X_test.loc[[idx]]  # uma linha, como DataFrame

    # Valores SHAP deste imóvel (espaço log).
    shap_values = explainer.shap_values(x)[0]
    base_log = explainer.expected_value
    # Alguns explainers retornam array; normalizar para escalar.
    if hasattr(base_log, '__len__'):
        base_log = float(np.array(base_log).ravel()[0])

    previsto_log = modelo.predict(x)[0]
    previsto_reais = np.expm1(previsto_log)

    # --- Conferência: a soma SHAP fecha com a previsão? ---
    soma_log = base_log + shap_values.sum()
    erro_reconstrucao = abs(soma_log - previsto_log)

    print(f"\n  Valor real:       R$ {info['real']:,.2f}")
    print(f"  Valor previsto:   R$ {previsto_reais:,.2f}")
    print(f"  Erro:             {info['erro_pct']:.1f}%")
    print(f"\n  Valor-base (média do modelo): R$ {np.expm1(base_log):,.2f}")
    print(f"  [conferência] base_log + Σ SHAP = {soma_log:.4f} | "
          f"log previsto = {previsto_log:.4f} | "
          f"diferença = {erro_reconstrucao:.6f}")

    # --- Leitura multiplicativa das contribuições ---
    contribs = pd.DataFrame({
        'feature': X_test.columns,
        'valor_feature': x.iloc[0].values,
        'shap_log': shap_values,
    })
    # Fator multiplicativo: e^(shap). Efeito percentual: (fator - 1) * 100.
    contribs['fator'] = np.exp(contribs['shap_log'])
    contribs['efeito_pct'] = (contribs['fator'] - 1) * 100
    contribs['abs_shap'] = contribs['shap_log'].abs()
    contribs = contribs.sort_values('abs_shap', ascending=False)

    print(f"\n  Top {N_FEATURES_MOSTRAR} features que mais moveram a previsão:")
    print(f"  (efeito %: quanto a feature multiplicou o preço, vs. a média)")
    print("  " + "-" * 68)
    for _, row in contribs.head(N_FEATURES_MOSTRAR).iterrows():
        sinal = "+" if row['efeito_pct'] >= 0 else ""
        print(f"  {row['feature']:30s} valor={row['valor_feature']:>12,.2f} "
              f"| {sinal}{row['efeito_pct']:6.1f}%")
    print("  " + "-" * 68)
    print(f"  Leitura: partindo da média (R$ {np.expm1(base_log):,.0f}), "
          f"essas features multiplicam até R$ {previsto_reais:,.0f}.")

    # --- Force plot (em log — é o que o SHAP gera nativamente) ---
    plt.figure()
    shap.force_plot(
        base_log, shap_values, x.iloc[0],
        matplotlib=True, show=False
    )
    caminho = OUTPUTS_FIGURES / f'shap_local_{rotulo}.png'
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Force plot salvo: {caminho.name}")


def main(nome_modelo='lightgbm.pkl'):
    print("=" * 80)
    print("ANÁLISE SHAP — EXPLICABILIDADE LOCAL (modelo de operação)")
    print("=" * 80)
    print(f"\nModelo a explicar: {nome_modelo}")

    modelo = carregar_modelo(nome_modelo)

    print("\n[2/4] Reconstruindo conjunto de teste (2023, holdout da operação)...")
    X_test, y_test = preparar_teste()
    if hasattr(modelo, 'feature_names_in_'):
        X_test = X_test[list(modelo.feature_names_in_)]
    print(f"  Conjunto de teste (2023): {len(X_test):,} imóveis")

    selecao, aux = selecionar_imoveis(modelo, X_test, y_test)

    print("\n[3/4] Criando explainer...")
    explainer = shap.TreeExplainer(modelo)

    print("\n[4/4] Explicando cada imóvel...")
    for rotulo, idx in selecao.items():
        explicar_imovel(explainer, modelo, X_test, idx, rotulo, aux.loc[idx])

    print("\n" + "=" * 80)
    print("ANÁLISE SHAP LOCAL CONCLUÍDA")
    print("=" * 80)
    print(f"\nForce plots em: {OUTPUTS_FIGURES}")


if __name__ == '__main__':
    main(nome_modelo='lightgbm.pkl')