import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

from lime.lime_tabular import LimeTabularExplainer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_FINAL, OUTPUTS_MODELS
from src.modeling.train import preparar_dados, dividir_dados

RANDOM_STATE = 42
N_FEATURES_MOSTRAR = 8

# Mesmos imóveis do shap_local.py — selecionados pela mesma lógica,
# para a comparação SHAP vs LIME ser sobre os mesmos casos.
def selecionar_imoveis(modelo, X_test, y_test):
    y_pred = np.expm1(modelo.predict(X_test))
    erro_pct = np.abs((y_test.values - y_pred) / y_test.values) * 100

    aux = pd.DataFrame({
        'real': y_test.values,
        'previsto': y_pred,
        'erro_pct': erro_pct,
    }, index=X_test.index)

    mediana = aux['real'].median()
    tipicos = aux[aux['erro_pct'] < 8].copy()
    tipicos['dist'] = (tipicos['real'] - mediana).abs()
    idx_tipico = tipicos.sort_values('dist').index[0]

    caros = aux[aux['erro_pct'] < 15]
    idx_caro = caros.sort_values('real', ascending=False).index[0]

    idx_pior = aux.sort_values('erro_pct', ascending=False).index[0]

    return {'tipico': idx_tipico, 'caro': idx_caro, 'pior_erro': idx_pior}, aux


def carregar_modelo(nome_arquivo):
    caminho = OUTPUTS_MODELS / nome_arquivo
    with open(caminho, 'rb') as f:
        modelo = pickle.load(f)
    print(f"  Modelo carregado: {caminho.name}")
    return modelo


def explicar_imovel(explainer, modelo, X_test, idx, rotulo, info):
    print("\n" + "=" * 80)
    print(f"IMÓVEL: {rotulo.upper()}  (índice {idx})")
    print("=" * 80)

    x = X_test.loc[idx]

    # O LIME chama predict várias vezes nas perturbações.
    # O modelo prevê em log — o LIME explica nesse espaço.
    def predict_log(dados):
        return modelo.predict(dados)

    explicacao = explainer.explain_instance(
        data_row=x.values,
        predict_fn=predict_log,
        num_features=N_FEATURES_MOSTRAR,
    )

    previsto_reais = np.expm1(modelo.predict(x.to_frame().T)[0])

    print(f"\nValor real: R$ {info['real']:,.2f}")
    print(f"Valor previsto: R$ {previsto_reais:,.2f}")
    print(f"Erro: {info['erro_pct']:.1f}%")

    # explicacao.as_list() devolve (descrição da feature, peso em log).
    print(f"\nTop features segundo o LIME (peso em log -> efeito %):")
    print("  " + "-" * 68)
    for desc, peso_log in explicacao.as_list():
        fator = np.exp(peso_log)
        efeito_pct = (fator - 1) * 100
        sinal = "+" if efeito_pct >= 0 else ""
        print(f"{desc:45s} | {sinal}{efeito_pct:7.1f}%")
    print("  " + "-" * 68)

    return explicacao


def main(nome_modelo='xgboost.pkl'):
    print("=" * 80)
    print("ANÁLISE LIME — EXPLICABILIDADE LOCAL")
    print("=" * 80)
    print(f"\nModelo a explicar: {nome_modelo}")

    modelo = carregar_modelo(nome_modelo)

    print("\n[1/3] Reconstruindo conjunto de teste e treino...")
    df = pd.read_csv(ITBI_FINAL)
    X, y = preparar_dados(df)
    X_train, _, _, _, X_test, y_test, _, _ = dividir_dados(X, y, df)
    print(f"  Treino: {len(X_train):,} | Teste: {len(X_test):,}")

    selecao, aux = selecionar_imoveis(modelo, X_test, y_test)

    print("\n[2/3] Criando explainer LIME...")
    # O LIME aprende a distribuição das features a partir do TREINO —
    # é desse conjunto que ele tira as estatísticas para perturbar.
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        mode='regression',
        random_state=RANDOM_STATE,
        discretize_continuous=True,
    )

    print("\n[3/3] Explicando cada imóvel...")
    for rotulo, idx in selecao.items():
        explicar_imovel(explainer, modelo, X_test, idx, rotulo, aux.loc[idx])

    print("\n" + "=" * 80)
    print("ANÁLISE LIME CONCLUÍDA")
    print("=" * 80)

if __name__ == '__main__':
    main(nome_modelo='xgboost.pkl')