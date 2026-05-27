import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.sistema_decisao.recomendador import RecomendadorImoveis
from config import ITBI_FINAL, COLUNAS_EXCLUIR_MODELO, TARGET

print("Carregando sistema de recomendação...\n")
recomendador = RecomendadorImoveis()

df = pd.read_csv(ITBI_FINAL)
# Imóveis recentes (2024), para os exemplos serem representativos do mercado atual.
df_recentes = df[df['ano_transacao'] == 2024].copy()

# Seleciona alguns imóveis variados para demonstração.
# Pega um de baixo valor, um mediano e um de alto valor, para mostrar
# o recomendador em faixas diferentes.
df_recentes = df_recentes.sort_values(TARGET)
n = len(df_recentes)
exemplos = df_recentes.iloc[[int(n * 0.1), int(n * 0.5), int(n * 0.9)]]

print("=" * 80)
print("DEMONSTRAÇÃO DO RECOMENDADOR")
print("=" * 80)
print("\nPara cada imóvel, simulamos 3 cenários de preço pedido:")
print("- 20% acima do valor estimado")
print("- alinhado com o valor estimado")
print("- 20% abaixo do valor estimado")

for posicao, (_, imovel) in enumerate(exemplos.iterrows(), 1):
    # Monta as features do imóvel — o recomendador alinha internamente,
    # mas removemos o que claramente não é feature para clareza.
    features = imovel.drop(
        [c for c in COLUNAS_EXCLUIR_MODELO + [TARGET] if c in imovel.index],
        errors='ignore'
    ).to_frame().T

    features = features.apply(pd.to_numeric, errors='coerce')

    valor_justo = recomendador.prever_preco_justo(features)

    print("\n" + "=" * 80)
    print(f"IMÓVEL DE EXEMPLO {posicao}")
    print(f"Bairro: {imovel['bairro']}")
    print(f"Área construída: {imovel['area_construida_m2']:.0f} m²")
    print(f"Idade: {imovel['idade_imovel']:.0f} anos")
    print(f"Valor de mercado estimado (modelo): R$ {valor_justo:,.2f}")

    # Três cenários de preço pedido hipotético.
    for fator, rotulo in [(1.20, "20% acima"),
                          (1.00, "alinhado"),
                          (0.80, "20% abaixo")]:
        preco_pedido = valor_justo * fator
        analise = recomendador.analisar_imovel(features, preco_pedido)

        print(f"\nCenário: preço pedido {rotulo} (R$ {preco_pedido:,.2f}) ---")
        print(f"Desvio: {analise['desvio_percentual']:+.1f}%")
        print(f"Recomendação: {analise['acao']} (prioridade {analise['prioridade']})")
        print(f"{analise['justificativa']}")

print("\n" + "=" * 80)
print("FIM DA DEMONSTRAÇÃO")
print("=" * 80)