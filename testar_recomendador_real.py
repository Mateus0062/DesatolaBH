"""
Testa o recomendador com imóveis REAIS do dataset
"""

import pandas as pd
import numpy as np
from src.sistema_decisao.recomendador import RecomendadorImoveis
from config import ITBI_FINAL

print("="*80)
print("TESTE DO RECOMENDADOR COM IMÓVEIS REAIS")
print("="*80)

# Carregar recomendador
print("\nCarregando sistema de recomendação...")
recomendador = RecomendadorImoveis()

# Carregar dados reais
print("Carregando dataset...")
df = pd.read_csv(ITBI_FINAL)
df_teste = df[df['ano_transacao'] >= 2023].copy()

print(f"  ✓ {len(df_teste):,} imóveis de 2023-2024\n")

# Preparar features (remover colunas que não são features)
colunas_remover = [
    'id', 'endereco', 'bairro', 'data_transacao', 'valor_declarado',
    'valor_base_calculo', 'cep', 'padrao_acabamento', 'tipo_construtivo',
    'tipo_ocupacao', 'zona_uso', 'faixa_idade',
    'preco_m2', 'preco_m2_x_idade', 'densidade_x_preco',
    'preco_medio_bairro_ano', 'preco_medio_bairro', 'preco_relativo_bairro',
]

# ══════════════════════════════════════════════════════════════════════════════
# CENÁRIO 1: Imóvel vendido abaixo do esperado
# ══════════════════════════════════════════════════════════════════════════════

print("="*80)
print("CENÁRIO 1: Imóvel Vendido ABAIXO do Previsto")
print("="*80)
print("Situação: Vendedor pode ter perdido dinheiro\n")

# Pegar um imóvel que vendeu barato
imovel1 = df_teste[
    (df_teste['valor_declarado'] >= 200000) &
    (df_teste['valor_declarado'] <= 400000) &
    (df_teste['area_total_m2'] >= 80) &
    (df_teste['area_total_m2'] <= 150)
].sample(1, random_state=42).iloc[0]

# Preparar features
features1 = imovel1.drop(colunas_remover, errors='ignore').to_frame().T

# Simular: vendedor pediu 10% MENOS que vendeu
preco_real_venda = imovel1['valor_declarado']
preco_pedido_simulado = preco_real_venda * 0.90

print(f"Imóvel real de {imovel1['bairro']}:")
print(f"  Área: {imovel1['area_total_m2']:.0f}m²")
print(f"  Idade: {imovel1['idade_imovel']:.0f} anos")
print(f"  Vendido por: R$ {preco_real_venda:,.2f}")
print(f"  Preço inicialmente pedido (simulado): R$ {preco_pedido_simulado:,.2f}\n")

analise1 = recomendador.analisar_imovel(
    dados_imovel=features1,
    preco_pedido=preco_pedido_simulado,
    tempo_parado_dias=90
)

recomendador.gerar_relatorio(analise1)

print(f"\n💡 VALIDAÇÃO:")
print(f"   Preço real de venda:  R$ {preco_real_venda:,.2f}")
print(f"   Modelo previu:        R$ {analise1['preco_justo_previsto']:,.2f}")
print(f"   Diferença:            R$ {abs(preco_real_venda - analise1['preco_justo_previsto']):,.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# CENÁRIO 2: Imóvel vendido no preço justo
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*80)
print("CENÁRIO 2: Imóvel Vendido NO PONTO")
print("="*80)
print("Situação: Preço estava correto\n")

imovel2 = df_teste[
    (df_teste['valor_declarado'] >= 300000) &
    (df_teste['valor_declarado'] <= 600000)
].sample(1, random_state=123).iloc[0]

features2 = imovel2.drop(colunas_remover, errors='ignore').to_frame().T
preco_real_venda2 = imovel2['valor_declarado']

print(f"Imóvel real de {imovel2['bairro']}:")
print(f"  Área: {imovel2['area_total_m2']:.0f}m²")
print(f"  Idade: {imovel2['idade_imovel']:.0f} anos")
print(f"  Vendido por: R$ {preco_real_venda2:,.2f}\n")

analise2 = recomendador.analisar_imovel(
    dados_imovel=features2,
    preco_pedido=preco_real_venda2,
    tempo_parado_dias=60
)

recomendador.gerar_relatorio(analise2)

print(f"\n💡 VALIDAÇÃO:")
print(f"   Preço real de venda:  R$ {preco_real_venda2:,.2f}")
print(f"   Modelo previu:        R$ {analise2['preco_justo_previsto']:,.2f}")
print(f"   Diferença:            R$ {abs(preco_real_venda2 - analise2['preco_justo_previsto']):,.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# CENÁRIO 3: Imóvel caro (simulando preço inflado)
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*80)
print("CENÁRIO 3: Imóvel com Preço INFLADO (simulação)")
print("="*80)
print("Situação: Vendedor pedindo 40% a mais que vale\n")

imovel3 = df_teste[
    (df_teste['valor_declarado'] >= 400000) &
    (df_teste['valor_declarado'] <= 800000)
].sample(1, random_state=456).iloc[0]

features3 = imovel3.drop(colunas_remover, errors='ignore').to_frame().T
preco_real_venda3 = imovel3['valor_declarado']
preco_inflado = preco_real_venda3 * 1.40

print(f"Imóvel real de {imovel3['bairro']}:")
print(f"  Área: {imovel3['area_total_m2']:.0f}m²")
print(f"  Idade: {imovel3['idade_imovel']:.0f} anos")
print(f"  Valor justo (vendido): R$ {preco_real_venda3:,.2f}")
print(f"  Preço pedido (simulado +40%): R$ {preco_inflado:,.2f}\n")

analise3 = recomendador.analisar_imovel(
    dados_imovel=features3,
    preco_pedido=preco_inflado,
    tempo_parado_dias=180
)

recomendador.gerar_relatorio(analise3)

print(f"\n💡 VALIDAÇÃO:")
print(f"   Preço justo (real):   R$ {preco_real_venda3:,.2f}")
print(f"   Modelo previu:        R$ {analise3['preco_justo_previsto']:,.2f}")
print(f"   Vendedor pediu:       R$ {preco_inflado:,.2f}")
print(f"   Recomendação foi:     {analise3['acao']}")


# ══════════════════════════════════════════════════════════════════════════════
# RESUMO
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "="*80)
print("RESUMO DOS TESTES")
print("="*80)

print(f"\nCenário 1 (preço baixo):")
print(f"  Recomendação: {analise1['acao']}")
print(f"  Erro do modelo: R$ {abs(preco_real_venda - analise1['preco_justo_previsto']):,.2f}")

print(f"\nCenário 2 (preço justo):")
print(f"  Recomendação: {analise2['acao']}")
print(f"  Erro do modelo: R$ {abs(preco_real_venda2 - analise2['preco_justo_previsto']):,.2f}")

print(f"\nCenário 3 (preço inflado):")
print(f"  Recomendação: {analise3['acao']}")
print(f"  Sugeriu reduzir de: R$ {preco_inflado:,.2f}")
print(f"  Para: R$ {analise3['preco_sugerido']:,.2f}")

print("\n" + "="*80)
print("✓✓✓ TESTES CONCLUÍDOS ✓✓✓")
print("="*80)