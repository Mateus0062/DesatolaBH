import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from src.sistema_decisao.recomendador import RecomendadorImoveis
from config import ITBI_FINAL, OUTPUTS_TABLES

# Carregar recomendador
print("Carregando sistema de recomendação...\n")
recomendador = RecomendadorImoveis()

# Carregar dados completos
print("Carregando dataset completo...")
df = pd.read_csv(ITBI_FINAL)
print(f"  ✓ {len(df):,} linhas carregadas\n")

# Filtrar apenas 2022-2024 (dados de teste)
df_teste = df[df['ano_transacao'] >= 2022].copy()
print(f"Imóveis de 2022-2024: {len(df_teste):,} transações")

# REMOVER COLUNAS (incluindo features com leakage!)
colunas_remover = [
    # Colunas não-features
    'id', 'endereco', 'bairro', 'data_transacao', 'valor_declarado',
    'valor_base_calculo', 'cep', 'padrao_acabamento', 'tipo_construtivo',
    'tipo_ocupacao', 'zona_uso', 'faixa_idade',

    'preco_m2',
    'preco_m2_x_idade',
    'densidade_x_preco',
    'preco_medio_bairro_ano',
    'preco_medio_bairro',
    'preco_relativo_bairro',
]

# VALIDAÇÃO: Aplicar recomendador em TODOS os imóveis de teste

print("\n" + "=" * 80)
print("APLICANDO RECOMENDADOR EM TODOS OS IMÓVEIS DE 2022-2024")
print("=" * 80)

resultados = []

# Processar em lotes (para não travar)
AMOSTRA = 10000  # Testar com 10k primeiro, depois aumentar
df_amostra = df_teste.sample(n=min(AMOSTRA, len(df_teste)), random_state=42)

print(f"\nProcessando amostra de {len(df_amostra):,} imóveis...")

for idx, row in df_amostra.iterrows():
    # Preparar features (removendo colunas problemáticas)
    features = row.drop(colunas_remover, errors='ignore').to_frame().T

    # Preço real que foi vendido
    preco_real = row['valor_declarado']

    # Prever preço justo
    preco_previsto = recomendador.prever_preco_justo(features)

    # Calcular desvio
    desvio = recomendador.calcular_desvio(preco_real, preco_previsto)

    # Armazenar resultado
    resultados.append({
        'ano': row['ano_transacao'],
        'bairro': row['bairro'],
        'area_m2': row['area_total_m2'],
        'idade': row['idade_imovel'],
        'preco_real': preco_real,
        'preco_previsto': preco_previsto,
        'desvio_percentual': desvio,
        'erro_absoluto': abs(preco_real - preco_previsto),
        'valorizacao_bairro': row['valorizacao_bairro_3anos']
    })

    # Progress
    if (len(resultados) % 1000 == 0):
        print(f"  Processados: {len(resultados):,}/{len(df_amostra):,}")

df_resultados = pd.DataFrame(resultados)

print(f"\n✓ Processamento concluído!")

# ANÁLISE 1: Distribuição de Desvios

print("\n" + "=" * 80)
print("ANÁLISE 1: DISTRIBUIÇÃO DE DESVIOS (Preço Real vs Previsto)")
print("=" * 80)

print(f"\nEstatísticas:")
print(f"Desvio médio: {df_resultados['desvio_percentual'].mean():>8.2f}%")
print(f"Desvio mediano: {df_resultados['desvio_percentual'].median():>8.2f}%")
print(f"Desvio padrão: {df_resultados['desvio_percentual'].std():>8.2f}%")

print(f"\nDistribuição por faixa:")
faixas = [
    (-100, -15, "Muito abaixo (< -15%)"),
    (-15, -5, "Abaixo (-15% a -5%)"),
    (-5, 5, "No ponto (±5%)"),
    (5, 15, "Acima (5% a 15%)"),
    (15, 100, "Muito acima (> 15%)")
]

for min_val, max_val, label in faixas:
    mask = (df_resultados['desvio_percentual'] > min_val) & \
           (df_resultados['desvio_percentual'] <= max_val)
    qtd = mask.sum()
    pct = qtd / len(df_resultados) * 100
    print(f"  {label:30s}: {qtd:>6,} imóveis ({pct:>5.1f}%)")

# ANÁLISE 2: Precisão do Modelo por Ano

print("\n" + "=" * 80)
print("ANÁLISE 2: PRECISÃO DO MODELO POR ANO")
print("=" * 80)

for ano in sorted(df_resultados['ano'].unique()):
    df_ano = df_resultados[df_resultados['ano'] == ano]
    mae = df_ano['erro_absoluto'].mean()
    mape = df_ano['desvio_percentual'].abs().mean()

    print(f"\n{ano}:")
    print(f"  Imóveis:     {len(df_ano):>6,}")
    print(f"  MAE:         R$ {mae:>12,.2f}")
    print(f"  MAPE:        {mape:>12.2f}%")

# ANÁLISE 3: Simulação de Recomendações

print("\n" + "=" * 80)
print("ANÁLISE 3: SIMULAÇÃO DE RECOMENDAÇÕES")
print("=" * 80)

print("\nSe aplicássemos o recomendador ANTES das vendas:")

def classificar_recomendacao(desvio):
    if desvio > 15:
        return "REDUZIR PREÇO"
    elif 5 < desvio <= 15:
        return "REDUZIR LEVEMENTE"
    elif -5 <= desvio <= 5:
        return "MANTER PREÇO"
    else:
        return "PODE AUMENTAR"


df_resultados['recomendacao'] = df_resultados['desvio_percentual'].apply(
    classificar_recomendacao
)

print(f"\nDistribuição de recomendações:")
for rec, qtd in df_resultados['recomendacao'].value_counts().items():
    pct = qtd / len(df_resultados) * 100
    print(f"  {rec:20s}: {qtd:>6,} imóveis ({pct:>5.1f}%)")

# ANÁLISE 4: Insights por Bairro

print("\n" + "=" * 80)
print("ANÁLISE 4: TOP 10 BAIRROS - Desvio Médio")
print("=" * 80)

bairros_desvio = df_resultados.groupby('bairro').agg({
    'desvio_percentual': 'mean',
    'preco_real': 'count'
}).rename(columns={'preco_real': 'qtd'})

bairros_desvio = bairros_desvio[bairros_desvio['qtd'] >= 20]
bairros_desvio = bairros_desvio.sort_values('desvio_percentual', ascending=False)

print("\nBairros onde imóveis tendem a ser vendidos ACIMA do previsto:")
for bairro, row in bairros_desvio.head(10).iterrows():
    print(f"  {bairro:25s}: +{row['desvio_percentual']:>6.2f}% ({row['qtd']:.0f} imóveis)")

print("\nBairros onde imóveis tendem a ser vendidos ABAIXO do previsto:")
for bairro, row in bairros_desvio.tail(10).iterrows():
    print(f"  {bairro:25s}: {row['desvio_percentual']:>7.2f}% ({row['qtd']:.0f} imóveis)")

# ANÁLISE 5: Validação da Estratégia

print("\n" + "=" * 80)
print("ANÁLISE 5: VALIDAÇÃO DAS REGRAS DE DECISÃO")
print("=" * 80)

print("\nSe seguíssemos as recomendações do sistema:")

no_ponto = df_resultados[df_resultados['desvio_percentual'].abs() <= 5]
print(f"\n✓ Imóveis vendidos no ponto (±5%): {len(no_ponto):,} ({len(no_ponto) / len(df_resultados) * 100:.1f}%)")
print(f"  → Sistema recomendaria: MANTER PREÇO")
print(f"  → Estratégia correta!")

acima = df_resultados[df_resultados['desvio_percentual'] > 5]
print(f"\n⚠ Imóveis vendidos acima do previsto (>5%): {len(acima):,} ({len(acima) / len(df_resultados) * 100:.1f}%)")
print(f"  → Sistema recomendaria: REDUZIR")
print(f"  → Mas venderam assim mesmo (talvez por características únicas)")

abaixo = df_resultados[df_resultados['desvio_percentual'] < -5]
print(
    f"\n💡 Imóveis vendidos abaixo do previsto (<-5%): {len(abaixo):,} ({len(abaixo) / len(df_resultados) * 100:.1f}%)")
print(f"  → Sistema recomendaria: PODE AUMENTAR ou REFORMAR")
print(f"  → Oportunidades perdidas de ganho")

# SALVAR RESULTADOS

output_path = OUTPUTS_TABLES / 'validacao_recomendador.csv'
df_resultados.to_csv(output_path, index=False)
print(f"\n✓ Resultados salvos em: {output_path}")

# CONCLUSÃO

print("\n" + "=" * 80)
print("CONCLUSÃO DA VALIDAÇÃO")
print("=" * 80)

mae_geral = df_resultados['erro_absoluto'].mean()
mape_geral = df_resultados['desvio_percentual'].abs().mean()
dentro_10pct = (df_resultados['desvio_percentual'].abs() <= 10).sum()

print(f"\nPerformance geral do sistema:")
print(f"  MAE: R$ {mae_geral:,.2f}")
print(f"  MAPE: {mape_geral:.2f}%")
print(f"  Dentro de ±10%: {dentro_10pct:,} imóveis ({dentro_10pct / len(df_resultados) * 100:.1f}%)")

print(f"\nInterpretação:")
if mape_geral < 5:
    print(" EXCELENTE - Sistema altamente confiável")
elif mape_geral < 10:
    print("MUITO BOM - Sistema confiável para uso prático")
elif mape_geral < 15:
    print("BOM - Sistema útil, com margem de melhoria")
else:
    print("REGULAR - Necessita ajustes")

print("\n" + "=" * 80)