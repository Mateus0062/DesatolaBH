"""
Diagnóstico: Identificar features não-numéricas
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import ITBI_FINAL

print("=" * 80)
print("DIAGNÓSTICO DE FEATURES")
print("=" * 80)

# Carregar dataset
print(f"\nCarregando: {ITBI_FINAL}")
df = pd.read_csv(ITBI_FINAL)

print(f"\nLinhas: {len(df):,}")
print(f"Colunas: {len(df.columns)}")

# Lista de features para REMOVER (sempre removidas)
FEATURES_REMOVER = [
    'data_transacao', 'ano_transacao', 'mes_transacao',
    'bairro', 'cep_padrao', 'tipo_imovel',
    'preco_m2', 'preco_m2_x_idade', 'densidade_x_preco',
    'preco_medio_bairro', 'preco_medio_bairro_ano', 'preco_relativo_bairro'
]

# Simular o que train.py faz
print("\n" + "=" * 80)
print("SIMULANDO PREPARAÇÃO DE DADOS (como train.py)")
print("=" * 80)

# Remover target
X = df.drop(columns=['valor_declarado'])
print(f"\n1. Após remover target: {len(X.columns)} colunas")

# Remover features indesejadas
X = X.drop(columns=[col for col in FEATURES_REMOVER if col in X.columns])
print(f"2. Após remover features indesejadas: {len(X.columns)} colunas")

# Verificar tipos
print("\n" + "=" * 80)
print("ANÁLISE DE TIPOS DE DADOS")
print("=" * 80)

print(f"\nFeatures por tipo:")
print(X.dtypes.value_counts())

print(f"\n" + "=" * 80)
print("FEATURES NÃO-NUMÉRICAS (PROBLEMA!):")
print("=" * 80)

problematicas = []
for col in X.columns:
    if X[col].dtype == 'object':
        valores_unicos = X[col].unique()[:5]
        print(f"\n❌ {col}")
        print(f"   Tipo: {X[col].dtype}")
        print(f"   Valores únicos (amostra): {valores_unicos}")
        problematicas.append(col)

if problematicas:
    print(f"\n" + "=" * 80)
    print(f"SOLUÇÃO: Remover estas {len(problematicas)} colunas:")
    print("=" * 80)
    for col in problematicas:
        print(f"  - {col}")

    print(f"\nADICIONE ao train.py na lista FEATURES_REMOVER:")
    print(f"  {problematicas}")

else:
    print(f"\n✅ Todas as features são numéricas!")

print(f"\n" + "=" * 80)
print("FEATURES NUMÉRICAS FINAIS:")
print("=" * 80)

numericas = [col for col in X.columns if X[col].dtype != 'object']
print(f"\nTotal: {len(numericas)} features")
for i, col in enumerate(numericas, 1):
    print(f"  {i:2d}. {col}")