"""
Script para corrigir features não-numéricas no dataset IBGE
Remove 'grupo_bairro' (texto) e mantém apenas versão numérica
"""

import pandas as pd
from pathlib import Path

# Caminho do dataset
ITBI_FINAL = Path(__file__).parent / 'data' / 'final' / 'ITBI_final_features.csv'

print("=" * 80)
print("CORRIGINDO FEATURES NÃO-NUMÉRICAS")
print("=" * 80)

# Carregar
print(f"\nCarregando: {ITBI_FINAL}")
df = pd.read_csv(ITBI_FINAL)

print(f"Linhas: {len(df):,}")
print(f"Colunas antes: {len(df.columns)}")

# Verificar tipos
print(f"\nVerificando features não-numéricas...")
non_numeric = []
for col in df.columns:
    if df[col].dtype == 'object':
        non_numeric.append(col)
        print(f"  ⚠️  {col}: {df[col].dtype}")

# Features que DEVEM ser mantidas como texto
features_texto_ok = ['bairro', 'cep_padrao', 'tipo_imovel']

# Features para remover (versões de texto de features que já têm versão numérica)
features_remover = [col for col in non_numeric if col not in features_texto_ok]

if features_remover:
    print(f"\n❌ Removendo features não-numéricas duplicadas:")
    for feat in features_remover:
        print(f"  - {feat}")

    df = df.drop(columns=features_remover)
else:
    print(f"\n✅ Nenhuma feature problemática encontrada!")

print(f"\nColunas depois: {len(df.columns)}")

# Salvar
df.to_csv(ITBI_FINAL, index=False, encoding='utf-8')
print(f"\n✓ Dataset corrigido salvo!")

print("\n" + "=" * 80)
print("AGORA VOCÊ PODE RODAR:")
print("  python test_modelos.py")
print("=" * 80)