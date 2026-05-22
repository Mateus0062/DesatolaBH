"""
Verificar e corrigir grupo_bairro
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config import ITBI_FINAL

print("=" * 80)
print("VERIFICANDO grupo_bairro")
print("=" * 80)

# Carregar
df = pd.read_csv(ITBI_FINAL)

print(f"\nColunas totais: {len(df.columns)}")

# Verificar grupo_bairro
if 'grupo_bairro' in df.columns:
    print(f"\n❌ grupo_bairro EXISTE!")
    print(f"   Tipo: {df['grupo_bairro'].dtype}")
    print(f"   Valores únicos: {df['grupo_bairro'].unique()[:10]}")
    print(f"   Amostra:")
    print(df[['bairro', 'grupo_bairro', 'grupo_bairro_num']].head(10))

    # Remover
    print(f"\n🔧 REMOVENDO grupo_bairro...")
    df = df.drop(columns=['grupo_bairro'])

    # Salvar
    df.to_csv(ITBI_FINAL, index=False, encoding='utf-8')
    print(f"✓ Dataset corrigido salvo!")
    print(f"  Colunas agora: {len(df.columns)}")

else:
    print(f"\n✅ grupo_bairro NÃO existe (já foi removida)")

print("\n" + "=" * 80)
print("Agora rode: python test_modelos.py")
print("=" * 80)