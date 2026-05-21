"""
Script para verificar data leakage nas features
"""

import pandas as pd
from config import ITBI_FINAL

print("Carregando dados...")
df = pd.read_csv(ITBI_FINAL)
print(f"  ✓ {len(df):,} linhas carregadas\n")

print("=" * 80)
print("VERIFICAÇÃO DE DATA LEAKAGE")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Verificar preco_m2
# ══════════════════════════════════════════════════════════════════════════════

print("\n[1] Verificando 'preco_m2'...")

# Calcular preco_m2 manualmente
preco_m2_calculado = df['valor_declarado'] / df['area_total_m2']

# Correlação com a feature existente
correlacao = df['preco_m2'].corr(preco_m2_calculado)
print(f"  Correlação entre preco_m2 e (valor_declarado/area): {correlacao:.10f}")

if correlacao > 0.9999:
    print("  🚨 DATA LEAKAGE CONFIRMADO!")
    print("     preco_m2 foi calculado usando valor_declarado (o target)")
else:
    print("  ✓ Sem leakage aparente")

# Verificar se são exatamente iguais
diferenca_media = (df['preco_m2'] - preco_m2_calculado).abs().mean()
print(f"  Diferença média absoluta: R$ {diferenca_media:.2f}")

if diferenca_media < 0.01:
    print("  🚨 Os valores são IDÊNTICOS - leakage confirmado!")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Verificar outras features suspeitas
# ══════════════════════════════════════════════════════════════════════════════

print("\n[2] Verificando features derivadas de preço...")

features_suspeitas = [
    'preco_medio_bairro_ano',
    'preco_medio_bairro',
    'preco_relativo_bairro',
    'densidade_x_preco',
    'preco_m2_x_idade',
    'novo_em_bairro_caro'
]

for feature in features_suspeitas:
    if feature in df.columns:
        corr = df[feature].corr(df['valor_declarado'])
        print(f"  {feature:30s}: correlação com target = {corr:.4f}")
        if abs(corr) > 0.95:
            print(f"    🚨 Correlação muito alta! Possível leakage")

# ══════════════════════════════════════════════════════════════════════════════
# 3. Análise detalhada de preco_medio_bairro
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] Verificando 'preco_medio_bairro'...")

# Para cada linha, verificar se preco_medio_bairro INCLUI o próprio imóvel
exemplo_bairro = df[df['bairro'] == 'SAVASSI'].head(100)

if len(exemplo_bairro) > 0:
    # Calcular média SEM o próprio imóvel (correto)
    for idx, row in exemplo_bairro.head(5).iterrows():
        # Média do bairro INCLUINDO este imóvel
        media_com = df[df['bairro'] == row['bairro']]['valor_declarado'].mean()

        # Média do bairro SEM este imóvel
        media_sem = df[(df['bairro'] == row['bairro']) & (df.index != idx)]['valor_declarado'].mean()

        diferenca = abs(media_com - media_sem)

        if idx == exemplo_bairro.index[0]:  # só printar primeira vez
            print(f"\n  Exemplo - Bairro: {row['bairro']}")
            print(f"    Valor do imóvel:              R$ {row['valor_declarado']:>12,.2f}")
            print(f"    preco_medio_bairro (dataset): R$ {row['preco_medio_bairro']:>12,.2f}")
            print(f"    Média COM este imóvel:        R$ {media_com:>12,.2f}")
            print(f"    Média SEM este imóvel:        R$ {media_sem:>12,.2f}")
            print(f"    Diferença:                    R$ {diferenca:>12,.2f}")

            # Se preco_medio_bairro é igual a media_com → LEAKAGE
            if abs(row['preco_medio_bairro'] - media_com) < 1:
                print(f"    🚨 LEAKAGE! A média INCLUI o próprio imóvel")
            elif abs(row['preco_medio_bairro'] - media_sem) < 1:
                print(f"    ✓ OK! A média EXCLUI o próprio imóvel")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Resumo e Recomendações
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("RESUMO E RECOMENDAÇÕES")
print("=" * 80)

print("\nFeatures COM data leakage (devem ser removidas ou recalculadas):")
leakage_confirmado = []

if correlacao > 0.9999:
    leakage_confirmado.append('preco_m2')
    print("  ❌ preco_m2 (calculado diretamente do target)")

if 'preco_m2_x_idade' in df.columns:
    leakage_confirmado.append('preco_m2_x_idade')
    print("  ❌ preco_m2_x_idade (deriva de preco_m2)")

if 'densidade_x_preco' in df.columns:
    leakage_confirmado.append('densidade_x_preco')
    print("  ❌ densidade_x_preco (deriva de preco_m2)")

print("\nFeatures SUSPEITAS (verificar se incluem o próprio imóvel):")
print("  ⚠️  preco_medio_bairro_ano")
print("  ⚠️  preco_medio_bairro")
print("  ⚠️  preco_relativo_bairro")

print("\nPróximos passos:")
print("  1. Remover features com leakage confirmado")
print("  2. Recalcular features de bairro EXCLUINDO o próprio imóvel")
print("  3. Retreinar modelo")
print("  4. Validar novamente")

if len(leakage_confirmado) > 0:
    print(f"\n⚠️  TOTAL DE FEATURES COM LEAKAGE: {len(leakage_confirmado)}")
else:
    print(f"\n✓ Nenhum leakage óbvio detectado")

print("\n" + "=" * 80)