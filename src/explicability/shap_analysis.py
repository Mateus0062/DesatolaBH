import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_FINAL, OUTPUTS_MODELS, OUTPUTS_FIGURES

print("=" * 80)
print("ANÁLISE SHAP - EXPLICABILIDADE DO MODELO")
print("=" * 80)

# ============================================================================
# 1. CARREGAR MODELO E DADOS
# ============================================================================

print("\n[1/6] Carregando modelo e dados...")

# Carregar modelo
modelo_path = OUTPUTS_MODELS / 'random_forest.pkl'
with open(modelo_path, 'rb') as f:
    modelo = pickle.load(f)
print(f"  ✓ Modelo carregado: {modelo.n_features_in_} features")

# Carregar dados
df = pd.read_csv(ITBI_FINAL)
df_test = df[df['ano_transacao'].isin([2023, 2024])].copy()
print(f"  ✓ Dados de teste: {len(df_test):,} imóveis")

# Preparar features
FEATURES_EXCLUIR = [
    'valor_declarado', 'id', 'endereco', 'data_transacao',
    'bairro', 'cep_padrao', 'tipo_imovel',
    'preco_m2', 'preco_m2_x_idade', 'densidade_x_preco',
    'preco_medio_bairro', 'preco_medio_bairro_ano', 'preco_relativo_bairro'
]

y_test = df_test['valor_declarado'].values
X_test = df_test.drop(columns=[col for col in FEATURES_EXCLUIR if col in df_test.columns])

# Ajustar features se necessário
if len(X_test.columns) != modelo.n_features_in_:
    X_test = X_test[modelo.feature_names_in_]

print(f"  ✓ Features preparadas: {len(X_test.columns)}")

# Usar amostra (SHAP é lento!)
SAMPLE_SIZE = 1000
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), size=min(SAMPLE_SIZE, len(X_test)), replace=False)
X_sample = X_test.iloc[sample_idx]
y_sample = y_test[sample_idx]

print(f"  ✓ Amostra para SHAP: {len(X_sample):,} imóveis")

# ============================================================================
# 2. CRIAR EXPLAINER SHAP
# ============================================================================

print("\n[2/6] Criando SHAP Explainer...")
print("  (Isso pode demorar alguns minutos...)")

explainer = shap.TreeExplainer(modelo)
shap_values = explainer(X_sample)

print(f"  ✓ SHAP values calculados!")
print(f"  ✓ Shape: {shap_values.values.shape}")

# ============================================================================
# 3. GRÁFICO 1: Summary Plot (Feature Importance)
# ============================================================================

print("\n[3/6] Gerando gráficos...")
print("  [1/5] Summary Plot (Feature Importance Global)...")

fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
plt.title('Importância das Features (SHAP)\nImpacto de cada feature no preço',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

summary_path = OUTPUTS_FIGURES / 'shap_summary_plot.png'
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {summary_path.name}")

# ============================================================================
# 4. GRÁFICO 2: Bar Plot (Feature Importance Média)
# ============================================================================

print("  [2/5] Bar Plot (Importância Média)...")

fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.bar(shap_values, max_display=20, show=False)
plt.title('Top 20 Features Mais Importantes\nImpacto médio absoluto no preço',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

bar_path = OUTPUTS_FIGURES / 'shap_bar_plot.png'
plt.savefig(bar_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {bar_path.name}")

# ============================================================================
# 5. GRÁFICO 3: Beeswarm Plot
# ============================================================================

print("  [3/5] Beeswarm Plot (Distribuição de Impacto)...")

fig, ax = plt.subplots(figsize=(12, 8))
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.title('Distribuição de Impacto das Features\nVermelho = valor alto, Azul = valor baixo',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

beeswarm_path = OUTPUTS_FIGURES / 'shap_beeswarm_plot.png'
plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {beeswarm_path.name}")

# ============================================================================
# 6. GRÁFICO 4: Dependence Plot (Top 3 Features)
# ============================================================================

print("  [4/5] Dependence Plots (Top 3 Features)...")

# Calcular importância média
feature_importance = np.abs(shap_values.values).mean(axis=0)
top_features_idx = np.argsort(feature_importance)[::-1][:3]
top_features = [X_sample.columns[i] for i in top_features_idx]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (feat_idx, feat_name) in enumerate(zip(top_features_idx, top_features)):
    shap.plots.scatter(shap_values[:, feat_idx], color=shap_values, show=False, ax=axes[idx])
    axes[idx].set_title(f'{feat_name}', fontsize=12, fontweight='bold')

plt.suptitle('Dependência das Top 3 Features Mais Importantes',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

dependence_path = OUTPUTS_FIGURES / 'shap_dependence_plots.png'
plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {dependence_path.name}")

# ============================================================================
# 7. EXPLICAÇÃO DE IMÓVEIS ESPECÍFICOS
# ============================================================================

print("  [5/5] Waterfall Plots (Imóveis Específicos)...")

# Selecionar 3 imóveis interessantes
idx_barato = y_sample.argmin()  # Mais barato
idx_caro = y_sample.argmax()  # Mais caro
idx_medio = np.argsort(y_sample)[len(y_sample) // 2]  # Mediano

casos = [
    (idx_barato, 'Imóvel Mais Barato'),
    (idx_medio, 'Imóvel Mediano'),
    (idx_caro, 'Imóvel Mais Caro')
]

fig, axes = plt.subplots(3, 1, figsize=(12, 15))

for ax, (idx, titulo) in zip(axes, casos):
    shap.plots.waterfall(shap_values[idx], max_display=15, show=False)

    # Pegar handle do plot atual
    plt.sca(ax)
    plt.title(f'{titulo}\nValor Real: R$ {y_sample[idx]:,.0f} | '
              f'Previsto: R$ {modelo.predict(X_sample.iloc[[idx]])[0]:,.0f}',
              fontsize=12, fontweight='bold', pad=10)

plt.tight_layout()
waterfall_path = OUTPUTS_FIGURES / 'shap_waterfall_examples.png'
plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {waterfall_path.name}")

# ============================================================================
# 8. RELATÓRIO DE FEATURE IMPORTANCE
# ============================================================================

print("\n[4/6] Gerando relatório de importância...")

# Calcular importância média de cada feature
feature_importance_df = pd.DataFrame({
    'feature': X_sample.columns,
    'importance': np.abs(shap_values.values).mean(axis=0),
    'importance_std': np.abs(shap_values.values).std(axis=0)
}).sort_values('importance', ascending=False)

print("\n" + "=" * 80)
print("TOP 15 FEATURES MAIS IMPORTANTES (SHAP)")
print("=" * 80)
print(f"\n{'Rank':<6} {'Feature':<35} {'Importância Média':<20} {'Desvio Padrão'}")
print("-" * 80)

for idx, row in feature_importance_df.head(15).iterrows():
    print(f"{idx + 1:<6} {row['feature']:<35} R$ {row['importance']:>12,.2f}    "
          f"± R$ {row['importance_std']:>10,.2f}")

# Salvar em CSV
importance_path = OUTPUTS_FIGURES / 'shap_feature_importance.csv'
feature_importance_df.to_csv(importance_path, index=False)
print(f"\n✓ Tabela completa salva: {importance_path.name}")

# ============================================================================
# 9. ANÁLISE DE CASOS ESPECÍFICOS
# ============================================================================

print("\n[5/6] Analisando casos específicos...")


def explicar_imovel(idx):
    """Explica a previsão de um imóvel específico"""

    # Dados do imóvel
    imovel = X_sample.iloc[idx]
    valor_real = y_sample[idx]
    valor_previsto = modelo.predict(X_sample.iloc[[idx]])[0]

    # SHAP values
    shap_vals = shap_values[idx].values
    base_value = shap_values[idx].base_values

    # Top contribuições
    contribuicoes = pd.DataFrame({
        'feature': X_sample.columns,
        'valor': imovel.values,
        'shap': shap_vals
    }).sort_values('shap', key=abs, ascending=False).head(10)

    print(f"\n{'=' * 80}")
    print(f"EXPLICAÇÃO DO IMÓVEL #{idx}")
    print(f"{'=' * 80}")
    print(f"\nValor Real:      R$ {valor_real:>12,.2f}")
    print(f"Valor Previsto:  R$ {valor_previsto:>12,.2f}")
    print(f"Erro:            R$ {abs(valor_real - valor_previsto):>12,.2f} "
          f"({abs(valor_real - valor_previsto) / valor_real * 100:.1f}%)")
    print(f"\nPreço Base (média): R$ {base_value:>12,.2f}")

    print(f"\nTop 10 Contribuições:")
    print(f"{'Feature':<30} {'Valor':<15} {'Impacto':<20}")
    print("-" * 80)

    for _, row in contribuicoes.iterrows():
        sinal = '+' if row['shap'] > 0 else ''
        print(f"{row['feature']:<30} {row['valor']:<15.2f} "
              f"{sinal}R$ {row['shap']:>12,.2f}")

    total_shap = shap_vals.sum()
    print("-" * 80)
    print(f"{'TOTAL DE AJUSTES':<30} {'':<15} "
          f"{'+' if total_shap > 0 else ''}R$ {total_shap:>12,.2f}")
    print(f"{'PREÇO FINAL PREVISTO':<30} {'':<15} R$ {base_value + total_shap:>12,.2f}")


# Explicar os 3 casos
explicar_imovel(idx_barato)
explicar_imovel(idx_medio)
explicar_imovel(idx_caro)

# ============================================================================
# 10. SUMÁRIO FINAL
# ============================================================================

print("\n[6/6] Gerando sumário final...")

summary = f"""
================================================================================
ANÁLISE SHAP - EXPLICABILIDADE DO MODELO
================================================================================

📊 DADOS ANALISADOS:
  Imóveis analisados:    {len(X_sample):,}
  Features explicadas:   {len(X_sample.columns)}
  Período:               2023-2024

📈 TOP 5 FEATURES MAIS IMPORTANTES:

"""

for idx, row in feature_importance_df.head(5).iterrows():
    summary += f"  {idx + 1}. {row['feature']:<35} R$ {row['importance']:>10,.2f}\n"

summary += f"""
📁 GRÁFICOS GERADOS:

  1. shap_summary_plot.png
     → Overview geral de todas as features
     → Mostra importância E direção do impacto

  2. shap_bar_plot.png
     → Ranking de importância das features
     → Impacto médio absoluto

  3. shap_beeswarm_plot.png
     → Distribuição de impacto de cada feature
     → Vermelho = valor alto, Azul = valor baixo

  4. shap_dependence_plots.png
     → Relação entre valor da feature e impacto no preço
     → Top 3 features mais importantes

  5. shap_waterfall_examples.png
     → Explicação detalhada de 3 imóveis específicos
     → Mostra cada contribuição passo a passo

📄 DADOS EXPORTADOS:

  - shap_feature_importance.csv
    → Tabela completa de importância de features

================================================================================
INTERPRETAÇÃO:

As features SHAP mostram:
- Quanto cada característica contribui para o preço (em R$)
- Direção do impacto (positivo ou negativo)
- Quais features são mais importantes globalmente

Isso permite:
✓ Explicar ao vendedor por que o preço é X
✓ Identificar quais melhorias valem a pena (ex: reformar)
✓ Validar se o modelo está fazendo sentido
✓ Aumentar confiança no sistema

================================================================================
USO PRÁTICO:

Para explicar uma previsão ao usuário:
"Seu imóvel vale R$ 850.000 porque:
  +R$ 320k → Bairro Savassi (nobre)
  +R$ 180k → Área grande (120m²)
  -R$ 50k  → Idade moderada (5 anos)
  +R$ 150k → Densidade alta (região valorizada)
  Base: R$ 250k (preço médio BH)"

================================================================================
Gerado em: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
"""

print(summary)

# Salvar sumário
summary_path = OUTPUTS_FIGURES / 'shap_LEIA-ME.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"\n✓ Sumário salvo: {summary_path.name}")

print("\n" + "=" * 80)
print("✓✓✓ ANÁLISE SHAP CONCLUÍDA COM SUCESSO! ✓✓✓")
print("=" * 80)
print(f"\nArquivos disponíveis em: {OUTPUTS_FIGURES}")
print("\nPróximos passos:")
print("  1. Revisar os gráficos gerados")
print("  2. Usar explicações SHAP no sistema de recomendação")
print("  3. Incluir gráficos no artigo/TCC")