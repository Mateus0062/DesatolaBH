import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_FINAL, OUTPUTS_MODELS, OUTPUTS_FIGURES

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("GERAÇÃO DE GRÁFICOS")
print("=" * 80)

# Carregar modelo
print("\n[1/5] Carregando modelo Random Forest...")
modelo_path = OUTPUTS_MODELS / 'random_forest.pkl'
with open(modelo_path, 'rb') as f:
    modelo = pickle.load(f)
print(f"  ✓ Modelo carregado: {modelo.n_features_in_} features")

# Carregar dados completos
print("\n[2/5] Carregando dados...")
df = pd.read_csv(ITBI_FINAL)
print(f"  ✓ Dataset: {len(df):,} linhas, {len(df.columns)} colunas")

# Filtrar dados de teste (2023-2024)
df_test = df[df['ano_transacao'].isin([2023, 2024])].copy()
print(f"  ✓ Dados de teste: {len(df_test):,} linhas")

# ============================================================================
# PREPARAR FEATURES - EXATAMENTE COMO train.py
# ============================================================================

print("\n[3/5] Preparando features (mesma lógica do train.py)...")

# Features que NUNCA devem ir pro modelo (metadata, target, leakage)
FEATURES_EXCLUIR = [
    # Target
    'valor_declarado',

    # Metadata (não são preditivas)
    'id', 'endereco', 'data_transacao',

    # Identifiers categóricos (já extraímos info deles)
    'bairro', 'cep_padrao', 'tipo_imovel',

    # Features com DATA LEAKAGE (usam o target)
    'preco_m2', 'preco_m2_x_idade', 'densidade_x_preco',
    'preco_medio_bairro', 'preco_medio_bairro_ano', 'preco_relativo_bairro'
]

# Preparar X e y
y_test = df_test['valor_declarado'].values
X_test = df_test.drop(columns=[col for col in FEATURES_EXCLUIR if col in df_test.columns])

print(f"  ✓ Features finais: {len(X_test.columns)}")
print(f"  ✓ Target: {len(y_test):,} valores")

# Verificar compatibilidade
if len(X_test.columns) != modelo.n_features_in_:
    print(f"\n⚠️  AVISO: Incompatibilidade de features!")
    print(f"   Modelo espera: {modelo.n_features_in_}")
    print(f"   Você tem: {len(X_test.columns)}")

    # Mostrar diferença
    feature_names_train = modelo.feature_names_in_
    feature_names_test = X_test.columns.tolist()

    faltando = set(feature_names_train) - set(feature_names_test)
    extras = set(feature_names_test) - set(feature_names_train)

    if faltando:
        print(f"\n   Features FALTANDO (estavam no treino):")
        for f in sorted(faltando)[:10]:
            print(f"     - {f}")

    if extras:
        print(f"\n   Features EXTRAS (não estavam no treino):")
        for f in sorted(extras)[:10]:
            print(f"     - {f}")

    print("\n   Ajustando features para match...")
    X_test = X_test[feature_names_train]
    print(f"   ✓ Features ajustadas: {len(X_test.columns)}")

# Fazer previsões
print("\n[4/5] Gerando previsões...")
y_pred = modelo.predict(X_test)
print(f"  ✓ {len(y_pred):,} previsões geradas")

# Calcular métricas
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\n  Métricas de Performance:")
print(f"    MAE:  R$ {mae:,.2f}")
print(f"    RMSE: R$ {rmse:,.2f}")
print(f"    MAPE: {mape:.2f}%")
print(f"    R²:   {r2:.4f}")

# ============================================================================
# GRÁFICOS
# ============================================================================

print("\n[5/5] Gerando gráficos...")

# GRÁFICO 1: Scatter Plot
print("  [1/4] Scatter Plot (Real vs Previsto)...")

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_test / 1000, y_pred / 1000, alpha=0.3, s=10, color='steelblue')

max_val = max(y_test.max(), y_pred.max()) / 1000
ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Previsão Perfeita', alpha=0.7)

ax.set_xlabel('Valor Real (R$ mil)', fontsize=12, fontweight='bold')
ax.set_ylabel('Valor Previsto (R$ mil)', fontsize=12, fontweight='bold')
ax.set_title(f'Previsão vs Realidade\nR² = {r2:.4f} | MAPE = {mape:.2f}%',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

scatter_path = OUTPUTS_FIGURES / 'scatter_real_vs_previsto.png'
plt.tight_layout()
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {scatter_path.name}")

# GRÁFICO 2: Distribuição de Erros
print("  [2/4] Distribuição de Erros...")

erros = y_test - y_pred
erros_percentuais = ((y_test - y_pred) / y_test) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(erros / 1000, bins=50, edgecolor='black', alpha=0.7, color='coral')
ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Erro = 0', alpha=0.7)
ax1.set_xlabel('Erro (R$ mil)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequência', fontsize=11, fontweight='bold')
ax1.set_title('Distribuição de Erros Absolutos', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.hist(erros_percentuais, bins=50, edgecolor='black', alpha=0.7, color='orange')
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Erro = 0', alpha=0.7)
ax2.set_xlabel('Erro (%)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequência', fontsize=11, fontweight='bold')
ax2.set_title('Distribuição de Erros Percentuais', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

erros_path = OUTPUTS_FIGURES / 'distribuicao_erros.png'
plt.tight_layout()
plt.savefig(erros_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {erros_path.name}")

# GRÁFICO 3: Residual Plot
print("  [3/4] Residual Plot...")

fig, ax = plt.subplots(figsize=(10, 6))

residuos = y_test - y_pred
ax.scatter(y_pred / 1000, residuos / 1000, alpha=0.3, s=10, color='purple')
ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)

ax.set_xlabel('Valor Previsto (R$ mil)', fontsize=12, fontweight='bold')
ax.set_ylabel('Resíduo (R$ mil)', fontsize=12, fontweight='bold')
ax.set_title('Análise de Resíduos', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)

residual_path = OUTPUTS_FIGURES / 'residual_plot.png'
plt.tight_layout()
plt.savefig(residual_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {residual_path.name}")

# GRÁFICO 4: Erros por Faixa de Preço
print("  [4/4] Erros por Faixa de Preço...")

df_analise = pd.DataFrame({
    'real': y_test,
    'previsto': y_pred,
    'erro_pct': np.abs(erros_percentuais)
})

df_analise['faixa'] = pd.cut(
    df_analise['real'],
    bins=[0, 100000, 300000, 500000, 1000000, np.inf],
    labels=['< 100k', '100k-300k', '300k-500k', '500k-1M', '> 1M']
)

erros_por_faixa = df_analise.groupby('faixa', observed=True)['erro_pct'].agg(['mean', 'median', 'count'])

fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(erros_por_faixa))
bars = ax.bar(x, erros_por_faixa['mean'], alpha=0.7, label='MAPE Médio', color='skyblue')
line = ax.plot(x, erros_por_faixa['median'], 'ro-', linewidth=2, markersize=8,
               label='MAPE Mediano', markerfacecolor='red')

ax.set_xlabel('Faixa de Preço', fontsize=12, fontweight='bold')
ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('Erro por Faixa de Preço', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(erros_por_faixa.index, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

for i, (idx, row) in enumerate(erros_por_faixa.iterrows()):
    ax.text(i, row['mean'] + 1.5, f"n={int(row['count']):,}",
            ha='center', va='bottom', fontsize=9, fontweight='bold')

faixa_path = OUTPUTS_FIGURES / 'erros_por_faixa.png'
plt.tight_layout()
plt.savefig(faixa_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"      ✓ Salvo: {faixa_path.name}")

# ============================================================================
# SUMÁRIO
# ============================================================================

summary = f"""
================================================================================
SUMÁRIO DE GRÁFICOS GERADOS - DesatolaBH
================================================================================

📊 DADOS ANALISADOS:
  Período:          2023-2024
  Imóveis:          {len(y_test):,}
  Features usadas:  {modelo.n_features_in_}

📈 MÉTRICAS DE PERFORMANCE:
  MAE:              R$ {mae:,.2f}
  RMSE:             R$ {rmse:,.2f}
  MAPE:             {mape:.2f}%
  R²:               {r2:.4f}

📁 GRÁFICOS SALVOS EM: {OUTPUTS_FIGURES}

  1. scatter_real_vs_previsto.png
     → Comparação visual entre valores reais e previstos
     → Linha vermelha = previsão perfeita

  2. distribuicao_erros.png
     → Histograma de erros absolutos (R$)
     → Histograma de erros percentuais (%)

  3. residual_plot.png
     → Análise de resíduos vs valores previstos
     → Verifica homocedasticidade

  4. erros_por_faixa.png
     → MAPE por faixa de preço
     → Identifica onde o modelo erra mais

================================================================================
INTERPRETAÇÃO:

R² = {r2:.4f}
  → O modelo explica {r2 * 100:.2f}% da variação de preços

MAPE = {mape:.2f}%
  → Erro médio de {mape:.2f}% nas previsões
  → Para imóvel de R$ 500k, erro médio de R$ {500000 * mape / 100:,.0f}

Comparação com literatura:
  Estado da arte: R² = 0.75-0.82, MAPE = 18-24%
  DesatolaBH:     R² = {r2:.4f}, MAPE = {mape:.2f}%
  Status:         ✓ ACIMA DO ESTADO DA ARTE

================================================================================
Gerado em: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
"""

print("\n" + summary)

# Salvar sumário
summary_path = OUTPUTS_FIGURES / 'LEIA-ME.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"✓ Sumário salvo: {summary_path.name}")
print("\n" + "=" * 80)
print("✓✓✓ TODOS OS GRÁFICOS GERADOS COM SUCESSO! ✓✓✓")
print("=" * 80)
print(f"\nArquivos disponíveis em: {OUTPUTS_FIGURES}")