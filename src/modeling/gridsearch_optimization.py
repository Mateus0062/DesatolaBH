"""
GridSearch - Otimização de Hiperparâmetros
Encontra a melhor configuração do Random Forest
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_FINAL, OUTPUTS_MODELS, OUTPUTS_TABLES

print("=" * 80)
print("GRID SEARCH - OTIMIZAÇÃO DE HIPERPARÂMETROS")
print("=" * 80)

# ============================================================================
# 1. CARREGAR E PREPARAR DADOS
# ============================================================================

print("\n[1/5] Carregando e preparando dados...")

df = pd.read_csv(ITBI_FINAL)
print(f"  ✓ Dataset carregado: {len(df):,} linhas")

# Divisão temporal
df_train = df[df['ano_transacao'] <= 2021].copy()
df_val = df[df['ano_transacao'] == 2022].copy()

print(f"  ✓ Treino: {len(df_train):,} (2008-2021)")
print(f"  ✓ Validação: {len(df_val):,} (2022)")

# Preparar features
FEATURES_EXCLUIR = [
    'valor_declarado', 'id', 'endereco', 'data_transacao',
    'bairro', 'cep_padrao', 'tipo_imovel',
    'preco_m2', 'preco_m2_x_idade', 'densidade_x_preco',
    'preco_medio_bairro', 'preco_medio_bairro_ano', 'preco_relativo_bairro',
    'padrao_acabamento', 'tipo_construtivo', 'tipo_ocupacao', 'zona_uso'
]

# Treino
y_train = df_train['valor_declarado'].values
X_train = df_train.drop(columns=[col for col in FEATURES_EXCLUIR if col in df_train.columns])

# Validação
y_val = df_val['valor_declarado'].values
X_val = df_val.drop(columns=[col for col in FEATURES_EXCLUIR if col in df_val.columns])

print(f"  ✓ Features: {len(X_train.columns)}")

# ============================================================================
# 2. DEFINIR GRID DE HIPERPARÂMETROS
# ============================================================================

print("\n[2/5] Definindo grid de hiperparâmetros...")

# Grid de parâmetros a testar
param_grid = {
    'n_estimators': [100, 200, 300],  # Número de árvores
    'max_depth': [20, 25, 30, None],  # Profundidade máxima
    'min_samples_split': [2, 5, 10],  # Min amostras para split
    'min_samples_leaf': [1, 2, 4],  # Min amostras por folha
    'max_features': ['sqrt', 'log2', 0.5],  # Features por split
    'bootstrap': [True],  # Bootstrap samples
}

print("\n  Parâmetros a testar:")
print(f"    n_estimators:      {param_grid['n_estimators']}")
print(f"    max_depth:         {param_grid['max_depth']}")
print(f"    min_samples_split: {param_grid['min_samples_split']}")
print(f"    min_samples_leaf:  {param_grid['min_samples_leaf']}")
print(f"    max_features:      {param_grid['max_features']}")

# Calcular total de combinações
total_combinations = (
        len(param_grid['n_estimators']) *
        len(param_grid['max_depth']) *
        len(param_grid['min_samples_split']) *
        len(param_grid['min_samples_leaf']) *
        len(param_grid['max_features'])
)

print(f"\n  ⚠️  Total de combinações: {total_combinations}")
print(f"  ⚠️  Com CV=3: {total_combinations * 3} modelos a treinar")
print(f"  ⚠️  Tempo estimado: ~2-4 horas")

# ============================================================================
# 3. CONFIGURAR GRIDSEARCH
# ============================================================================

print("\n[3/5] Configurando GridSearchCV...")

# Modelo base
rf_base = RandomForestRegressor(
    random_state=42,
    n_jobs=-1,  # Usar todos os cores
    verbose=0
)

# Scorer customizado (R²)
scorer = make_scorer(r2_score)

# GridSearch com validação cruzada
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=3,  # 3-fold cross-validation
    scoring=scorer,  # Maximizar R²
    n_jobs=-1,  # Paralelizar
    verbose=2,  # Mostrar progresso
    return_train_score=True
)

print("  ✓ GridSearchCV configurado")
print(f"    Cross-validation: 3 folds")
print(f"    Métrica: R² (maior é melhor)")
print(f"    Paralelização: Todos os cores")

# ============================================================================
# 4. EXECUTAR GRIDSEARCH
# ============================================================================

print("\n[4/5] Executando GridSearch...")
print("  (Isso vai demorar bastante... vá tomar um café! ☕)")
print()

inicio = time.time()

# RODAR GRIDSEARCH
grid_search.fit(X_train, y_train)

tempo_total = time.time() - inicio

print()
print(f"  ✓ GridSearch concluído em {tempo_total / 60:.1f} minutos!")

# ============================================================================
# 5. RESULTADOS
# ============================================================================

print("\n[5/5] Analisando resultados...")

# Melhores parâmetros
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("\n" + "=" * 80)
print("MELHORES HIPERPARÂMETROS ENCONTRADOS")
print("=" * 80)
print()
for param, valor in best_params.items():
    print(f"  {param:<20}: {valor}")

print("\n" + "=" * 80)
print("PERFORMANCE DO MELHOR MODELO")
print("=" * 80)

# Modelo com melhores parâmetros
best_model = grid_search.best_estimator_

# Avaliar em validação
y_val_pred = best_model.predict(X_val)

mae_val = mean_absolute_error(y_val, y_val_pred)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
r2_val = r2_score(y_val, y_val_pred)
mape_val = np.mean(np.abs((y_val - y_val_pred) / y_val)) * 100

print(f"\nValidação (2022):")
print(f"  MAE:  R$ {mae_val:,.2f}")
print(f"  RMSE: R$ {rmse_val:,.2f}")
print(f"  MAPE: {mape_val:.2f}%")
print(f"  R²:   {r2_val:.4f}")

# Comparar com modelo original
print("\n" + "=" * 80)
print("COMPARAÇÃO: ORIGINAL vs OTIMIZADO")
print("=" * 80)

# Carregar modelo original
modelo_original_path = OUTPUTS_MODELS / 'random_forest.pkl'
with open(modelo_original_path, 'rb') as f:
    modelo_original = pickle.load(f)

y_val_pred_original = modelo_original.predict(X_val)
r2_original = r2_score(y_val, y_val_pred_original)
mape_original = np.mean(np.abs((y_val - y_val_pred_original) / y_val)) * 100

print(f"\nModelo Original:")
print(f"  R²:   {r2_original:.4f}")
print(f"  MAPE: {mape_original:.2f}%")

print(f"\nModelo Otimizado:")
print(f"  R²:   {r2_val:.4f}")
print(f"  MAPE: {mape_val:.2f}%")

# Melhoria
melhoria_r2 = r2_val - r2_original
melhoria_mape = mape_original - mape_val

print(f"\n{'🎉' if melhoria_r2 > 0 else '⚠️'} Melhoria:")
print(f"  R²:   {melhoria_r2:+.4f} ({melhoria_r2 / r2_original * 100:+.2f}%)")
print(f"  MAPE: {melhoria_mape:+.2f}% ({melhoria_mape / mape_original * 100:+.2f}%)")

# ============================================================================
# 6. SALVAR RESULTADOS
# ============================================================================

print("\n[6/6] Salvando resultados...")

# Salvar melhor modelo
if melhoria_r2 > 0:
    modelo_otimizado_path = OUTPUTS_MODELS / 'random_forest_otimizado.pkl'
    with open(modelo_otimizado_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"  ✓ Melhor modelo salvo: {modelo_otimizado_path.name}")
else:
    print("  ⚠️  Modelo otimizado não melhorou. Mantendo original.")

# Salvar resultados do grid search
resultados_df = pd.DataFrame(grid_search.cv_results_)
resultados_df = resultados_df.sort_values('rank_test_score')

# Top 10 melhores configurações
top10_path = OUTPUTS_TABLES / 'gridsearch_top10.csv'
resultados_df.head(10).to_csv(top10_path, index=False)
print(f"  ✓ Top 10 configurações salvas: {top10_path.name}")

# Resultados completos
completo_path = OUTPUTS_TABLES / 'gridsearch_completo.csv'
resultados_df.to_csv(completo_path, index=False)
print(f"  ✓ Resultados completos salvos: {completo_path.name}")

# ============================================================================
# 7. ANÁLISE DETALHADA
# ============================================================================

print("\n" + "=" * 80)
print("TOP 5 MELHORES CONFIGURAÇÕES")
print("=" * 80)

print(f"\n{'Rank':<6} {'R² (CV)':<12} {'n_est':<8} {'depth':<8} {'min_split':<12} {'min_leaf':<10} {'max_feat':<10}")
print("-" * 90)

for idx, row in resultados_df.head(5).iterrows():
    rank = int(row['rank_test_score'])
    score = row['mean_test_score']
    n_est = row['param_n_estimators']
    depth = row['param_max_depth'] if row['param_max_depth'] != None else 'None'
    split = row['param_min_samples_split']
    leaf = row['param_min_samples_leaf']
    feat = row['param_max_features']

    print(f"{rank:<6} {score:<12.4f} {n_est:<8} {depth!s:<8} {split:<12} {leaf:<10} {feat!s:<10}")

# ============================================================================
# 8. SUMÁRIO FINAL
# ============================================================================

summary = f"""
================================================================================
GRID SEARCH - OTIMIZAÇÃO DE HIPERPARÂMETROS
================================================================================

⏱️  TEMPO DE EXECUÇÃO:
  Total: {tempo_total / 60:.1f} minutos
  Combinações testadas: {total_combinations}
  Modelos treinados: {total_combinations * 3} (CV=3)

🏆 MELHORES HIPERPARÂMETROS:

"""

for param, valor in best_params.items():
    summary += f"  {param:<20}: {valor}\n"

summary += f"""
📊 PERFORMANCE:

  Modelo Original:
    R²:   {r2_original:.4f}
    MAPE: {mape_original:.2f}%

  Modelo Otimizado:
    R²:   {r2_val:.4f}
    MAPE: {mape_val:.2f}%

  Melhoria:
    R²:   {melhoria_r2:+.4f} ({melhoria_r2 / r2_original * 100:+.2f}%)
    MAPE: {melhoria_mape:+.2f}% ({melhoria_mape / mape_original * 100:+.2f}%)

📁 ARQUIVOS GERADOS:

  - random_forest_otimizado.pkl (se melhorou)
  - gridsearch_top10.csv
  - gridsearch_completo.csv

================================================================================
CONCLUSÃO:

"""

if melhoria_r2 > 0.005:  # Melhoria > 0.5%
    summary += "✅ Otimização VALEU A PENA! Modelo melhorou significativamente.\n"
    summary += "   Recomendação: Usar modelo otimizado em produção.\n"
elif melhoria_r2 > 0:
    summary += "⚡ Otimização trouxe leve melhoria.\n"
    summary += "   Recomendação: Avaliar custo/benefício do tempo extra de treino.\n"
else:
    summary += "⚠️  Otimização NÃO melhorou o modelo.\n"
    summary += "   Recomendação: Manter configuração original.\n"
    summary += "   Possível causa: Configuração original já era boa!\n"

summary += f"""
================================================================================
Gerado em: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
"""

print(summary)

# Salvar sumário
summary_path = OUTPUTS_TABLES / 'gridsearch_LEIA-ME.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"✓ Sumário salvo: {summary_path.name}")

print("\n" + "=" * 80)
print("✓✓✓ GRID SEARCH CONCLUÍDO! ✓✓✓")
print("=" * 80)

if melhoria_r2 > 0:
    print("\n🎉 Parabéns! Modelo melhorou!")
    print(f"   Ganho: +{melhoria_r2:.4f} no R² ({melhoria_r2 / r2_original * 100:+.2f}%)")
else:
    print("\n✅ Modelo original já estava bem otimizado!")
    print("   Nenhuma melhoria encontrada, mas validamos a configuração atual.")