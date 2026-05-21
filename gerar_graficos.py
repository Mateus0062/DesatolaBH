"""
Script para gerar todos os gráficos do artigo
"""

from src.modeling.train import pipeline_completo
from src.modeling.visualizacoes import gerar_todos_graficos
import pickle

print("=" * 80)
print("SCRIPT DE GERAÇÃO DE GRÁFICOS")
print("=" * 80)

# Opção 1: Se já treinou, carregar dados salvos
from config import OUTPUTS_MODELS
from pathlib import Path

test_data_path = OUTPUTS_MODELS / 'test_data.pkl'

if test_data_path.exists():
    print("\n✓ Dados de teste encontrados! Carregando...")
    with open(test_data_path, 'rb') as f:
        X_test, y_test = pickle.load(f)

    # Carregar modelo
    with open(OUTPUTS_MODELS / 'random_forest.pkl', 'rb') as f:
        modelo = pickle.load(f)

    # Fazer predições
    y_pred = modelo.predict(X_test)

else:
    print("\n⚠️  Dados de teste não encontrados.")
    print("Executando pipeline completo (vai treinar novamente)...\n")

    # Treinar
    modelos, resultados, (X_test, y_test) = pipeline_completo()

    # Salvar dados de teste para próxima vez
    with open(test_data_path, 'wb') as f:
        pickle.dump((X_test, y_test), f)
    print(f"\n✓ Dados de teste salvos em: {test_data_path}")

    # Fazer predições
    y_pred = modelos['Random Forest'].predict(X_test)

# Gerar gráficos
gerar_todos_graficos(y_test, y_pred, modelo_nome='Random Forest')

print("\n" + "=" * 80)
print("✓✓✓ PROCESSO CONCLUÍDO ✓✓✓")
print("=" * 80)
print("\nAbra a pasta 'outputs/figures/' para ver os gráficos!")