import pandas as pd

from src.feature_engineering.ibge_features import processar_dataset_completo
from src.limpeza_dados.Limpeza import limpar_dataset_itbi, salvar_dataset_limpo
from src.feature_engineering.derived_features import aplicar_features_completo
from config import ITBI_RAW, ITBI_CLEANED, ITBI_FINAL

print("="*80)
print("RE-LIMPEZA COM FILTROS RIGOROSOS")
print("="*80)

# Carregar dados brutos
print(f"\nCarregando dados brutos de: {ITBI_RAW}")
df = pd.read_csv(ITBI_RAW, encoding='utf-8')
print(f"  ✓ {len(df):,} linhas carregadas\n")

# Limpar
df_limpo = limpar_dataset_itbi(df, ITBI_RAW)

# Salvar versão limpa
salvar_dataset_limpo(df_limpo, ITBI_CLEANED)

# Aplicar feature engineering
print("\n" + "="*80)
print("APLICANDO FEATURE ENGINEERING")
print("="*80)

df_final = aplicar_features_completo(df_input=df_limpo, salvar=True)

print("\n" + "="*80)
print("✓✓✓ PROCESSO CONCLUÍDO ✓✓✓")
print("="*80)
print(f"\nDataset final salvo em: {ITBI_FINAL}")
print(f"Linhas finais: {len(df_final):,}")
print(f"Colunas finais: {len(df_final.columns)}")