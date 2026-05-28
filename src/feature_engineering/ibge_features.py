import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    ITBI_FINAL,
    BAIRROS_CLASSE_1, BAIRROS_CLASSE_2, BAIRROS_CLASSE_3, BAIRROS_CLASSE_4,
)


# ============================================================================
# CLASSIFICAÇÃO SOCIOECONÔMICA DOS BAIRROS (IBGE 2005)
# ============================================================================

def criar_mapeamento_classe():
    mapeamento = {}

    # Classe 1: Popular
    for bairro in BAIRROS_CLASSE_1:
        mapeamento[bairro] = {
            'tipo': 1,
            'grupo': 'Popular',
            'faixa_salarial': 1,
            'salario_medio_sm': 3.5,
            'salario_min_sm': 0,
            'salario_max_sm': 5
        }

    # Classe 2: Médio
    for bairro in BAIRROS_CLASSE_2:
        mapeamento[bairro] = {
            'tipo': 2,
            'grupo': 'Medio',
            'faixa_salarial': 2,
            'salario_medio_sm': 6.75,
            'salario_min_sm': 5,
            'salario_max_sm': 8.5
        }

    # Classe 3: Alto
    for bairro in BAIRROS_CLASSE_3:
        mapeamento[bairro] = {
            'tipo': 3,
            'grupo': 'Alto',
            'faixa_salarial': 3,
            'salario_medio_sm': 11.5,
            'salario_min_sm': 8.5,
            'salario_max_sm': 14.5
        }

    # Classe 4: Luxo
    for bairro in BAIRROS_CLASSE_4:
        mapeamento[bairro] = {
            'tipo': 4,
            'grupo': 'Luxo',
            'faixa_salarial': 4,
            'salario_medio_sm': 20.0,
            'salario_min_sm': 14.5,
            'salario_max_sm': 50
        }

    return mapeamento


def carregar_densidade_demografica(path_csv):
    df_densidade = pd.read_csv(path_csv, encoding='utf-8-sig')

    # Normalizar nome dos bairros (uppercase, sem acentos especiais)
    df_densidade['BAIRRO_NORM'] = df_densidade['BAIRRO'].str.upper()
    df_densidade['BAIRRO_NORM'] = df_densidade['BAIRRO_NORM'].str.replace('Ã£', 'A', regex=False)
    df_densidade['BAIRRO_NORM'] = df_densidade['BAIRRO_NORM'].str.replace('Ã©', 'E', regex=False)
    df_densidade['BAIRRO_NORM'] = df_densidade['BAIRRO_NORM'].str.replace('Ã´', 'O', regex=False)
    df_densidade['BAIRRO_NORM'] = df_densidade['BAIRRO_NORM'].str.replace('Ã§', 'C', regex=False)

    densidade_dict = {}

    for _, row in df_densidade.iterrows():
        bairro = row['BAIRRO_NORM']
        densidade_dict[bairro] = {
            'populacao': row['POPULACAO'],
            'domicilios': row['DOMICILIOS'],
            'area_km2': row['AREA_KM'],
            'densidade_demografica': row['DENSIDADE_DEMOGRAFICA']
        }

    return densidade_dict


def aplicar_features_ibge(df, path_densidade_csv, verbose=True):
    print("APLICANDO FEATURES IBGE")

    print(f"\nLinhas de entrada: {len(df):,}")

    # Carregar mapeamentos
    classe_map = criar_mapeamento_classe()
    densidade_map = carregar_densidade_demografica(path_densidade_csv)

    print(f"\n[1/4] Carregados:")
    print(f"  - {len(classe_map)} bairros com classificação socioeconômica")
    print(f"  - {len(densidade_map)} bairros com densidade demográfica")

    # Features de classificação
    print(f"\n[2/4] Aplicando features de classificação socioeconômica...")

    def get_classe_features(bairro):
        if bairro in classe_map:
            return classe_map[bairro]
        else:
            return {
                'tipo': 2,
                'grupo': 'Medio',
                'faixa_salarial': 2,
                'salario_medio_sm': 6.75,
                'salario_min_sm': 5,
                'salario_max_sm': 8.5
            }

    df['tipo_bairro'] = df['bairro'].apply(lambda x: get_classe_features(x)['tipo'])
    df['grupo_bairro'] = df['bairro'].apply(lambda x: get_classe_features(x)['grupo'])
    df['faixa_salarial'] = df['bairro'].apply(lambda x: get_classe_features(x)['faixa_salarial'])
    df['salario_medio_sm'] = df['bairro'].apply(lambda x: get_classe_features(x)['salario_medio_sm'])

    grupo_map = {'Popular': 1, 'Medio': 2, 'Alto': 3, 'Luxo': 4}
    df['grupo_bairro_num'] = df['grupo_bairro'].map(grupo_map)

    print(f"5 features criadas")
    print(f"\n  Distribuição por classe:")
    print(df['grupo_bairro'].value_counts().sort_index())

    # Features de densidade
    print(f"\n[3/4] Aplicando features de densidade demográfica...")

    def get_densidade_features(bairro):
        if bairro in densidade_map:
            return densidade_map[bairro]
        else:
            return {
                'populacao': 5000,
                'domicilios': 2000,
                'area_km2': 1.0,
                'densidade_demografica': 5000
            }

    df['populacao_bairro'] = df['bairro'].apply(lambda x: get_densidade_features(x)['populacao'])
    df['domicilios_bairro'] = df['bairro'].apply(lambda x: get_densidade_features(x)['domicilios'])
    df['area_km2_bairro'] = df['bairro'].apply(lambda x: get_densidade_features(x)['area_km2'])
    df['densidade_demografica'] = df['bairro'].apply(lambda x: get_densidade_features(x)['densidade_demografica'])
    df['pessoas_por_domicilio'] = df['populacao_bairro'] / (df['domicilios_bairro'] + 1)

    print(f"5 features criadas")

    # Features de interação
    print(f"\n[4/4] Criando features de interação...")

    df['tipo_x_densidade'] = df['tipo_bairro'] * (df['densidade_demografica'] / 10000)

    if 'preco_medio_bairro_loo' in df.columns:
        ANOS_BASE = list(range(2008, 2023))
        base = df[df['ano_transacao'].isin(ANOS_BASE)]
        ref_por_tipo = base.groupby('tipo_bairro')['preco_medio_bairro_loo'].mean()

        df['preco_vs_tipo'] = (df['preco_medio_bairro_loo'] /
                               df['tipo_bairro'].map(ref_por_tipo))
        # Fallback: classe ausente da base (raro) -> razão neutra 1.0
        df['preco_vs_tipo'] = df['preco_vs_tipo'].replace(
            [float('inf'), float('-inf')], 1.0).fillna(1.0)

    df['area_vs_densidade'] = df['area_construida_m2'] / (df['densidade_demografica'] + 1)
    df['poder_compra_bairro'] = df['salario_medio_sm'] * df['populacao_bairro']

    print(f"4 features de interação criadas")

    print("\nFEATURES IBGE APLICADAS COM SUCESSO!")

    print(f"\nTotal de features IBGE: 14")

    return df


def processar_dataset_completo(path_itbi_final, path_densidade_csv, salvar=True):
    print("Carregando dataset...")
    df = pd.read_csv(path_itbi_final)

    print(f"Dataset carregado: {len(df):,} linhas, {len(df.columns)} colunas")

    df = aplicar_features_ibge(df, path_densidade_csv, verbose=True)

    if salvar:
        df.to_csv(path_itbi_final, index=False, encoding='utf-8')
        print(f"\nDataset atualizado salvo: {path_itbi_final}")
        print(f"Linhas: {len(df):,}")
        print(f"Colunas: {len(df.columns)}")

    return df


if __name__ == '__main__':
    PATH_ITBI = ITBI_FINAL

    # Opção 1: Se você colocou o CSV em data/external/
    PATH_DENSIDADE = Path(
        __file__).resolve().parent.parent.parent / 'data' / 'external' / 'dataset-densidade-demografica.csv'

    print("\nPROCESSANDO FEATURES IBGE")

    print(f"\nBuscando CSV em: {PATH_DENSIDADE}")

    # Verificar se arquivo existe
    if not Path(PATH_DENSIDADE).exists():
        print(f"\nERRO: Arquivo não encontrado!")
        sys.exit(1)

    df_final = processar_dataset_completo(PATH_ITBI, str(PATH_DENSIDADE), salvar=True)

    print("✓✓✓ CONCLUÍDO ✓✓✓")

    print("\nPróximo passo: retreinar modelo com features IBGE")
    print("  python test_modelos.py")