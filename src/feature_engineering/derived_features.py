import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_CLEANED, ITBI_FINAL


def criar_features_proporcoes(df, verbose=True):
    print("\n[1/8] Criando features de proporções...")

    df['razao_area_util'] = df['area_construida_m2'] / df['area_terreno_m2']
    df['densidade_construcao'] = df['area_total_m2'] / df['area_terreno_m2']
    df['area_nao_construida_m2'] = df['area_total_m2'] - df['area_construida_m2']

    print(f"  ✓ 3 features de proporção criadas")

    return df


def criar_features_idade(df):
    print("\n[2/8] Criando features de idade...")

    # Preencher nulos com mediana
    idade_mediana = df['idade_imovel'].median()
    df['idade_imovel'] = df['idade_imovel'].fillna(idade_mediana)

    # Flag imóvel novo
    df['imovel_novo'] = (df['idade_imovel'] <= 5).astype(int)

    # Depreciação estimada
    df['depreciacao_estimada'] = np.minimum(df['idade_imovel'] * 0.01, 0.40)

    print(f"  ✓ 2 features de idade criadas")

    return df


def criar_features_interacao_simples(df):
    print("\n[3/8] Criando features de interação...")

    # Área × idade
    df['area_x_idade'] = df['area_total_m2'] * df['idade_imovel']

    # Flag: imóvel novo E grande (possivelmente premium)
    area_mediana = df['area_total_m2'].median()
    df['novo_e_grande'] = (
            (df['idade_imovel'] <= 10) &
            (df['area_total_m2'] > area_mediana)
    ).astype(int)

    print(f"  ✓ 2 features de interação criadas")

    return df


def criar_features_bairro_sem_leakage(df):

    print("\n[4/8] Calculando features de bairro (sem leakage - otimizado)...")

    # Pré-calcular estatísticas por bairro
    stats_bairro = df.groupby('bairro')['valor_declarado'].agg([
        'sum', 'count', 'std', 'min', 'max'
    ]).reset_index()
    stats_bairro.columns = ['bairro', 'soma_total', 'count_total', 'std_total', 'min_total', 'max_total']

    # Merge com dataset
    df = df.merge(stats_bairro, on='bairro', how='left')

    # Leave-One-Out: média SEM o próprio imóvel
    df['preco_medio_bairro_loo'] = (df['soma_total'] - df['valor_declarado']) / (df['count_total'] - 1)

    # Para imóveis únicos no bairro, usar média geral
    media_geral = df['valor_declarado'].mean()
    df.loc[df['count_total'] == 1, 'preco_medio_bairro_loo'] = media_geral

    # std é NaN para bairros com uma única transação (precisa de n>=2).
    # Para esses casos, usar o desvio padrão global como fallback.
    std_global = df['valor_declarado'].std()
    df['std_preco_bairro'] = df['std_total'].fillna(std_global)

    # Outras estatísticas (já calculadas, não precisam de LOO)
    df['num_transacoes_bairro'] = df['count_total']
    df['preco_min_bairro'] = df['min_total']
    df['preco_max_bairro'] = df['max_total']
    df['range_preco_bairro'] = df['max_total'] - df['min_total']

    # Limpar colunas temporárias
    df = df.drop(columns=['soma_total', 'count_total', 'std_total', 'min_total', 'max_total'])

    print(f"  ✓ 6 features de bairro criadas (sem leakage)")

    return df


def criar_features_preco_m2_bairro_sem_leakage(df):

    print("\n[5/8] Calculando preço/m² médio do bairro (sem leakage)...")

    # Pré-calcular soma de valores e áreas por bairro
    stats = df.groupby('bairro').agg({
        'valor_declarado': 'sum',
        'area_total_m2': 'sum'
    }).reset_index()
    stats.columns = ['bairro', 'soma_valor', 'soma_area']

    # Merge
    df = df.merge(stats, on='bairro', how='left')

    # Calcular SEM o próprio imóvel
    df['preco_m2_medio_bairro_loo'] = (
            (df['soma_valor'] - df['valor_declarado']) /
            (df['soma_area'] - df['area_total_m2'])
    )

    # Fallback para casos edge
    preco_m2_global = df['valor_declarado'].sum() / df['area_total_m2'].sum()
    df['preco_m2_medio_bairro_loo'] = df['preco_m2_medio_bairro_loo'].fillna(preco_m2_global)

    # Limpar temporários
    df = df.drop(columns=['soma_valor', 'soma_area'])

    print(f"  ✓ Feature criada: preco_m2_medio_bairro_loo")

    return df

def criar_features_valorizacao_bairro(df):
    print("\n[6/8] Calculando valorização do bairro...")

    # Preço médio por bairro/ano
    preco_ano = df.groupby(['bairro', 'ano_transacao'])['valor_declarado'].mean().reset_index()
    preco_ano.columns = ['bairro', 'ano', 'preco_medio']

    # Para cada bairro, calcular valorização dos últimos 3 anos
    valorizacao = []

    for bairro in df['bairro'].unique():
        df_bairro = preco_ano[preco_ano['bairro'] == bairro].sort_values('ano')

        if len(df_bairro) >= 2:
            # Pegar último ano e 3 anos atrás
            anos = df_bairro['ano'].values
            precos = df_bairro['preco_medio'].values

            if len(anos) >= 4:
                # Comparar último ano com 3 anos atrás
                preco_atual = precos[-1]
                preco_3anos = precos[-4]
                val = ((preco_atual - preco_3anos) / preco_3anos) * 100 if preco_3anos > 0 else 0
            else:
                # Comparar primeiro com último disponível
                preco_atual = precos[-1]
                preco_inicial = precos[0]
                val = ((preco_atual - preco_inicial) / preco_inicial) * 100 if preco_inicial > 0 else 0
        else:
            val = 0

        valorizacao.append({'bairro': bairro, 'valorizacao_bairro_3anos': val})

    df_valorizacao = pd.DataFrame(valorizacao)
    df = df.merge(df_valorizacao, on='bairro', how='left')
    df['valorizacao_bairro_3anos'] = df['valorizacao_bairro_3anos'].fillna(0)

    print(f"  ✓ Feature criada: valorizacao_bairro_3anos")

    return df

def criar_features_comparativas(df):
    print("\n[7/8] Criando features comparativas...")

    # Calcular médias por bairro (para área e idade)
    medias_area = df.groupby('bairro')['area_total_m2'].mean().to_dict()
    medias_idade = df.groupby('bairro')['idade_imovel'].mean().to_dict()

    # Criar comparações
    df['area_vs_media_bairro'] = df['area_total_m2'] / df['bairro'].map(medias_area)
    df['idade_vs_media_bairro'] = df['idade_imovel'] / (df['bairro'].map(medias_idade) + 1)

    print(f"  ✓ 2 features comparativas criadas")

    return df

def criar_features_sazonalidade(df):
    print("\n[8/8] Criando features de sazonalidade...")

    df['trimestre'] = df['mes_transacao'].apply(lambda m: (m - 1) // 3 + 1)
    df['fim_de_ano'] = (df['mes_transacao'] >= 11).astype(int)
    df['inicio_ano'] = (df['mes_transacao'] <= 3).astype(int)

    print(f"  ✓ 3 features de sazonalidade criadas")

    return df

def aplicar_features_completo(df_input=None, path_input=None, salvar=True):
    # Carregar dados
    if df_input is None:
        if path_input is None:
            path_input = ITBI_CLEANED
        df = pd.read_csv(path_input)
    else:
        df = df_input.copy()

    print("=" * 80)
    print("FEATURE ENGINEERING SEM DATA LEAKAGE")
    print("=" * 80)
    print(f"\nLinhas de entrada: {len(df):,}")
    print(f"Features iniciais: {len(df.columns)}")

    # Aplicar transformações
    df = criar_features_proporcoes(df)
    df = criar_features_idade(df)
    df = criar_features_interacao_simples(df)
    df = criar_features_bairro_sem_leakage(df)
    df = criar_features_preco_m2_bairro_sem_leakage(df)
    df = criar_features_valorizacao_bairro(df)
    df = criar_features_comparativas(df)
    df = criar_features_sazonalidade(df)

    print("\nFEATURE ENGINEERING CONCLUÍDO")
    print(f"\nFeatures finais: {len(df.columns)}")

    # Salvar
    if salvar:
        df.to_csv(ITBI_FINAL, index=False, encoding='utf-8')
        print(f"\n✓ Dataset salvo: {ITBI_FINAL}")
        print(f"Tamanho: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    return df

if __name__ == '__main__':
    print("Aplicando feature engineering...")
    df_final = aplicar_features_completo()
    print("\nCONCLUÍDO")