import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_CLEANED, ITBI_FINAL

def criar_features_proporcoes(df):
    print("\n[1/5] Criando features de proporções...")

    # Razão entre área construída e área do terreno
    df['razao_area_util'] = df['area_construida_m2'] / df['area_terreno_m2']

    # Densidade de construção (quanto do terreno foi usado)
    df['densidade_construcao'] = df['area_total_m2'] / df['area_terreno_m2']

    # Diferença entre área total e área construída (áreas comuns, garagem, etc.)
    df['area_nao_construida_m2'] = df['area_total_m2'] - df['area_construida_m2']

    print(f"  ✓ 3 features de proporção criadas")
    return df

def criar_features_temporais_bairro(df):
    print("\n[2/5] Criando features temporais por bairro...")

    # Preço médio por bairro e ano
    preco_medio_bairro_ano = df.groupby(['bairro', 'ano_transacao'])['valor_declarado'].mean().reset_index()
    preco_medio_bairro_ano.columns = ['bairro', 'ano_transacao', 'preco_medio_bairro_ano']

    df = df.merge(preco_medio_bairro_ano, on=['bairro', 'ano_transacao'], how='left')

    # Preço médio geral do bairro (todos os anos)
    preco_medio_bairro_geral = df.groupby('bairro')['valor_declarado'].mean().reset_index()
    preco_medio_bairro_geral.columns = ['bairro', 'preco_medio_bairro']

    df = df.merge(preco_medio_bairro_geral, on='bairro', how='left')

    # Preço relativo ao bairro (esse imóvel é caro ou barato para o bairro?)
    df['preco_relativo_bairro'] = df['valor_declarado'] / df['preco_medio_bairro']

    # Calcular valorização dos últimos 3 anos por bairro
    def calcular_valorizacao_3anos(bairro_df):
        if len(bairro_df) < 2:
            return 0

        anos_disponiveis = sorted(bairro_df['ano_transacao'].unique())

        if len(anos_disponiveis) < 2:
            return 0

        # Pegar último ano e 3 anos atrás
        ano_final = anos_disponiveis[-1]
        ano_inicial = ano_final - 3

        preco_final = bairro_df[bairro_df['ano_transacao'] == ano_final]['valor_declarado'].mean()
        preco_inicial = bairro_df[bairro_df['ano_transacao'] >= ano_inicial]['valor_declarado'].iloc[0] if len(
            bairro_df[bairro_df['ano_transacao'] >= ano_inicial]) > 0 else preco_final

        if preco_inicial == 0:
            return 0

        return ((preco_final - preco_inicial) / preco_inicial) * 100

    valorizacao_por_bairro = df.groupby('bairro').apply(calcular_valorizacao_3anos).reset_index()
    valorizacao_por_bairro.columns = ['bairro', 'valorizacao_bairro_3anos']

    df = df.merge(valorizacao_por_bairro, on='bairro', how='left')
    df['valorizacao_bairro_3anos'] = df['valorizacao_bairro_3anos'].fillna(0)

    print("4 features temporais criadas")
    return df

def criar_features_idade_avancadas(df):
    print("\n[3/5] Criando features de idade...")

    # Tratar nulos na idade_imovel (preencher com mediana)
    idade_mediana = df['idade_imovel'].median()
    df['idade_imovel'] = df['idade_imovel'].fillna(idade_mediana)

    # Flag: imóvel novo (até 5 anos)
    df['imovel_novo'] = (df['idade_imovel'] <= 5).astype(int)

    # Estimativa de depreciação (1% ao ano, cap em 40%)
    df['depreciacao_estimada'] = np.minimum(df['idade_imovel'] * 0.01, 0.40)

    print(f"  ✓ 2 features de idade criadas")
    return df

def criar_features_interacao(df):
    print("\n[4/5] Criando features de interação...")

    # Interação área × idade
    df['area_x_idade'] = df['area_total_m2'] * df['idade_imovel']

    # Interação preço/m² × idade
    df['preco_m2_x_idade'] = df['preco_m2'] * df['idade_imovel']

    # Flag: imóvel novo em bairro caro (combo valorizado)
    preco_m2_mediano = df['preco_m2'].median()
    df['novo_em_bairro_caro'] = (
            (df['idade_imovel'] <= 10) &
            (df['preco_m2'] > preco_m2_mediano)
    ).astype(int)

    # Densidade × preço (imóveis densos em áreas caras podem ser apartamentos premium)
    df['densidade_x_preco'] = df['densidade_construcao'] * df['preco_m2']

    print("4 features de interação criadas")
    return df

def criar_features_estatisticas_bairro(df):
    print("\n[5/5] Criando features estatísticas por bairro...")

    # Desvio padrão de preços no bairro
    std_preco = df.groupby('bairro')['valor_declarado'].std().reset_index()
    std_preco.columns = ['bairro', 'std_preco_bairro']
    df = df.merge(std_preco, on='bairro', how='left')
    df['std_preco_bairro'] = df['std_preco_bairro'].fillna(0)

    # Número de transações por bairro
    num_transacoes = df.groupby('bairro').size().reset_index()
    num_transacoes.columns = ['bairro', 'num_transacoes_bairro']
    df = df.merge(num_transacoes, on='bairro', how='left')

    # Preço máximo já visto no bairro
    preco_max = df.groupby('bairro')['valor_declarado'].max().reset_index()
    preco_max.columns = ['bairro', 'preco_max_bairro']
    df = df.merge(preco_max, on='bairro', how='left')

    # Preço mínimo do bairro
    preco_min = df.groupby('bairro')['valor_declarado'].min().reset_index()
    preco_min.columns = ['bairro', 'preco_min_bairro']
    df = df.merge(preco_min, on='bairro', how='left')

    # Range de preços no bairro
    df['range_preco_bairro'] = df['preco_max_bairro'] - df['preco_min_bairro']

    print(f"  ✓ 5 features estatísticas criadas")
    return df

def aplicar_feature_engineering(df_input=None, path_input=None, salvar=True):
    # Carregar dados limpos
    if df_input is None:
        if path_input is None:
            path_input = ITBI_CLEANED
        df = pd.read_csv(path_input)
    else:
        df = df_input.copy()

    print("=" * 80)
    print("INICIANDO FEATURE ENGINEERING")
    print("=" * 80)
    print(f"\nLinhas de entrada: {len(df):,}")
    print(f"Features iniciais: {len(df.columns)}")

    # Aplicar cada módulo
    df = criar_features_proporcoes(df)
    df = criar_features_temporais_bairro(df)
    df = criar_features_idade_avancadas(df)
    df = criar_features_interacao(df)
    df = criar_features_estatisticas_bairro(df)

    # Finalização
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING CONCLUÍDO")
    print("=" * 80)
    print(f"\nLinhas finais:    {len(df):,}")
    print(
        f"Features finais:  {len(df.columns)} (+{len(df.columns) - len(df_input.columns) if df_input is not None else 'N/A'} novas)")
    print(f"\nNovas features criadas:")

    # Listar novas colunas
    if df_input is not None:
        colunas_novas = set(df.columns) - set(df_input.columns)
        for i, col in enumerate(sorted(colunas_novas), 1):
            print(f"  {i:2d}. {col}")

    # Salvar
    if salvar:
        df.to_csv(ITBI_FINAL, index=False, encoding='utf-8')
        print(f"\n✓ Dataset final salvo: {ITBI_FINAL}")
        print(f"  Tamanho: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    return df

if __name__ == '__main__':
    print("Carregando dados limpos...")
    df_final = aplicar_feature_engineering()

    print("\n" + "=" * 80)
    print("PROCESSO CONCLUÍDO!")
    print("=" * 80)
    print("\nDataset pronto para modelagem:")
    print(f"  - Linhas: {len(df_final):,}")
    print(f"  - Features: {len(df_final.columns)}")
    print(f"  - Arquivo: {ITBI_FINAL}")