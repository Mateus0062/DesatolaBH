import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import ITBI_CLEANED, ITBI_FINAL

# Anos usados como base para TODAS as estatísticas de bairro.
# Mantém coerência com a divisão temporal do train.py (treino+val = <=2022).
# As estatísticas de bairro são "aprendidas" apenas nesses anos e então
# aplicadas a todas as transações, eliminando vazamento temporal: nenhuma
# transação de 2023-2024 entra no cálculo das features de bairro.
ANOS_BASE_BAIRRO = list(range(2008, 2023))  # 2008-2022 inclusive

def criar_features_proporcoes(df, verbose=True):
    print("\n[1/8] Criando features de proporções...")

    df['razao_area_util'] = df['area_construida_m2'] / df['area_terreno_m2']

    print(f"1 feature de proporção criada")

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

    print(f"2 features de idade criadas")

    return df

def criar_features_interacao_simples(df):
    print("\n[3/8] Criando features de interação...")

    # Área × idade
    df['area_x_idade'] = df['area_construida_m2'] * df['idade_imovel']

    # Flag: imóvel novo E grande (possivelmente premium)
    area_mediana = df['area_construida_m2'].median()
    df['novo_e_grande'] = (
            (df['idade_imovel'] <= 10) &
            (df['area_construida_m2'] > area_mediana)
    ).astype(int)

    print(f"2 features de interação criadas")

    return df

def criar_features_bairro_sem_leakage(df, anos_base=ANOS_BASE_BAIRRO):
    print("\n[4/8] Calculando features de bairro (sem leakage temporal)...")

    mask_base = df['ano_transacao'].isin(anos_base)
    df_base = df[mask_base]
    print(f"  Estatísticas de bairro sobre {len(df_base):,} linhas-base "
          f"(anos {min(anos_base)}-{max(anos_base)})")

    # Estatísticas do bairro calculadas SÓ na base (treino+val).
    stats_bairro = df_base.groupby('bairro')['valor_base_calculo_real'].agg(
        ['sum', 'count', 'std', 'min', 'max']
    )
    stats_bairro.columns = ['soma', 'count', 'std', 'min', 'max']

    # Mapear estatísticas do bairro para toda transação (pelo bairro).
    soma_b = df['bairro'].map(stats_bairro['soma'])
    count_b = df['bairro'].map(stats_bairro['count'])
    std_b = df['bairro'].map(stats_bairro['std'])
    min_b = df['bairro'].map(stats_bairro['min'])
    max_b = df['bairro'].map(stats_bairro['max'])

    media_geral = df_base['valor_base_calculo_real'].mean()
    std_global = df_base['valor_base_calculo_real'].std()

    # --- preco_medio_bairro_loo ---
    # Para linhas-base: LOO (remove o próprio imóvel da soma/contagem do bairro).
    # Para linhas fora da base: média do bairro sem remover (não está na base).
    soma_loo = soma_b.where(~mask_base, soma_b - df['valor_base_calculo_real'])
    count_loo = count_b.where(~mask_base, count_b - 1)

    df['preco_medio_bairro_loo'] = soma_loo / count_loo

    # Casos degenerados: bairro com 1 transação na base (count_loo == 0),
    # ou bairro ausente da base (NaN). Usar média geral da base.
    df['preco_medio_bairro_loo'] = df['preco_medio_bairro_loo'].replace(
        [np.inf, -np.inf], np.nan).fillna(media_geral)

    # --- std_preco_bairro --- (não-LOO; std do bairro na base)
    df['std_preco_bairro'] = std_b.fillna(std_global)

    # --- demais estatísticas do bairro (na base) ---
    df['num_transacoes_bairro'] = count_b.fillna(0)
    df['preco_min_bairro'] = min_b.fillna(media_geral)
    df['preco_max_bairro'] = max_b.fillna(media_geral)
    df['range_preco_bairro'] = (df['preco_max_bairro'] - df['preco_min_bairro'])

    print(f"6 features de bairro criadas (sem leakage temporal)")

    return df

def criar_features_preco_m2_bairro_sem_leakage(df, anos_base=ANOS_BASE_BAIRRO):
    print("\n[5/8] Calculando preço/m² médio do bairro (sem leakage temporal)...")

    mask_base = df['ano_transacao'].isin(anos_base)
    df_base = df[mask_base]

    stats = df_base.groupby('bairro').agg(
        soma_valor=('valor_base_calculo_real', 'sum'),
        soma_area=('area_construida_m2', 'sum'),
    )

    soma_valor_b = df['bairro'].map(stats['soma_valor'])
    soma_area_b = df['bairro'].map(stats['soma_area'])

    # LOO só para linhas-base; teste usa a soma da base inteira.
    soma_valor_loo = soma_valor_b.where(
        ~mask_base, soma_valor_b - df['valor_base_calculo_real'])
    soma_area_loo = soma_area_b.where(
        ~mask_base, soma_area_b - df['area_construida_m2'])

    df['preco_m2_medio_bairro_loo'] = soma_valor_loo / soma_area_loo

    preco_m2_global = (df_base['valor_base_calculo_real'].sum() /
                       df_base['area_construida_m2'].sum())
    df['preco_m2_medio_bairro_loo'] = df['preco_m2_medio_bairro_loo'].replace(
        [np.inf, -np.inf], np.nan).fillna(preco_m2_global)

    print(f"Feature criada: preco_m2_medio_bairro_loo")

    return df

def criar_features_valorizacao_bairro(df, anos_base=ANOS_BASE_BAIRRO):
    """
    Valorização real do bairro, calculada SÓ sobre os anos-base.
    O valor é uma característica do bairro (não do imóvel), então a mesma
    valorização aprendida na base é atribuída a todas as transações do
    bairro — inclusive as de teste, que recebem o que se conhecia até 2022.
    """
    print("\n[6/8] Calculando valorização do bairro (sem leakage temporal)...")

    df_base = df[df['ano_transacao'].isin(anos_base)]

    # Preço médio por bairro/ano, só na base.
    preco_ano = (df_base.groupby(['bairro', 'ano_transacao'])
                 ['valor_base_calculo_real'].mean().reset_index())
    preco_ano.columns = ['bairro', 'ano', 'preco_medio']

    valorizacao = []
    for bairro in df['bairro'].unique():
        df_bairro = preco_ano[preco_ano['bairro'] == bairro].sort_values('ano')

        if len(df_bairro) >= 2:
            precos = df_bairro['preco_medio'].values
            if len(precos) >= 4:
                preco_atual, preco_ref = precos[-1], precos[-4]
            else:
                preco_atual, preco_ref = precos[-1], precos[0]
            val = ((preco_atual - preco_ref) / preco_ref) * 100 if preco_ref > 0 else 0
        else:
            val = 0

        valorizacao.append({'bairro': bairro, 'valorizacao_bairro_3anos': val})

    df_valorizacao = pd.DataFrame(valorizacao)
    df = df.merge(df_valorizacao, on='bairro', how='left')
    df['valorizacao_bairro_3anos'] = df['valorizacao_bairro_3anos'].fillna(0)

    print(f"Feature criada: valorizacao_bairro_3anos")

    return df

def criar_features_comparativas(df, anos_base=ANOS_BASE_BAIRRO):
    """
    Compara área e idade do imóvel com a média do bairro.
    As médias de bairro são calculadas SÓ sobre os anos-base, então uma
    transação de teste é comparada com o perfil do bairro conhecido até 2022.
    Estas não dependem do alvo (área/idade), mas mantemos a separação
    temporal por consistência metodológica.
    """
    print("\n[7/8] Criando features comparativas (sem leakage temporal)...")

    df_base = df[df['ano_transacao'].isin(anos_base)]

    medias_area = df_base.groupby('bairro')['area_construida_m2'].mean()
    medias_idade = df_base.groupby('bairro')['idade_imovel'].mean()

    # Fallbacks globais (base) para bairros ausentes.
    area_global = df_base['area_construida_m2'].mean()
    idade_global = df_base['idade_imovel'].mean()

    area_ref = df['bairro'].map(medias_area).fillna(area_global)
    idade_ref = df['bairro'].map(medias_idade).fillna(idade_global)

    df['area_vs_media_bairro'] = df['area_construida_m2'] / area_ref
    df['idade_vs_media_bairro'] = df['idade_imovel'] / (idade_ref + 1)

    print(f"2 features comparativas criadas")

    return df

def criar_features_sazonalidade(df):
    print("\n[8/8] Criando features de sazonalidade...")

    df['trimestre'] = df['mes_transacao'].apply(lambda m: (m - 1) // 3 + 1)
    df['fim_de_ano'] = (df['mes_transacao'] >= 11).astype(int)
    df['inicio_ano'] = (df['mes_transacao'] <= 3).astype(int)

    print(f"3 features de sazonalidade criadas")

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