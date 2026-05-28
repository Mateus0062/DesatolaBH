import pandas as pd
import numpy as np
import re
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import *


def converter_numero_brasileiro(valor):
    if pd.isna(valor):
        return np.nan

    if isinstance(valor, (int, float)):
        return float(valor)

    valor = str(valor).strip()

    if valor == '' or valor == 'nan':
        return np.nan

    valor = valor.replace('.', '').replace(',', '.')

    try:
        return float(valor)
    except ValueError:
        return np.nan


def converter_data_brasileira(data_str):
    if pd.isna(data_str):
        return pd.NaT

    try:
        return pd.to_datetime(data_str, format='%d/%m/%Y')
    except ValueError:
        return pd.NaT


def extrair_cep(endereco):
    if pd.isna(endereco):
        return None

    match = re.search(r'(\d{5}-\d{3})', str(endereco))
    if match:
        return match.group(1).replace('-', '')
    return None


def renomear_colunas(df, verbose=True):
    mapeamento = {
        '_id': 'id',
        'Endereco': 'endereco',
        'Bairro': 'bairro',
        'Ano de Construcao Unidade': 'ano_construcao',
        'Area Terreno Total': 'area_terreno_m2',
        'Area Construida Adquirida': 'area_construida_m2',
        'Area Adquirida Unidades Somadas': 'area_total_m2',
        'Padrao Acabamento Unidade': 'padrao_acabamento',
        'Fracao Ideal Adquirida': 'fracao_ideal',
        'Tipo Construtivo Preponderante': 'tipo_construtivo',
        'Descricao Tipo Ocupacao Unidade': 'tipo_ocupacao',
        'Valor Declarado': 'valor_declarado',
        'Valor Base Calculo': 'valor_base_calculo',
        'Zona Uso ITBI': 'zona_uso',
        'Data Quitacao Transacao': 'data_transacao',
    }

    colunas_faltando = set(mapeamento.keys()) - set(df.columns)
    if colunas_faltando:
        print(f"ERRO: Colunas não encontradas no CSV:")
        for col in colunas_faltando:
            print(f" - '{col}'")
        raise KeyError(f"Colunas faltando: {colunas_faltando}")

    if verbose:
        print("[1/8] Renomeando colunas...")

    return df.rename(columns=mapeamento)


def converter_numeros_brasileiros(df, verbose=True):
    if verbose:
        print("\n[2/8] Convertendo valores numéricos...")

    colunas_numericas = [
        'area_terreno_m2', 'area_construida_m2', 'area_total_m2',
        'fracao_ideal', 'valor_declarado', 'valor_base_calculo',
    ]

    for col in colunas_numericas:
        df[col] = df[col].apply(converter_numero_brasileiro)
        if verbose:
            n_nulos = df[col].isna().sum()
            print(f"{col:25s} - {n_nulos:>6,} nulos")

    return df


def converter_datas(df, verbose=True):
    if verbose:
        print("\n[3/8] Convertendo datas...")

    df['data_transacao'] = df['data_transacao'].apply(converter_data_brasileira)
    df['ano_transacao'] = df['data_transacao'].dt.year
    df['mes_transacao'] = df['data_transacao'].dt.month

    if verbose:
        n_nulos = df['data_transacao'].isna().sum()
        print(f"data_transacao - {n_nulos:,} nulos")

    return df


def extrair_cep_coluna(df):
    print("\n[4/8] Extraindo CEP...")

    df['cep'] = df['endereco'].apply(extrair_cep)

    n_nulos = df['cep'].isna().sum()
    print(f"CEPs extraídos - {n_nulos:,} nulos")

    return df


def padronizar_bairros(df):
    print("\n[5/8] Padronizando bairros...")

    df['bairro'] = df['bairro'].str.strip().str.upper()

    n_bairros = df['bairro'].nunique()
    print(f"{n_bairros} bairros únicos")

    return df


def criar_feature_padrao_acabamento(df):
    print("\n[*/*] Codificando padrão de acabamento (ordinal)...")

    # Nulo disfarçado de texto, vindo do sistema da PBH -> NaN explícito.
    df['padrao_acabamento'] = df['padrao_acabamento'].replace('VALOR NULO', np.nan)

    mapa_padrao = {'P1': 1, 'P2': 2, 'P3': 3, 'P4': 4, 'P5': 5}
    df['padrao_acabamento_num'] = df['padrao_acabamento'].map(mapa_padrao)

    # Trava: qualquer valor NÃO-nulo fora de P1-P5 é um problema real.
    # (NaN é esperado para os 'VALOR NULO'; serão removidos no dropna.)
    mask_problema = df['padrao_acabamento_num'].isna() & df['padrao_acabamento'].notna()
    if mask_problema.any():
        valores_estranhos = df.loc[mask_problema, 'padrao_acabamento'].unique().tolist()
        raise ValueError(
            f"{mask_problema.sum()} registros com padrao_acabamento fora da "
            f"escala P1-P5: {valores_estranhos}. Atualize mapa_padrao."
        )

    n_nulos = df['padrao_acabamento_num'].isna().sum()
    print(f"padrao_acabamento_num criado (escala 1-5) — "
          f"{n_nulos} nulos a remover no dropna")
    return df


def criar_features_basicas(df):
    print("\n[6/8] Criando features básicas...")

    # Preço por m² (necessário para filtro rigoroso)
    df['preco_m2'] = df['valor_base_calculo_real'] / df['area_construida_m2']

    # Idade do imóvel
    df['idade_imovel'] = df['ano_transacao'] - df['ano_construcao']

    df['is_apartamento'] = (df['tipo_construtivo'] == 'AP').astype(int)

    print("preco_m2 criado")
    print("idade_imovel criado")
    print("is_apartamento criado")

    return df

def filtrar_subdeclaracao_por_classe(df):
    print(f"\n[7.6/8] Filtrando subdeclaração por piso de classe de bairro...")
    print(f"  Linhas antes: {len(df):,}")

    # Classe de cada imóvel (mesma regra do ibge_features: padrão = Médio).
    classe = df['bairro'].map(BAIRRO_PARA_CLASSE).fillna(GRUPO_BAIRRO_PADRAO)

    # Piso correspondente à classe de cada imóvel.
    piso = classe.map(PISO_PRECO_M2_POR_CLASSE)

    mask_ok = df['preco_m2'] >= piso
    removidos = int((~mask_ok).sum())

    print(f"  ✓ Subdeclarações removidas (preço/m² abaixo do piso da classe): "
          f"{removidos:,} ({removidos / len(df) * 100:.2f}%)")

    # Sanity check: o que está sendo cortado, por classe.
    df_rem = df[~mask_ok]
    if len(df_rem) > 0:
        nomes_classe = {1: 'Popular', 2: 'Médio', 3: 'Alto', 4: 'Luxo'}
        classe_rem = classe[~mask_ok]
        print(f"  Remoções por classe:")
        for c in sorted(nomes_classe):
            n = int((classe_rem == c).sum())
            if n:
                print(f"    {nomes_classe[c]:8s} (piso R$ "
                      f"{PISO_PRECO_M2_POR_CLASSE[c]:,}/m²): {n:,}")

        cols = ['bairro', 'tipo_construtivo', 'area_construida_m2',
                'valor_base_calculo_real', 'preco_m2']
        print(f"\n  Amostra (menores preços/m² removidos):")
        for _, r in df_rem.nsmallest(8, 'preco_m2')[cols].iterrows():
            print(f"    {r['bairro']:18s} {r['tipo_construtivo']:3s} "
                  f"{r['area_construida_m2']:>6.0f}m²  "
                  f"R$ {r['valor_base_calculo_real']:>12,.0f}  "
                  f"(R$ {r['preco_m2']:>8,.0f}/m²)")

    df = df[mask_ok].copy()
    print(f"\n  Linhas depois: {len(df):,}")
    return df

def aplicar_filtros_rigorosos(df):
    from config import (
        MIN_AREA_M2, MAX_AREA_M2,
        MIN_VALOR, MAX_VALOR,
        MIN_PRECO_M2, MAX_PRECO_M2,
        MAX_IDADE_IMOVEL
    )

    print("\n[7/8] Aplicando filtros rigorosos...")
    print(f"Linhas antes dos filtros: {len(df):,}")

    linhas_inicial = len(df)

    # FILTRO 0: Manter apenas imóveis residenciais e manter apenas casas (CA) e apartamentos (AP)
    mask_residencial = df['tipo_ocupacao'] == 'RESIDENCIAL'
    removidos_residencial = (~mask_residencial).sum()
    df = df[mask_residencial].copy()
    print(f"Filtro residencial: {removidos_residencial:,} removidos "
          f"(não-residenciais)")

    # Filtro para deixar apenas casas e apartamentos.
    tipos_habitacionais = ['AP', 'CA']
    mask_tipo = df['tipo_construtivo'].isin(tipos_habitacionais)
    removidos_tipo = (~mask_tipo).sum()
    df = df[mask_tipo].copy()
    print(f"Filtro tipo habitacional: {removidos_tipo:,} removidos "
          f"(mantidos apenas AP e CA)")

    # FILTRO 1: Área válida
    mask_area = (
            (df['area_total_m2'] >= MIN_AREA_M2) &
            (df['area_total_m2'] <= MAX_AREA_M2) &
            (df['area_construida_m2'] >= MIN_AREA_M2) &
            (df['area_construida_m2'] <= MAX_AREA_M2) &
            (df['area_construida_m2'] <= df['area_total_m2'])
    )
    removidos_area = (~mask_area).sum()
    df = df[mask_area].copy()

    print(f"Filtro área: {removidos_area:,} removidos")

    # FILTRO 2: Valor válido
    mask_valor = (
            (df['valor_base_calculo_real'] >= MIN_VALOR) &
            (df['valor_base_calculo_real'] <= MAX_VALOR)
    )
    removidos_valor = (~mask_valor).sum()
    df = df[mask_valor].copy()

    print(f"Filtro valor: {removidos_valor:,} removidos")

    # FILTRO 3: Preço por m² válido (CRÍTICO!)
    mask_preco_m2 = (
            (df['preco_m2'] >= MIN_PRECO_M2) &
            (df['preco_m2'] <= MAX_PRECO_M2)
    )
    removidos_preco_m2 = (~mask_preco_m2).sum()
    df = df[mask_preco_m2].copy()

    print(f"Filtro preço/m²: {removidos_preco_m2:,} removidos")

    # FILTRO 4: Idade válida
    mask_idade = df['idade_imovel'] <= MAX_IDADE_IMOVEL
    removidos_idade = (~mask_idade).sum()
    df = df[mask_idade].copy()

    print(f"Filtro idade: {removidos_idade:,} removidos")

    # FILTRO 5: Duplicatas
    duplicatas_antes = len(df)
    df = df.drop_duplicates(
        subset=['endereco', 'valor_base_calculo_real', 'data_transacao'],
        keep='first'
    )
    removidos_duplicatas = duplicatas_antes - len(df)

    print(f"Duplicatas: {removidos_duplicatas:,} removidos")

    # FILTRO 6: Outliers de preço/m² relativos ao bairro
    ANOS_TREINO = list(range(2008, 2022))
    df = filtrar_outliers_preco_m2_por_bairro(
        df, anos_treino=ANOS_TREINO,
        k_iqr_inf=1.5, k_iqr_sup=3.0, min_transacoes=30
    )

    # Filtro 7
    df = filtrar_subdeclaracao_por_classe(df)

    # Resumo
    linhas_final = len(df)
    total_removido = linhas_inicial - linhas_final
    pct_removido = (total_removido / linhas_inicial) * 100

    print(f"\nResumo da limpeza rigorosa:")
    print(f"Linhas iniciais: {linhas_inicial:>8,}")
    print(f"Linhas finais: {linhas_final:>8,}")
    print(f"Removidas: {total_removido:>8,} ({pct_removido:.1f}%)")

    return df

def filtrar_outliers_preco_m2_por_bairro(df, anos_treino=None, k_iqr_inf=1.5, k_iqr_sup=3.0, min_transacoes=30):
    print(f"\n[7.5/8] Filtrando outliers de preço/m² por bairro e tipo...")
    print(f"Linhas antes: {len(df):,}")

    log_preco_m2 = np.log1p(df['preco_m2'])

    # Chave de grupo: bairro + tipo construtivo, como uma única série.
    chave = df['bairro'].astype(str) + ' | ' + df['tipo_construtivo'].astype(str)

    # Base para as estatísticas: só treino, se anos_treino for informado.
    if anos_treino is not None:
        mask_base = df['ano_transacao'].isin(anos_treino)
        df_base = df[mask_base]
        print(f"Estatísticas calculadas sobre {len(df_base):,} "
              f"linhas de treino (anos {sorted(anos_treino)})")
    else:
        df_base = df
        print(f"Estatísticas calculadas sobre todo o dataset "
              f"(atenção: possível leakage leve)")

    chave_base = chave[df_base.index]
    log_base = np.log1p(df_base['preco_m2'])

    g = log_base.groupby(chave_base)
    quartis = g.quantile([0.25, 0.75]).unstack()
    quartis.columns = ['q1', 'q3']

    stats = pd.DataFrame({
        'n': g.count(),
        'q1': quartis['q1'],
        'q3': quartis['q3'],
    })
    stats['iqr'] = stats['q3'] - stats['q1']

    # Corte robusto assimétrico
    stats['lim_inf'] = stats['q1'] - k_iqr_inf * stats['iqr']
    stats['lim_sup'] = stats['q3'] + k_iqr_sup * stats['iqr']

    # Grupos (bairro+tipo) com poucas transações: estatística não confiável.
    grupos_confiaveis = stats['n'] >= min_transacoes
    stats.loc[~grupos_confiaveis, 'lim_inf'] = -np.inf
    stats.loc[~grupos_confiaveis, 'lim_sup'] = np.inf

    n_conf = int(grupos_confiaveis.sum())
    print(f"Grupos (bairro+tipo) com estatística confiável "
          f"(>= {min_transacoes} transações): {n_conf} de {len(stats)}")

    # Mapear limites de volta para cada linha pelo par bairro+tipo.
    lim_inf = chave.map(stats['lim_inf']).fillna(-np.inf)
    lim_sup = chave.map(stats['lim_sup']).fillna(np.inf)

    mask_ok = (log_preco_m2 >= lim_inf) & (log_preco_m2 <= lim_sup)
    removidos = int((~mask_ok).sum())

    print(f"Outliers removidos: {removidos:,} "
          f"({removidos / len(df) * 100:.2f}%)")

    # Sanity check: amostra dos dois extremos.
    df_removidos = df[~mask_ok]
    if len(df_removidos) > 0:
        cols = ['bairro', 'tipo_construtivo', 'area_construida_m2',
                'valor_base_calculo_real', 'preco_m2']

        print(f"\nAmostra de remoções (menores preços/m²):")
        for _, r in df_removidos.nsmallest(5, 'preco_m2')[cols].iterrows():
            print(f"{r['bairro']:18s} {r['tipo_construtivo']:3s} "
                  f"{r['area_construida_m2']:>6.0f}m²  "
                  f"R$ {r['valor_base_calculo_real']:>12,.0f}  "
                  f"(R$ {r['preco_m2']:>8,.0f}/m²)")

        print(f"\nAmostra de remoções (maiores preços/m²):")
        for _, r in df_removidos.nlargest(5, 'preco_m2')[cols].iterrows():
            print(f"{r['bairro']:18s} {r['tipo_construtivo']:3s} "
                  f"{r['area_construida_m2']:>6.0f}m²  "
                  f"R$ {r['valor_base_calculo_real']:>12,.0f}  "
                  f"(R$ {r['preco_m2']:>8,.0f}/m²)")

    df = df[mask_ok].copy()
    print(f"\nLinhas depois: {len(df):,}")
    return df

def _construir_indice_ipca():
    meses = sorted(IPCA_VAR_MENSAL.keys())
    indice = {}
    acumulado = 1.0
    for mes in meses:
        acumulado *= (1.0 + IPCA_VAR_MENSAL[mes] / 100.0)
        indice[mes] = acumulado

    indice_base = indice[IPCA_MES_BASE]
    return {mes: indice_base / valor for mes, valor in indice.items()}

def deflacionar_valores(df):
    print("\n[3.5/8] Deflacionando valores pelo IPCA (base dez/2024)...")

    fatores = _construir_indice_ipca()

    # Chave (ano, mes) de cada transação
    chaves = list(zip(df['ano_transacao'].astype('Int64'),
                      df['mes_transacao'].astype('Int64')))

    # Trava tod-o mes presente nos dados precisa ter fator IPCA
    meses_sem_fator = sorted({c for c in chaves
                              if c not in fatores and pd.notna(c[0])})
    if meses_sem_fator:
        raise ValueError(
            f"Meses sem fator IPCA na tabela embutida: {meses_sem_fator}. "
            f"Atualize IPCA_VAR_MENSAL no Limpeza.py com esses meses."
        )

    fator_serie = pd.Series([fatores.get(c, np.nan) for c in chaves],
                            index=df.index)

    df['valor_base_calculo_real'] = df['valor_base_calculo'] * fator_serie
    df['valor_declarado_real'] = df['valor_declarado'] * fator_serie

    f_min, f_max = fator_serie.min(), fator_serie.max()
    print(f"Fatores aplicados — varia de {f_min:.3f}x "
          f"(transações recentes) a {f_max:.3f}x (mais antigas)")
    med_nom = df['valor_base_calculo'].median()
    med_real = df['valor_base_calculo_real'].median()
    print(f"Mediana base de cálculo: "
          f"R$ {med_nom:,.0f} (nominal) -> "
          f"R$ {med_real:,.0f} (dez/2024)")

    return df

def limpar_dataset_itbi(df=None, path_input=None):
    # Carregar dados
    if df is None:
        if path_input is None:
            path_input = ITBI_RAW
        df = pd.read_csv(path_input, encoding='utf-8')

    print("PIPELINE DE LIMPEZA - ITBI BELO HORIZONTE")

    print(f"\nArquivo de entrada: {path_input}")
    print(f"Linhas iniciais: {len(df):,}\n")

    df_clean = df.copy()

    # Executar pipeline
    df_clean = renomear_colunas(df_clean)
    df_clean = converter_numeros_brasileiros(df_clean)
    df_clean = converter_datas(df_clean)
    df_clean = deflacionar_valores(df_clean)
    df_clean = extrair_cep_coluna(df_clean)
    df_clean = padronizar_bairros(df_clean)
    df_clean = criar_features_basicas(df_clean)  # ANTES dos filtros!
    df_clean = aplicar_filtros_rigorosos(df_clean)  # Filtros rigorosos
    df_clean = criar_feature_padrao_acabamento(df_clean)

    # Remover NaNs finais
    print(f"\n[8/8] Removendo valores faltantes finais...")
    print(f"Linhas antes: {len(df_clean):,}")

    df_clean = df_clean.dropna(subset=['valor_base_calculo_real', 'area_construida_m2',
                                       'data_transacao', 'bairro',
                                       'padrao_acabamento_num'])

    print(f"Linhas depois: {len(df_clean):,}")

    # Resumo final
    print("LIMPEZA CONCLUÍDA")

    print(f"\nDataset final: {len(df_clean):,} linhas, {len(df_clean.columns)} colunas")
    print(f"Período: {df_clean['ano_transacao'].min():.0f} - {df_clean['ano_transacao'].max():.0f}")
    print(f"Valor médio (base cálc., dez/2024): R$ {df_clean['valor_base_calculo_real'].mean():,.2f}")
    print(f"Área média: {df_clean['area_total_m2'].mean():.2f} m²")
    print(f"Preço/m² médio: R$ {df_clean['preco_m2'].mean():,.2f}")

    return df_clean


def salvar_dataset_limpo(df, path_output=None):
    if path_output is None:
        path_output = ITBI_CLEANED

    df.to_csv(path_output, index=False, encoding='utf-8')

    print(f"\nDataset salvo: {path_output}")
    print(f"Tamanho: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")


if __name__ == "__main__":
    print("Carregando dados brutos...")
    df_clean = limpar_dataset_itbi()

    print("\nSalvando dataset limpo...")
    salvar_dataset_limpo(df_clean)

    print("\nPROCESSO CONCLUÍDO")