"""
Módulo de Limpeza de Dados - ITBI Belo Horizonte
Pipeline completo com filtros rigorosos
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import *


def converter_numero_brasileiro(valor):
    """Converte número no formato brasileiro (1.234,56) para float"""
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
    """Converte data no formato DD/MM/YYYY para datetime"""
    if pd.isna(data_str):
        return pd.NaT

    try:
        return pd.to_datetime(data_str, format='%d/%m/%Y')
    except ValueError:
        return pd.NaT


def extrair_cep(endereco):
    """Extrai CEP do endereço (formato XXXXX-XXX)"""
    if pd.isna(endereco):
        return None

    match = re.search(r'(\d{5}-\d{3})', str(endereco))
    if match:
        return match.group(1).replace('-', '')
    return None


def renomear_colunas(df, verbose=True):
    """Renomeia colunas para padrão snake_case"""

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
            print(f"   - '{col}'")
        raise KeyError(f"Colunas faltando: {colunas_faltando}")

    if verbose:
        print("[1/8] Renomeando colunas...")

    return df.rename(columns=mapeamento)


def converter_numeros_brasileiros(df, verbose=True):
    """Converte todas as colunas numéricas"""

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
            print(f"  {col:25s} - {n_nulos:>6,} nulos")

    return df


def converter_datas(df, verbose=True):
    """Converte datas e extrai ano/mês"""

    if verbose:
        print("\n[3/8] Convertendo datas...")

    df['data_transacao'] = df['data_transacao'].apply(converter_data_brasileira)
    df['ano_transacao'] = df['data_transacao'].dt.year
    df['mes_transacao'] = df['data_transacao'].dt.month

    if verbose:
        n_nulos = df['data_transacao'].isna().sum()
        print(f"  data_transacao - {n_nulos:,} nulos")

    return df


def extrair_cep_coluna(df, verbose=True):
    """Extrai CEP do endereço"""

    if verbose:
        print("\n[4/8] Extraindo CEP...")

    df['cep'] = df['endereco'].apply(extrair_cep)

    if verbose:
        n_nulos = df['cep'].isna().sum()
        print(f"  CEPs extraídos - {n_nulos:,} nulos")

    return df


def padronizar_bairros(df, verbose=True):
    """Padroniza nomes de bairros"""

    if verbose:
        print("\n[5/8] Padronizando bairros...")

    df['bairro'] = df['bairro'].str.strip().str.upper()

    if verbose:
        n_bairros = df['bairro'].nunique()
        print(f"  {n_bairros} bairros únicos")

    return df


def criar_features_basicas(df, verbose=True):
    """Cria features básicas (ANTES dos filtros rigorosos)"""

    if verbose:
        print("\n[6/8] Criando features básicas...")

    # Preço por m² (necessário para filtro rigoroso)
    df['preco_m2'] = df['valor_declarado'] / df['area_total_m2']

    # Idade do imóvel
    df['idade_imovel'] = df['ano_transacao'] - df['ano_construcao']

    # Flag residencial
    df['is_residencial'] = (df['tipo_ocupacao'] == 'RESIDENCIAL').astype(int)

    if verbose:
        print(f"  ✓ preco_m2 criado")
        print(f"  ✓ idade_imovel criado")
        print(f"  ✓ is_residencial criado")

    return df


def aplicar_filtros_rigorosos(df, verbose=True):
    """
    Aplica filtros rigorosos para remover outliers e erros
    """
    from config import (
        MIN_AREA_M2, MAX_AREA_M2,
        MIN_VALOR, MAX_VALOR,
        MIN_PRECO_M2, MAX_PRECO_M2,
        MAX_IDADE_IMOVEL
    )

    if verbose:
        print("\n[7/8] Aplicando filtros rigorosos...")
        print(f"  Linhas antes dos filtros: {len(df):,}")

    linhas_inicial = len(df)

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

    if verbose:
        print(f"  ✓ Filtro área: {removidos_area:,} removidos")

    # FILTRO 2: Valor válido
    mask_valor = (
            (df['valor_declarado'] >= MIN_VALOR) &
            (df['valor_declarado'] <= MAX_VALOR)
    )
    removidos_valor = (~mask_valor).sum()
    df = df[mask_valor].copy()

    if verbose:
        print(f"  ✓ Filtro valor: {removidos_valor:,} removidos")

    # FILTRO 3: Preço por m² válido (CRÍTICO!)
    mask_preco_m2 = (
            (df['preco_m2'] >= MIN_PRECO_M2) &
            (df['preco_m2'] <= MAX_PRECO_M2)
    )
    removidos_preco_m2 = (~mask_preco_m2).sum()
    df = df[mask_preco_m2].copy()

    if verbose:
        print(f"  ✓ Filtro preço/m²: {removidos_preco_m2:,} removidos")

    # FILTRO 4: Idade válida
    mask_idade = df['idade_imovel'] <= MAX_IDADE_IMOVEL
    removidos_idade = (~mask_idade).sum()
    df = df[mask_idade].copy()

    if verbose:
        print(f"  ✓ Filtro idade: {removidos_idade:,} removidos")

    # FILTRO 5: Duplicatas
    duplicatas_antes = len(df)
    df = df.drop_duplicates(
        subset=['endereco', 'valor_declarado', 'data_transacao'],
        keep='first'
    )
    removidos_duplicatas = duplicatas_antes - len(df)

    if verbose:
        print(f"  ✓ Duplicatas: {removidos_duplicatas:,} removidos")

    # Resumo
    linhas_final = len(df)
    total_removido = linhas_inicial - linhas_final
    pct_removido = (total_removido / linhas_inicial) * 100

    if verbose:
        print(f"\n  Resumo da limpeza rigorosa:")
        print(f"    Linhas iniciais:  {linhas_inicial:>8,}")
        print(f"    Linhas finais:    {linhas_final:>8,}")
        print(f"    Removidas:        {total_removido:>8,} ({pct_removido:.1f}%)")

    return df


def limpar_dataset_itbi(df=None, path_input=None, verbose=True):
    """
    Pipeline completo de limpeza do dataset ITBI
    """

    # Carregar dados
    if df is None:
        if path_input is None:
            path_input = ITBI_RAW
        df = pd.read_csv(path_input, encoding='utf-8')

    if verbose:
        print("=" * 80)
        print("PIPELINE DE LIMPEZA - ITBI BELO HORIZONTE")
        print("=" * 80)
        print(f"\nArquivo de entrada: {path_input}")
        print(f"Linhas iniciais: {len(df):,}\n")

    df_clean = df.copy()

    # Executar pipeline
    df_clean = renomear_colunas(df_clean, verbose)
    df_clean = converter_numeros_brasileiros(df_clean, verbose)
    df_clean = converter_datas(df_clean, verbose)
    df_clean = extrair_cep_coluna(df_clean, verbose)
    df_clean = padronizar_bairros(df_clean, verbose)
    df_clean = criar_features_basicas(df_clean, verbose)  # ANTES dos filtros!
    df_clean = aplicar_filtros_rigorosos(df_clean, verbose)  # Filtros rigorosos

    # Remover NaNs finais
    if verbose:
        print(f"\n[8/8] Removendo valores faltantes finais...")
        print(f"  Linhas antes: {len(df_clean):,}")

    df_clean = df_clean.dropna(subset=['valor_declarado', 'area_total_m2',
                                       'data_transacao', 'bairro'])

    if verbose:
        print(f"  Linhas depois: {len(df_clean):,}")

    # Resumo final
    if verbose:
        print("\n" + "=" * 80)
        print("LIMPEZA CONCLUÍDA")
        print("=" * 80)
        print(f"\nDataset final: {len(df_clean):,} linhas, {len(df_clean.columns)} colunas")
        print(f"Período: {df_clean['ano_transacao'].min():.0f} - {df_clean['ano_transacao'].max():.0f}")
        print(f"Valor médio: R$ {df_clean['valor_declarado'].mean():,.2f}")
        print(f"Área média: {df_clean['area_total_m2'].mean():.2f} m²")
        print(f"Preço/m² médio: R$ {df_clean['preco_m2'].mean():,.2f}")

    return df_clean


def salvar_dataset_limpo(df, path_output=None, verbose=True):
    if path_output is None:
        path_output = ITBI_CLEANED

    df.to_csv(path_output, index=False, encoding='utf-8')

    if verbose:
        print(f"\n✓ Dataset salvo: {path_output}")
        print(f"  Tamanho: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")


if __name__ == "__main__":
    print("Carregando dados brutos...")
    df_clean = limpar_dataset_itbi()

    print("\nSalvando dataset limpo...")
    salvar_dataset_limpo(df_clean)

    print("\n✓✓✓ PROCESSO CONCLUÍDO ✓✓✓")