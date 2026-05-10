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
        pd.NaT

def extrair_cep(endereco):
    if pd.isna(endereco):
        return None

    match = re.search(r'(\d{5}-\d{3})', str(endereco))
    if match:
        return match.group(1).replace('-', '')
    return None

def renomear_colunas(df):

    # Função para renomear as colunas da base de dados,
    # removendo espaços em branco,
    # letras maiusculas e deixando o texto menor

    # Esse mapeamento vai ser alterado a medida que formos avançando com o projeto.
    # Pois, em cada etapa removeremos as colunas menos importantes, afim de deixar a base menor e mais objetiva.

    mapeamento = {
        '_id': 'id',
        'Endereco': 'endereco',
        'Bairro': 'bairro',
        'Ano Construcao Unidade': 'ano_construcao',
        'Area Terreno Total': 'area_terreno_m2',
        'Area Construida Adquirida': 'area_construida_m2',
        'Area Adquirida Unidades Somadas':'area_total_m2',
        'Padrao Acabamento Unidade': 'padrao_acabamento',
        'Fracao Ideal Adquirida': 'fracao_ideal',
        'Tipo Construtivo Preponderante': 'tipo_construtivo',
        'Descricao Tipo Ocupacao Unidade': 'tipo_ocupacao',
        'Valor Declarado': 'valor_declarado',
        'Valor Base Calculo': 'valor_base_calculo',
        'Zona Uso ITBI': 'zona_uso',
        'Data Quitacao Transacao': 'data_transacao',
    }

    return df.rename(columns=mapeamento)

def limpar_dataset_itbi(df=None, path_input=None, verbose=True):
    if df is None:
        if path_input is None:
            path_input = ITBI_RAW
        df = pd.read_csv(path_input, encoding='utf-8')

    if verbose:
        print("INICIANDO LIMPEZA DO DATASET")

    df_clean = df.copy()

    # ===== 1 - Renomear =============================

    if verbose:
        print("\n[1/7] Renomeando colunas...")
    df_clean = renomear_colunas(df_clean)

    # ===== 2 - Converter Numéricos ==================
    if verbose:
        print("\n[2/7] Convertendo valores numéricos...")

    colunas_numericas = [
        'area_terreno_m2', 'area_construida_m2', 'area_total_m2',
        'fracao_ideal', 'valor_declarado', 'valor_base_calculo',
    ]

    for col in colunas_numericas:
        df_clean[col] = df_clean[col].apply(converter_numero_brasileiro)
        if verbose:
            n_nulos = df_clean[col].isna().sum()
            print(f" * {col:25s} - {n_nulos:>6,} nulos")

    # ===== 3 - Datas ================================

    if verbose:
        print("\n[3/7] Convertendo datas...")

    df_clean['data_transacao'] = df_clean['data_transacao'].apply(converter_data_brasileira)
    df_clean['ano_transacao'] = df_clean['data_transacao'].dt.year
    df_clean['mes_transacao'] = df_clean['data_transacao'].dt.month

    # ===== 4 - CEP ==================================

    if verbose:
        print("\n[4/7] Extraindo CEP...")

    df_clean['cep'] = df_clean['cep'].apply(extrair_cep)

    # ===== 5 - Bairros ==============================

    if verbose:
        print("\n[5/7] Padronizando Bairros...")

    df_clean['bairro'] = df_clean['bairro'].str.strip().str.upper()

    # ===== 6 - Outliers =============================

    if verbose:
        print("\n[6/7] Removendo Outliers...")

    len_antes = len(df_clean)

    df_clean = df_clean[
        # Tratando Area Total
        (df_clean['area_total_m2'] > 0) &
        (df_clean['area_total_m2'] >= MIN_AREA_M2) &
        (df_clean['area_total_m2'] <= MAX_AREA_M2) &

        # Tratando o Valor Declarado
        (df_clean['valor_declarado'] > 0) &
        (df_clean['valor_declarado'] >= MIN_VALOR) &
        (df_clean['valor_declarado'] <= MAX_VALOR) &

        # Tratando o ano da Construção
        (df_clean['ano_construcao'] >= MIN_ANO_CONSTRUCAO) &
        (df_clean['ano_construcao'] <= MAX_ANO_CONSTRUCAO) &

        # Tratando a data de transação
        (df_clean['data_transacao'].notna())
    ]

    removidos = len_antes - len(df_clean)
    if verbose:
        print(f" * Removidas: {removidos:,} ({removidos/len_antes*100:.2f}%)")

    # 7 - Features Básicas ===========================

    if verbose:
        print(f"[7/7] Criando features básicas...")

    df_clean['preco_m2'] = df_clean['valor_declarado'] / df_clean['area_total_m2']
    df_clean['idade_imovel'] = df_clean['ano_transacao'] - df_clean['ano_construcao']
    df_clean['is_residencial'] = (df_clean['tipo_ocupacao'] == 'RESIDENCIAL').astype(int)

    # ===== Finalização ==============================

    print("LIMPEZA CONCLUIDA!")
    print(f"Linhas finais: {len(df_clean):,}")
    print(f"Colunas: {len(df_clean.columns)}")
    print(f"Período: {df_clean['ano_transacao'].min():.0f} - {df_clean['ano_transacao'].max():.0f}")
    print(f"Valor médio: R$ {df_clean['valor_declarado'].mean():,.2f}")
    print(f"Área média: {df_clean['area_total_m2'].mean():.2f} m²")
    print(f"Preço/m² médio: R$ {df_clean['preco_m2'].mean():,.2f}")

    return df_clean

def salvar_dataset_limpo(df, path_output=None):
    if path_output is None:
        path_output = ITBI_CLEANED

    df.to_csv(path_output, index=False, encoding='utf-8')

    print(f"Dataset salvo: {path_output}")
    print(f"Tamanho: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

if __name__ == "__main__":
    print("Carregando dados brutos...")
    df_clean = limpar_dataset_itbi(verbose=True)

    print("\nSalvando dataset limpo...")
    salvar_dataset_limpo(df_clean)

    print("PROCESSO CONCLUÍDO!")