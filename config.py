from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED = BASE_DIR / 'data' / 'processed'
DATA_FINAL = BASE_DIR / 'data' / 'final'
DATA_EXTERNAL = BASE_DIR / 'data' / 'external'

OUTPUTS_MODELS = BASE_DIR / 'outputs' / 'models'
OUTPUTS_FIGURES = BASE_DIR / 'outputs' / 'figures'
OUTPUTS_TABLES = BASE_DIR / 'outputs' / 'tables'

# Criar diretórios se não existirem
for dir_path in [DATA_RAW, DATA_PROCESSED, DATA_FINAL,
                 OUTPUTS_MODELS, OUTPUTS_FIGURES, OUTPUTS_TABLES]:
    dir_path.mkdir(parents=True, exist_ok=True)

ITBI_RAW = DATA_RAW / 'ITBI_Relatorios.csv'
ITBI_CLEANED = DATA_PROCESSED / 'ITBI_cleaned.csv'
ITBI_FINAL = DATA_FINAL / 'ITBI_final_features.csv'

MIN_AREA_M2 = 10
MAX_AREA_M2 = 10_000
MIN_VALOR = 50_000
MAX_VALOR = 50_000_000
MIN_ANO_CONSTRUCAO = 1900
MAX_ANO_CONSTRUCAO = 2024

# Preço por m² (novo filtro!)
MIN_PRECO_M2 = 500        # ← R$ 500/m² mínimo (remove declarações fraudulentas)
MAX_PRECO_M2 = 30_000     # ← R$ 30k/m² máximo (imóveis ultra-premium são outliers)

MAX_IDADE_IMOVEL = 150    # ← Remove imóveis "antigos demais" (provavelmente erro)

TARGET = 'valor_base_calculo_real'   # base de cálculo do ITBI, deflacionada (dez/2024)

# Colunas que NUNCA entram no modelo, com o motivo de cada uma.
COLUNAS_EXCLUIR_MODELO = [
    # --- Metadados (não preditivos) ---
    'id', 'endereco', 'data_transacao',

    # --- Identificadores categóricos string ---
    # 'bairro' vira features numéricas via ibge_features / loo; 'cep' idem.
    'bairro', 'cep',

    # --- Categóricas string substituídas por versão numérica ---
    # padrao_acabamento -> padrao_acabamento_num (ordinal)
    # tipo_construtivo  -> is_apartamento (binária)
    # grupo_bairro      -> grupo_bairro_num
    'padrao_acabamento', 'tipo_construtivo', 'tipo_ocupacao', 'zona_uso',
    'grupo_bairro',

    # --- LEAKAGE: são o target, quase-o-target, ou derivam dele ---
    'valor_declarado',          # subdeclarado, correlaciona com o target
    'valor_declarado_real',     # idem, deflacionado
    'valor_base_calculo',       # é o target em valor nominal
    'preco_m2',                 # = valor_base_calculo_real / area
    'preco_m2_x_idade', 'densidade_x_preco',
    'preco_medio_bairro', 'preco_medio_bairro_ano', 'preco_relativo_bairro',

    # --- Splitters temporais: usados para dividir, não para prever ---
    'ano_transacao', 'mes_transacao',
]

IPCA_VAR_MENSAL = {
    (2008, 1): 0.54, (2008, 2): 0.49, (2008, 3): 0.48, (2008, 4): 0.55,
    (2008, 5): 0.79, (2008, 6): 0.74, (2008, 7): 0.53, (2008, 8): 0.28,
    (2008, 9): 0.26, (2008, 10): 0.45, (2008, 11): 0.36, (2008, 12): 0.28,
    (2009, 1): 0.48, (2009, 2): 0.55, (2009, 3): 0.20, (2009, 4): 0.48,
    (2009, 5): 0.47, (2009, 6): 0.36, (2009, 7): 0.24, (2009, 8): 0.15,
    (2009, 9): 0.24, (2009, 10): 0.28, (2009, 11): 0.41, (2009, 12): 0.37,
    (2010, 1): 0.75, (2010, 2): 0.78, (2010, 3): 0.52, (2010, 4): 0.57,
    (2010, 5): 0.43, (2010, 6): 0.00, (2010, 7): 0.01, (2010, 8): 0.04,
    (2010, 9): 0.45, (2010, 10): 0.75, (2010, 11): 0.83, (2010, 12): 0.63,
    (2011, 1): 0.83, (2011, 2): 0.80, (2011, 3): 0.79, (2011, 4): 0.77,
    (2011, 5): 0.47, (2011, 6): 0.15, (2011, 7): 0.16, (2011, 8): 0.37,
    (2011, 9): 0.53, (2011, 10): 0.43, (2011, 11): 0.52, (2011, 12): 0.50,
    (2012, 1): 0.56, (2012, 2): 0.45, (2012, 3): 0.21, (2012, 4): 0.64,
    (2012, 5): 0.36, (2012, 6): 0.08, (2012, 7): 0.43, (2012, 8): 0.41,
    (2012, 9): 0.57, (2012, 10): 0.59, (2012, 11): 0.60, (2012, 12): 0.79,
    (2013, 1): 0.86, (2013, 2): 0.60, (2013, 3): 0.47, (2013, 4): 0.55,
    (2013, 5): 0.37, (2013, 6): 0.26, (2013, 7): 0.03, (2013, 8): 0.24,
    (2013, 9): 0.35, (2013, 10): 0.57, (2013, 11): 0.54, (2013, 12): 0.92,
    (2014, 1): 0.55, (2014, 2): 0.69, (2014, 3): 0.92, (2014, 4): 0.67,
    (2014, 5): 0.46, (2014, 6): 0.40, (2014, 7): 0.01, (2014, 8): 0.25,
    (2014, 9): 0.57, (2014, 10): 0.42, (2014, 11): 0.51, (2014, 12): 0.78,
    (2015, 1): 1.24, (2015, 2): 1.22, (2015, 3): 1.32, (2015, 4): 0.71,
    (2015, 5): 0.74, (2015, 6): 0.79, (2015, 7): 0.62, (2015, 8): 0.22,
    (2015, 9): 0.54, (2015, 10): 0.82, (2015, 11): 1.01, (2015, 12): 0.96,
    (2016, 1): 1.27, (2016, 2): 0.90, (2016, 3): 0.43, (2016, 4): 0.61,
    (2016, 5): 0.78, (2016, 6): 0.35, (2016, 7): 0.52, (2016, 8): 0.44,
    (2016, 9): 0.08, (2016, 10): 0.26, (2016, 11): 0.18, (2016, 12): 0.30,
    (2017, 1): 0.38, (2017, 2): 0.33, (2017, 3): 0.25, (2017, 4): 0.14,
    (2017, 5): 0.31, (2017, 6): -0.23, (2017, 7): 0.24, (2017, 8): 0.19,
    (2017, 9): 0.16, (2017, 10): 0.42, (2017, 11): 0.28, (2017, 12): 0.44,
    (2018, 1): 0.29, (2018, 2): 0.32, (2018, 3): 0.09, (2018, 4): 0.22,
    (2018, 5): 0.40, (2018, 6): 1.26, (2018, 7): 0.33, (2018, 8): -0.09,
    (2018, 9): 0.48, (2018, 10): 0.45, (2018, 11): -0.21, (2018, 12): 0.15,
    (2019, 1): 0.32, (2019, 2): 0.43, (2019, 3): 0.75, (2019, 4): 0.57,
    (2019, 5): 0.13, (2019, 6): 0.01, (2019, 7): 0.19, (2019, 8): 0.11,
    (2019, 9): -0.04, (2019, 10): 0.10, (2019, 11): 0.51, (2019, 12): 1.15,
    (2020, 1): 0.21, (2020, 2): 0.25, (2020, 3): 0.07, (2020, 4): -0.31,
    (2020, 5): -0.38, (2020, 6): 0.26, (2020, 7): 0.36, (2020, 8): 0.24,
    (2020, 9): 0.64, (2020, 10): 0.86, (2020, 11): 0.89, (2020, 12): 1.35,
    (2021, 1): 0.25, (2021, 2): 0.86, (2021, 3): 0.93, (2021, 4): 0.31,
    (2021, 5): 0.83, (2021, 6): 0.53, (2021, 7): 0.96, (2021, 8): 0.87,
    (2021, 9): 1.16, (2021, 10): 1.25, (2021, 11): 0.95, (2021, 12): 0.73,
    (2022, 1): 0.54, (2022, 2): 1.01, (2022, 3): 1.62, (2022, 4): 1.06,
    (2022, 5): 0.47, (2022, 6): 0.67, (2022, 7): -0.68, (2022, 8): -0.36,
    (2022, 9): -0.29, (2022, 10): 0.59, (2022, 11): 0.41, (2022, 12): 0.62,
    (2023, 1): 0.53, (2023, 2): 0.84, (2023, 3): 0.71, (2023, 4): 0.61,
    (2023, 5): 0.23, (2023, 6): -0.08, (2023, 7): 0.12, (2023, 8): 0.23,
    (2023, 9): 0.26, (2023, 10): 0.24, (2023, 11): 0.28, (2023, 12): 0.56,
    (2024, 1): 0.42, (2024, 2): 0.83, (2024, 3): 0.16, (2024, 4): 0.38,
    (2024, 5): 0.46, (2024, 6): 0.21, (2024, 7): 0.38, (2024, 8): -0.02,
    (2024, 9): 0.44, (2024, 10): 0.56, (2024, 11): 0.39, (2024, 12): 0.52,
}

IPCA_MES_BASE = (2024, 12)  # data-base: reais de dezembro/2024

# Criando variaveis para armazenar o parametros dos modelos aqui

HIPERPARAMETROS = {
    'xgboost': {
        'colsample_bytree': 0.6697465716019966,
        'learning_rate': 0.21037194404971513,
        'max_depth': 9,
        'n_estimators': 487,
        'reg_alpha': 0.837710105907328,
        'reg_lambda': 1.8513802340785614,
        'subsample': 0.8940864476963089,
    },
    'lightgbm': {
        'colsample_bytree': 0.794696861183782,
        'learning_rate': 0.27276864843838067,
        'max_depth': 13,
        'min_child_samples': 30,
        'n_estimators': 435,
        'num_leaves': 116,
        'subsample': 0.8307615538505436,
    },
    # 'random_forest': preenchido após o tuning de amanhã
}