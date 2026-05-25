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

TARGET = 'valor_declarado'

# Colunas que NUNCA entram no modelo, com o motivo de cada uma.
COLUNAS_EXCLUIR_MODELO = [
    # --- Metadados (não preditivos) ---
    'id', 'endereco', 'data_transacao',

    # --- Identificadores categóricos string ---
    # 'bairro' vira features numéricas via ibge_features / loo; 'cep' idem.
    'bairro', 'cep',

    # --- Categóricas string ainda não codificadas ---
    # padronize: ou vira one-hot/ordinal, ou sai. Por ora, sai.
    'padrao_acabamento', 'tipo_construtivo', 'tipo_ocupacao', 'zona_uso',
    'grupo_bairro',

    # --- LEAKAGE: derivam do target ---
    'valor_base_calculo',    # base de cálculo do ITBI ~= valor_declarado
    'preco_m2',              # = valor_declarado / area
    'preco_m2_x_idade', 'densidade_x_preco',
    'preco_medio_bairro', 'preco_medio_bairro_ano', 'preco_relativo_bairro',

    # --- Splitters temporais: usados para dividir, não para prever ---
    'ano_transacao', 'mes_transacao',
]