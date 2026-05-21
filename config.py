from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED = BASE_DIR / 'data' / 'processed'
DATA_FINAL = BASE_DIR / 'data' / 'final'

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