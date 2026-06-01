from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED = BASE_DIR / 'data' / 'processed'
DATA_FINAL = BASE_DIR / 'data' / 'final'
DATA_EXTERNAL = BASE_DIR / 'data' / 'external'

OUTPUTS_MODELS = BASE_DIR / 'outputs' / 'models'
OUTPUTS_FIGURES = BASE_DIR / 'outputs' / 'figures'
OUTPUTS_TABLES = BASE_DIR / 'outputs' / 'tables'

OUTPUTS_TABLES_TRAIN2 = BASE_DIR / 'outputs' / 'tables' / 'treino_ate_2023'
OUTPUTS_MODELS_TRAIN2 = BASE_DIR / 'outputs' / 'models' / 'treino_ate_2023'

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

    "salario_medio_sm", "faixa_salarial", "grupo_bairro_num",
    "inicio_ano", "fim_de_ano", "imovel_novo", "valorizacao_bairro_3anos",
    # string crua (entra como zona_uso_te):
    "zona_uso", "provavel_teardown", "area_km2_bairro", "populacao_bairro", "domicilios_bairro", "trimestre", "fator_construcao", "depreciacao_estimada",
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
    "xgboost": {
        "colsample_bytree": 0.6321296730347047,
        "gamma": 0.04707849413428006,
        "learning_rate": 0.05359786331478119,
        "max_depth": 9,
        "min_child_weight": 56,
        "reg_alpha": 0.5259908096680156,
        "reg_lambda": 1.1544499164857522,
        "subsample": 0.8671775480513688,
        "n_estimators": 4000,   # o train.py sobrescreve com N_ESTIMATORS_MAX
    },
    "lightgbm": {
        "colsample_bytree": 0.694791287941726,
        "learning_rate": 0.07145599889878906,
        "max_depth": 10,
        "min_child_samples": 109,
        "num_leaves": 109,
        "reg_alpha": 2.887945244033775,
        "reg_lambda": 7.523821084609852,
        "subsample": 0.8435909721327942,
        "subsample_freq": 1,
        "n_estimators": 4000,
    },
    "random_forest": {
        "max_depth": 18,
        "max_features": 0.5,
        "min_samples_leaf": 10,
        "min_samples_split": 20,
        "n_estimators": 300,
    },
}

# Classe 1: Popular (< 5 SM)
BAIRROS_CLASSE_1 = [
    'AARAO REIS', 'ALTO DOS PINHEIROS', 'ALTO PARAISO', 'ALVARO CAMARGOS',
    'ALVORADA', 'ANA LUCIA', 'APARECIDA', 'APARECIDA 7A SECAO',
    'BAIRRO DAS INDUSTRIAS', 'BALEIA', 'BARREIRO', 'BARREIRO DE CIMA',
    'BETANIA', 'BOA VISTA', 'BOM JESUS', 'BOM SUCESSO', 'BONFIM',
    'BRASIL INDUSTRIAL', 'CABANA', 'CACHOEIRINHA', 'CAETANO FURQUIM',
    'CALIFORNIA', 'CAMARGOS', 'CAMPO ALEGRE', 'CANAA', 'CANDELARIA',
    'CAPITAO EDUARDO', 'CARDOSO', 'CASA BRANCA', 'CEU AZUL', 'CONFISCO',
    'CONJ.HAB.', 'CONJ.ATILA DE PAIVA', 'CONJ.JOAO PAULO II', 'COPACABANA',
    'COQUEIROS', 'DIAMANTE', 'DOM BOSCO', 'DOM JOAQUIM', 'DOM SILVERIO',
    'DURVAL DE BARROS', 'ENGENHO NOGUEIRA', 'ERMELINDA', 'ESTRELA DALVA',
    'ETELVINA CARNEIRO', 'EYMARD', 'FERNAO DIAS', 'FLAVIO MARQUES LISBOA',
    'FLORAMAR', 'FREI EUSTAQUIO', 'FREI LEOPOLDO', 'GAMELEIRA', 'GLALIJA',
    'GLORIA', 'GOIANIA', 'GORDURAS', 'GOV.BENEDITO VALADARES', 'GUARANI',
    'HAVAI', 'HELIOPOLIS', 'IAPI', 'INCONFIDENCIA', 'INDEPENDENCIA',
    'INDUSTRIAL RODRIGUES CUNHA', 'IPANEMA', 'IPIRANGA', 'JAQUELINE',
    'JARDIM ALVORADA', 'JARDIM COMERCIARIOS', 'JARDIM EUROPA',
    'JARDIM FELICIDADE', 'JARDIM FILADELFIA', 'JARDIM GUANABARA',
    'JARDIM MONTANHES', 'JARDIM VITORIA', 'JARDINOPOLIS', 'JATOBA',
    'JULIANA', 'LAGOA', 'LEBLON', 'LETICIA', 'LINDEIA', 'MADRE GERTRUDES',
    'MAGNESITA', 'MANTIQUEIRA', 'MARAJO', 'MARIA GORETE', 'MARIA HELENA',
    'MARIA VIRGINIA', 'MARIZE', 'MILIONARIOS', 'MINAS CAIXA', 'MINASLANDIA',
    'MORRO DO PAPAGAIO', 'NAZARE', 'NOVA AMERICA', 'NOVA BARROCA',
    'NOVA CACHOEIRINHA', 'NOVA CINTRA', 'NOVA ESPERANCA', 'NOVA GAMELEIRA',
    'NOVA PAMPULHA', 'NOVA VISTA', 'OLARIA', "OLHOS D'AGUA", 'PALMEIRAS',
    'PARAISO', 'PARQUE RIACHUELO', 'PATROCINIO', 'PAULO VI', 'PEDREIRA',
    'PRADO LOPES', 'PINDORAMA', 'PIRAJA', 'PIRATININGA', 'PONGELUPE',
    'PRACA XII', 'PRIMAVERA', 'PRIMEIRO DE MAIO', 'PROVIDENCIA', 'REGINA',
    'RIBEIRO DE ABREU', 'RIO BRANCO', 'S.J.BATISTA VN', 'SALGADO FILHO',
    'SANTA CRUZ', 'SANTA HELENA', 'SANTA MONICA', 'SANTA TEREZINHA',
    'SANTO ANDRE', 'SAO BERNARDO', 'SAO CRISTOVAO', 'SAO GABRIEL',
    'SAO GERALDO', 'SAO JOAO BATISTA', 'SAO MARCOS', 'SAO PAULO',
    'SAO PEDRO VN', 'SAO SALVADOR', 'SAO THOMAS', 'SARAMENHA', 'SARANDI',
    'SAUDADE', 'SERRA DO CURRAL', 'SERRA VERDE', 'SERRANO', 'SINIMBU',
    'SOLIMOES', 'SUMARE', 'SUZANA', 'TAQUARIL', 'TEIXEIRA DIAS', 'TIROL',
    'TREVO', 'TUPI', 'UNIAO', 'UNIVERSITARIO', 'URUCUIA', 'VALE DO JATOBA',
    'VERA CRUZ', 'VILA BRASILIA', 'VILA CAFEZAL', 'VILA CEMIG',
    'VILA MAGNESITA', 'VILA OESTE', 'VILA VIRGINIA', 'VISTA ALEGRE',
    'WASHINGTON PIRES', 'XANGRILA', 'ZONA RURAL'
]

# Classe 2: Médio (5-8.5 SM)
BAIRROS_CLASSE_2 = [
    'ALIPIO DE MELO', 'BAIRRO DA GRACA', 'BARREIRO DE BAIXO', 'BRAUNAS',
    'CAICARA', 'CAICARA ADELAIDE', 'CALAFATE', 'CARLOS PRATES', 'CONCORDIA',
    'CONJUNTO CALIFORNIA I', 'CONJUNTO CALIFORNIA II',
    'CONJUNTO CELSO MACHADO', 'CONJUNTO ITACOLOMI', 'DOM CABRAL',
    'ESPLANADA', 'GARCAS', 'HORTO', 'INSTITUTO AGRONOMICO', 'JARDIM AMERICA',
    'JOAO PINHEIRO', 'LAGOINHA', 'MINAS BRASIL', 'MONSENHOR MESSIAS',
    'NOVA FLORESTA', 'NOVA GRANADA', 'NOVA SUICA', 'PADRE EUSTAQUIO',
    'PALMARES', 'PAQUETA', 'PEDRO II', 'PLANALTO', 'POMPEIA', 'RENASCENCA',
    'SAGRADA FAMILIA', 'SANTA EFIGENIA', 'SANTA INES', 'SANTA MARIA',
    'SAO FRANCISCO', 'SILVEIRA', 'UFMG', 'VENDA NOVA', 'VILA CLORIS'
]

# Classe 3: Alto (8.5-14.5 SM)
BAIRROS_CLASSE_3 = [
    'ALTO BARROCA', 'ALTO DOS CAICARAS', 'BAIRRO DAS MANSOES', 'BARRO PRETO',
    'BARROCA', 'BURITIS', 'CASTELO', 'CENTRO', 'CIDADE NOVA', 'AEROPORTO',
    'COLEGIO BATISTA', 'CORACAO EUCARISTICO', 'DONA CLARA', 'ESTORIL',
    'FLORESTA', 'GRAJAU', 'ITAPOA', 'JARAGUA', 'JARDIM ATLANTICO',
    'LIBERDADE', 'NOVO SAO LUCAS', 'OURO PRETO', 'PAMPULHA', 'PRADO',
    'SANTA AMELIA', 'SANTA BRANCA', 'SANTA ROSA', 'SANTA TEREZA', 'SAO LUCAS'
]

# Classe 4: Luxo (>14.5 SM)
BAIRROS_CLASSE_4 = [
    'SAO JOSE', 'SAO LUIS', 'SAO PEDRO', 'SAVASSI', 'SERRA', 'SION',
    'VILA PARIS', 'SANTO ANTONIO', 'SAO BENTO', 'PARQUE DAS MANGABEIRAS',
    'SANTA LUCIA', 'SANTO AGOSTINHO', 'LOURDES', 'LUXEMBURGO', 'MANGABEIRAS',
    'ANCHIETA', 'BANDEIRANTES', 'CARMO', 'BELVEDERE', 'CIDADE JARDIM',
    'CORACAO DE JESUS', 'CRUZEIRO', 'FUNCIONARIOS', 'GUTIERREZ'
]

# Mapa bairro -> número da classe (1=Popular, 2=Médio, 3=Alto, 4=Luxo).
# Bairros não listados caem na classe 2 (Médio) por padrão — mesma regra
# que o ibge_features.py já usava.
GRUPO_BAIRRO_PADRAO = 2

BAIRRO_PARA_CLASSE = {}
for _classe, _lista in [(1, BAIRROS_CLASSE_1), (2, BAIRROS_CLASSE_2),
                        (3, BAIRROS_CLASSE_3), (4, BAIRROS_CLASSE_4)]:
    for _b in _lista:
        BAIRRO_PARA_CLASSE[_b] = _classe

# ----------------------------------------------------------------------------
# PISO DE PREÇO/M² POR CLASSE DE BAIRRO (reais de dez/2024, deflacionado)
# ----------------------------------------------------------------------------
# Rede de segurança ABSOLUTA contra subdeclaração, complementar ao filtro
# IQR por bairro+tipo. Um imóvel cujo preço/m² está abaixo do piso da sua
# classe é declaração parcial/fraudulenta, independentemente da distribuição
# do bairro (critério imune à contaminação do IQR pelos próprios outliers).
# Valores conservadores: bem abaixo da mediana real de cada classe, para
# remover apenas o que é claramente subdeclaração. CALIBRAR olhando a
# amostra de remoções.
PISO_PRECO_M2_POR_CLASSE = {
    1: 800,  # Popular
    2: 1_200,  # Médio
    3: 1_800,  # Alto
    4: 2_500,  # Luxo
}

import numpy as np  # se já não estiver importado no topo

import pandas as pd
DATA_INICIO_TRANSICAO = pd.Timestamp("2022-01-01")
DATA_REGIME_NOVO = pd.Timestamp("2023-01-01")
DATA_MUDANCA_REGIME = DATA_REGIME_NOVO  # alias p/ o script de diagnóstico


def classificar_regime(datas):
    d = pd.to_datetime(datas)
    cond = [d < DATA_INICIO_TRANSICAO, d < DATA_REGIME_NOVO]
    return np.select(cond, ["antigo", "transicao"], default="novo")


# ─── ITEM 9: piso de erro grosseiro no regime novo ──────────────────────────
# No regime novo a base É o valor declarado; não filtramos subdeclaração (seria
# remover o próprio alvo). Só cortamos erro óbvio.
PISO_PRECO_M2_ERRO_GROSSEIRO = 300

# ─── ITEM 5: restrições de monotonicidade (feature -> sinal) ────────────────
MONOTONIC_CONSTRAINTS = {
    "area_construida_m2": 1, "area_total_m2": 1, "area_terreno_m2": 1,
    "padrao_acabamento_num": 1, "idade_imovel": -1,
    "preco_medio_bairro_loo": 1, "preco_m2_medio_bairro_loo": 1,
}

# ─── ITEM 3: zona_uso entra via target encoding temporal (numérica) ─────────
# A coluna string 'zona_uso' continua excluída; a versão numérica 'zona_uso_te'
# (criada no derived_features) é que entra no modelo.

# ─── ITEM 5/6: toggles ──────────────────────────────────────────────────────
PREVER_PRECO_M2 = False          # True -> modela log(R$/m²) e multiplica por área
TREINAR_APENAS_REGIME_NOVO = False  # True -> split temporal DENTRO do regime novo

# ─── ITEM 8: banda de previsão ──────────────────────────────────────────────
QUANTIS = [0.1, 0.5, 0.9]
NIVEL_CONFORMAL = 0.80


