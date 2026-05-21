from src.sistema_decisao.recomendador import RecomendadorImoveis

print("Carregando sistema de recomendação...\n")
recomendador = RecomendadorImoveis()

print("\n" + "█" * 80)
print("CASO 1: Apartamento no Buritis - Parado há 7 meses")
print("█" * 80)

imovel_caso1 = {
    'ano_construcao': 2010,
    'area_terreno_m2': 250.0,
    'area_construida_m2': 120.0,
    'area_total_m2': 150.0,
    'fracao_ideal': 0.005,
    'ano_transacao': 2024,
    'mes_transacao': 5,

    'idade_imovel': 14,
    'is_residencial': 1,

    'razao_area_util': 0.48,
    'densidade_construcao': 0.60,
    'area_nao_construida_m2': 30.0,

    'valorizacao_bairro_3anos': 5.2,

    'imovel_novo': 0,
    'depreciacao_estimada': 0.14,

    'area_x_idade': 2100.0,
    'novo_em_bairro_caro': 0,

    'std_preco_bairro': 125000.0,
    'num_transacoes_bairro': 3250,
    'preco_max_bairro': 1500000.0,
    'preco_min_bairro': 180000.0,
    'range_preco_bairro': 1320000.0
}

analise1 = recomendador.analisar_imovel(
    dados_imovel=imovel_caso1,
    preco_pedido=650000,  # vendedor pede R$ 650k
    tempo_parado_dias=210  # parado há 7 meses
)

recomendador.gerar_relatorio(analise1)

print("\n\n" + "█" * 80)
print("CASO 2: Casa na Serra - Preço muito acima do mercado")
print("█" * 80)

imovel_caso2 = {
    'ano_construcao': 2015,
    'area_terreno_m2': 450.0,
    'area_construida_m2': 280.0,
    'area_total_m2': 320.0,
    'fracao_ideal': 0.01,
    'ano_transacao': 2024,
    'mes_transacao': 5,

    'idade_imovel': 9,
    'is_residencial': 1,

    'razao_area_util': 0.62,
    'densidade_construcao': 0.71,
    'area_nao_construida_m2': 40.0,

    'valorizacao_bairro_3anos': 12.5,

    'imovel_novo': 0,
    'depreciacao_estimada': 0.09,

    'area_x_idade': 2880.0,
    'novo_em_bairro_caro': 0,

    'std_preco_bairro': 350000.0,
    'num_transacoes_bairro': 1250,
    'preco_max_bairro': 4500000.0,
    'preco_min_bairro': 850000.0,
    'range_preco_bairro': 3650000.0
}

analise2 = recomendador.analisar_imovel(
    dados_imovel=imovel_caso2,
    preco_pedido=2500000,  # vendedor pede R$ 2.5M
    tempo_parado_dias=45  # parado há 1.5 meses
)

recomendador.gerar_relatorio(analise2)

print("\n" + "█" * 80)
print("TESTES CONCLUÍDOS")
print("█" * 80)