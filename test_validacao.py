"""Testes do modelo e dos algoritmos: executar com  python test_validacao.py

1. Rotas escritas à mão (viáveis e inviáveis) para testar o modelo.
2. Algoritmos de procura em subconjuntos pequenos, validados.
3. Algoritmos para a instância completa (80 artigos), validados pelo
   validador independente (validador.py), que lê os CSV diretamente.
"""
from dataclasses import replace
from pathlib import Path

import pandas as pd

import algoritmos as A
import model
import validador

PASTA = Path(__file__).parent / "data"


def problema(n=None):
    p = A.load_fleet_problem(PASTA)
    if n is not None:
        p.requests = p.requests[:n]
    return p


def so_artigos(p, ids):
    p.requests = [r for r in p.requests if r.item_id in ids]
    return p


# ---------------------------------------------------------------- 1. modelo
def teste_rotas_manuais():
    # Artigos 17 e 60: ambos do produtor 4 (no mesmo sítio do centro) e com
    # entrega no Restaurante 3 (location 10, a 18,16 km do centro).
    p = so_artigos(problema(), {17, 60})
    rota = [("recolha", 17), ("recolha", 60), ("armazem", 0),
            ("entrega", 17), ("entrega", 60)]
    # Distância esperada calculada à mão: centro->produtor 4 = 0 km,
    # armazém->location 10 = 18,16 km, regresso = 18,16 km.
    assert model.viavel(p, [rota, []])
    assert abs(model.custo(p, [rota, []]) - 36.32) < 1e-6
    sim = model.simular(p, [rota, []])
    horas = [pd.Timestamp(op[3]).strftime("%H:%M") for op in sim.routes[1]]
    # 17 recolhido às 06:45; 60 só fica disponível às 07:15 (espera e novo
    # serviço); armazém às 07:25; entregas juntas às 11:15 (mesma paragem).
    assert horas[:5] == ["06:45", "07:15", "07:25", "11:15", "11:15"], horas

    # Entrega sem passar pelo armazém -> inviável.
    assert not model.viavel(p, [[("recolha", 17), ("entrega", 17)], []],
                            todos_os_artigos=False)
    # Entrega antes da recolha -> inviável.
    assert not model.viavel(p, [[("armazem", 0), ("entrega", 17)], []],
                            todos_os_artigos=False)
    # Artigo esquecido -> solução incompleta.
    assert not model.viavel(p, [rota[:1] + rota[2:4], []])
    # Capacidade: com 3 kg, 17 (2,1 kg) + 60 (2,6 kg) não cabem juntos.
    pequeno = so_artigos(problema(), {17, 60})
    pequeno.vehicles = [replace(v, capacity_kg=3.0) for v in pequeno.vehicles]
    assert not model.viavel(pequeno, [rota, []])
    # Duas viagens no mesmo veículo também falham: a segunda recolha (60)
    # só seria possível depois das 11:25, e a janela fecha às 09:00.
    duas = [("recolha", 17), ("armazem", 0), ("entrega", 17),
            ("recolha", 60), ("armazem", 0), ("entrega", 60)]
    assert not model.viavel(pequeno, [duas, []])
    # Um artigo em cada veículo: viável.
    assert model.viavel(pequeno, [duas[:3], duas[3:]])
    # Janela: 2 (produtor 6, 08:15) -> 1 (produtor 3) -> 60 (produtor 4)
    # chega depois das 09:00, fim da janela de recolha do artigo 60.
    p3 = so_artigos(problema(), {1, 2, 60})
    tarde = [("recolha", 2), ("recolha", 1), ("recolha", 60)]
    assert not model.viavel(p3, [tarde, []], todos_os_artigos=False)


# ------------------------------------------- 2. procura em instâncias pequenas
def teste_procura_pequena():
    for n in (3, 5):
        p = problema(n)
        ids = {r.item_id for r in p.requests}
        resultados = [A.fleet_greedy(p, 1), A.fleet_greedy(p, 2),
                      A.fleet_gbfs(p, 1, 20000, True), A.fleet_gbfs(p, 2, 20000, True),
                      A.fleet_astar(p, 1, 20000, True), A.fleet_astar(p, 2, 20000, True)]
        for r in resultados:
            if r.feasible:
                ok, erros = A.validar_rota_frota(p, r)
                assert ok, (n, r.algorithm, erros)
                ok, erros, km = validador.validar(PASTA, r.routes, r.distance_km, ids)
                assert ok, (n, r.algorithm, erros)
        # O A* com h=0 (custo uniforme) é ótimo: nenhum método o pode bater.
        otimo = resultados[4]
        assert otimo.feasible
        for r in resultados:
            if r.feasible:
                assert r.distance_km >= otimo.distance_km - 1e-6


# ---------------------------------------------- 3. instância completa (80)
def teste_instancia_completa():
    p = problema()
    assert len(p.vehicles) == 2 and len(p.requests) == 80
    g = A.fleet_greedy_repair(p, 1)
    c = A.fleet_csp(p)
    h = A.fleet_hill_climbing(p, g, max_iterations=3)
    for r in (g, c, h):
        assert r.feasible and r.delivered == 80, (r.algorithm, r.message)
        ok, erros, km = validador.validar(PASTA, r.routes, r.distance_km)
        assert ok, (r.algorithm, erros[:5])
    assert h.distance_km <= g.distance_km + 1e-6
    assert all(b <= a + 1e-9 for a, b in zip(h.history, h.history[1:]))


if __name__ == "__main__":
    for teste in (teste_rotas_manuais, teste_procura_pequena, teste_instancia_completa):
        teste()
        print(f"OK  {teste.__name__}")
    print("Testes concluídos sem falhas.")