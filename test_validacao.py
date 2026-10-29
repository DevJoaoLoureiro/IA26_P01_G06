"""Testes de regressão para a frota: executar python test_validacao.py."""
from pathlib import Path
from algoritmos import load_fleet_problem, fleet_greedy, fleet_astar, fleet_csp, validar_rota_frota

folder = Path(__file__).parent / "data"
base = load_fleet_problem(folder)
assert len(base.vehicles) == 2 and len(base.requests) == 80
for n in (3, 5):
    p = load_fleet_problem(folder)
    p.requests = p.requests[:n]
    for result in (fleet_greedy(p, 2), fleet_astar(p, 1, 1500), fleet_astar(p, 2, 1500), fleet_csp(p, 10000)):
        if result.feasible:
            valid, errors = validar_rota_frota(p, result)
            assert valid, (n, result.algorithm, errors)
            assert result.delivered == n
        else:
            assert result.delivered <= n
print("Testes de regressão concluídos sem falhas.")
