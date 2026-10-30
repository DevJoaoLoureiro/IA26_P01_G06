"""Modelo do problema (Problem, Request, Vehicle, State), leitura das
instâncias a partir dos CSV, função de custo e verificador de viabilidade.

As classes e funções estão implementadas em algoritmos.py; este módulo reúne
a parte de MODELAÇÃO numa interface simples, como pedido no enunciado.

Representação de uma solução: uma lista de operações por veículo, cada uma
("recolha", artigo), ("armazem", 0) ou ("entrega", artigo). O regresso ao
centro é acrescentado automaticamente no fim.
"""
from algoritmos import (Vehicle, Request, Problem, State, FleetProblem,
                        FleetResult, load_problem, load_fleet_problem,
                        RouteEvaluator, evaluate_fleet_sequences,
                        validar_rota_frota)

__all__ = ["Vehicle", "Request", "Problem", "State", "FleetProblem",
           "FleetResult", "load_problem", "load_fleet_problem",
           "custo", "viavel", "simular"]


def custo(problem, rotas):
    """Distância total (km) da solução, incluindo o regresso ao centro.

    Devolve None se alguma rota for inviável.
    """
    return RouteEvaluator(problem).total(rotas)


def viavel(problem, rotas, todos_os_artigos=True):
    """True se as rotas respeitam janelas, capacidade, precedência
    (recolha -> armazém -> entrega), tempos de serviço e horizonte.

    Com todos_os_artigos=True exige também que todos os artigos sejam entregues.
    """
    evaluator = RouteEvaluator(problem)
    if any(evaluator.evaluate(vi, ops) is None for vi, ops in enumerate(rotas)):
        return False
    if todos_os_artigos:
        entregues = {item for ops in rotas for kind, item in ops if kind == "entrega"}
        return entregues == {r.item_id for r in problem.requests}
    return True


def simular(problem, rotas):
    """Simula a solução passo a passo e devolve um FleetResult com os horários
    de início de cada serviço, ou None se for inviável ou incompleta."""
    return evaluate_fleet_sequences(problem, rotas)
