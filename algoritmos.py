"""Projeto IA26_P01_G18 — algoritmos de rotas (datasets data_tiny e data_small).

Modelo: recolha -> centro -> entrega, com capacidade e retorno ao centro.
CSP e A* respeitam as janelas temporais na procura da frota.
A data da encomenda é considerada a primeira data permitida para recolha.
Inclui as abordagens para um veículo e para frota de dois veículos.
Os algoritmos de pesquisa usam limites e não garantem otimalidade global.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import heapq
import itertools
import math
import time
import pandas as pd

@dataclass(frozen=True)
class Vehicle:
    vehicle_id: int
    capacity_kg: float
    center_location: int

@dataclass(frozen=True)
class Request:
    item_id: int
    order_id: int
    pickup: int
    delivery: int
    weight: float
    ready: pd.Timestamp
    pickup_end: pd.Timestamp
    delivery_start: pd.Timestamp
    delivery_end: pd.Timestamp

@dataclass
class Problem:
    vehicle: Vehicle
    requests: list[Request]
    distance: dict[tuple[int,int],float]
    duration: dict[tuple[int,int],float]
    start_time: pd.Timestamp
    rejected: list[tuple[int,str]] = field(default_factory=list)
    stop_service_min: float = 0.0
    warehouse_handling_min: float = 0.0
    horizon_end: pd.Timestamp | None = None

@dataclass
class State:
    location: int
    moment: pd.Timestamp
    load: float = 0.0
    picked: frozenset[int] = frozenset()
    delivered: frozenset[int] = frozenset()
    route: tuple[tuple[str,int,int,str],...] = ()
    km: float = 0.0
    warehouse_passed: frozenset[int] = frozenset()

@dataclass
class Result:
    algorithm: str
    feasible: bool
    distance_km: float
    expanded: int
    runtime_s: float
    route: list[tuple[str,int,int,str]]
    delivered: int
    total: int
    message: str = ""


def load_problem(folder="data", vehicle_id=1, start_time=None):
    """Lê os 7 CSV necessários, mesmo que os outros estejam presentes."""
    folder = Path(folder)
    names = ["orders", "order_items", "products", "producers", "vehicles",
             "locations", "distance_matrix"]
    data = {name: pd.read_csv(folder / f"{name}.csv") for name in names}
    rules = {}
    rules_file = folder / "rules.csv"
    if rules_file.exists():
        rules = dict(pd.read_csv(rules_file).loc[:, ["rule", "value"]].values)
    stop_min = float(rules.get("stop_service_min", 0))
    handling_min = float(rules.get("warehouse_handling_min", 0))
    # O descarregamento e carregamento ocorrem na mesma visita ao centro.
    # Por isso, a permanência em armazém é 0h (<= max_warehouse_dwell_h).
    if rules and rules.get("flow") != "producer>warehouse>delivery_location":
        raise ValueError("Fluxo logístico desconhecido em rules.csv")
    horizon_end = None
    if "start_date" in rules and "horizon_days" in rules:
        horizon_end = pd.Timestamp(rules["start_date"]) + pd.Timedelta(days=int(rules["horizon_days"]))
    locs = data["locations"]
    def locate(kind, ref):
        rows = locs[(locs.location_type == kind) & (locs.ref_id == ref)]
        if len(rows) != 1:
            raise ValueError(f"Localização não única: {kind}, {ref}")
        return int(rows.iloc[0].location_id)
    vrow = data["vehicles"].set_index("vehicle_id").loc[vehicle_id]
    vehicle = Vehicle(int(vehicle_id), float(vrow.capacity_kg),
                      locate("collection_center", int(vrow.center_id)))
    orders = data["orders"].set_index("order_id")
    products = data["products"].set_index("product_id")
    requests = []
    rejected = []
    for r in data["order_items"].itertuples(index=False):
        order = orders.loc[r.order_id]
        product = products.loc[r.product_id]
        placed = pd.Timestamp(order.order_datetime)
        available = pd.Timestamp(r.availability_start)
        ready = max(placed, available)
        pickup_end = pd.Timestamp(r.availability_end)
        delivery_start = pd.Timestamp(r.delivery_start)
        delivery_end = pd.Timestamp(r.delivery_end)
        weight = float(r.weight_kg) if "weight_kg" in r._fields else float(r.quantity) * float(product.unit_weight_kg)
        item_id = int(r.order_item_id)
        # Seleção comum de artigos temporalmente elegíveis para comparação justa.
        # A frota usa as mesmas janelas temporais em todos os métodos.
        if pickup_end < ready:
            rejected.append((item_id, "janela de recolha impossível"))
            continue
        if delivery_end < ready or delivery_end < delivery_start:
            rejected.append((item_id, "janela de entrega impossível"))
            continue
        if weight > vehicle.capacity_kg:
            rejected.append((item_id, "peso excede capacidade do veículo"))
            continue
        requests.append(Request(item_id, int(r.order_id),
            locate("producer", int(product.producer_id)),
            locate("delivery_location", int(order.delivery_location_id)),
            weight, ready, pickup_end, delivery_start, delivery_end))
    dist = {(int(r.origin_location_id),int(r.destination_location_id)):float(r.distance_km)
            for r in data["distance_matrix"].itertuples(index=False)}
    dur = {(int(r.origin_location_id),int(r.destination_location_id)):float(r.duration_min)
           for r in data["distance_matrix"].itertuples(index=False)}
    if start_time is None:
        start_time = min((r.ready for r in requests), default=pd.Timestamp.now())
    return Problem(vehicle, requests, dist, dur, pd.Timestamp(start_time), rejected, stop_min, handling_min, horizon_end)


def travel(problem, origin, destination):
    if origin == destination:
        return 0.0, 0.0
    return problem.distance[(origin,destination)], problem.duration[(origin,destination)]


def initial(problem):
    return State(problem.vehicle.center_location, problem.start_time)


def _service_end(problem, service_start, kind, check_time_windows):
    """Regista o início de serviço na rota; atualiza o relógio após o serviço."""
    minutes = (problem.warehouse_handling_min if kind == "armazem"
               else problem.stop_service_min)
    end = service_start + pd.Timedelta(minutes=minutes)
    if check_time_windows and problem.horizon_end is not None and end > problem.horizon_end:
        return None
    return end


def actions(problem, state, check_time_windows=True):
    """Uma visita ao armazém descarrega/recarrega todos os artigos a bordo.

    Modelo sem armazenamento entre visitas: permanência de cada artigo no
    centro = 0 horas. A carga mantém-se a bordo no estado agregado.
    A hora na rota é o INÍCIO do serviço (regra window_semantics).
    """
    center = problem.vehicle.center_location
    pending = state.picked - state.warehouse_passed
    if pending:
        km, minutes = travel(problem, state.location, center)
        arrival = state.moment + pd.Timedelta(minutes=minutes)
        end = _service_end(problem, arrival, "armazem", check_time_windows)
        if end is not None:
            yield State(center, end, state.load, state.picked, state.delivered,
                        state.route + (("armazem", 0, center, str(arrival)),),
                        state.km + km, state.warehouse_passed | pending)
    for req in problem.requests:
        item = req.item_id
        if item not in state.picked:
            if state.load + req.weight > problem.vehicle.capacity_kg + 1e-9:
                continue
            loc, kind, delta, earliest, latest = req.pickup, "recolha", req.weight, req.ready, req.pickup_end
        elif item in state.warehouse_passed and item not in state.delivered:
            loc, kind, delta, earliest, latest = req.delivery, "entrega", -req.weight, req.delivery_start, req.delivery_end
        else:
            continue
        km, minutes = travel(problem, state.location, loc)
        arrival = state.moment + pd.Timedelta(minutes=minutes)
        if check_time_windows:
            arrival = max(arrival, earliest)
            if arrival > latest:
                continue
        end = _service_end(problem, arrival, kind, check_time_windows)
        if end is None:
            continue
        yield State(loc, end, max(0., state.load + delta),
                    state.picked | ({item} if kind == "recolha" else set()),
                    state.delivered | ({item} if kind == "entrega" else set()),
                    state.route + ((kind, item, loc, str(arrival)),),
                    state.km + km, state.warehouse_passed)


def finish(problem, state):
    km, minutes = travel(problem, state.location, problem.vehicle.center_location)
    arrival = state.moment + pd.Timedelta(minutes=minutes)
    return State(problem.vehicle.center_location, arrival, state.load,
                 state.picked, state.delivered,
                 state.route + (("regresso", 0, problem.vehicle.center_location, str(arrival)),),
                 state.km + km, state.warehouse_passed)


def heuristic(problem, state, variant=1):
    """Limites inferiores em km; H1=0, H2=distância mínima à próxima ação.

    O custo de regressar ao centro é incluído quando não restam artigos.
    """
    if variant not in (1, 2):
        raise ValueError("heuristic_variant deve ser 1 ou 2")
    if variant == 1:
        return 0.0
    targets = []
    if state.picked - state.warehouse_passed:
        targets.append(problem.vehicle.center_location)
    for req in problem.requests:
        if req.item_id not in state.picked:
            targets.append(req.pickup)
        elif req.item_id in state.warehouse_passed and req.item_id not in state.delivered:
            targets.append(req.delivery)
    if not targets:
        return travel(problem, state.location, problem.vehicle.center_location)[0]
    return min(travel(problem, state.location, loc)[0] for loc in targets)


def search(problem, method="astar", heuristic_variant=1, max_expansions=50000):
    """A* e GBFS de um veículo sem restrições horárias; custo em km."""
    start = time.perf_counter()
    counter = itertools.count()
    root = initial(problem)
    heap = [(heuristic(problem, root, heuristic_variant), next(counter), root)]
    expanded = 0
    best = {}
    n = len(problem.requests)
    while heap and expanded < max_expansions:
        _, _, state = heapq.heappop(heap)
        if len(state.delivered) == n:
            done = finish(problem, state)
            return Result(method, True, done.km, expanded, time.perf_counter()-start,
                          list(done.route), n, n)
        key = (state.location, state.picked, state.warehouse_passed,
               state.delivered, round(state.load, 5))
        if best.get(key, math.inf) <= state.km + 1e-9:
            continue
        best[key] = state.km
        expanded += 1
        for nxt in actions(problem, state, check_time_windows=False):
            h = heuristic(problem, nxt, heuristic_variant)
            priority = h if method == "greedy" else nxt.km + h
            heapq.heappush(heap, (priority, next(counter), nxt))
    return Result(method, False, math.inf, expanded, time.perf_counter()-start,
                  [], 0, n, "Sem solução completa ou limite de expansões atingido")


def greedy(problem, heuristic_variant=1, max_expansions=50000):
    return search(problem,"greedy",heuristic_variant,max_expansions)




def astar(problem, heuristic_variant=1, max_expansions=50000):
    return search(problem,"astar",heuristic_variant,max_expansions)


def evaluate_sequence(problem, sequence, check_time_windows=False):
    """Avalia uma sequência fixa de (recolha|armazem|entrega, artigo)."""
    state = initial(problem)
    for kind, item in sequence:
        options = actions(problem, state, check_time_windows=check_time_windows)
        found = next((s for s in options if s.route[-1][0] == kind and
                      s.route[-1][1] == item), None)
        if found is None:
            return None
        state = found
    if len(state.delivered) != len(problem.requests):
        return None
    return finish(problem, state)


def hill_climbing(problem, initial_result, max_iterations=200):
    """Melhoria local por trocas de duas operações, mantendo apenas rotas viáveis."""
    start = time.perf_counter()
    n = len(problem.requests)
    if not initial_result.feasible:
        return Result("hill_climbing",False,math.inf,0,time.perf_counter()-start,[],0,n,
                      "Requer solução inicial completa")
    sequence = [(kind,item) for kind,item,_,_ in initial_result.route if kind != "regresso"]
    current = evaluate_sequence(problem,sequence)
    iterations = 0
    while iterations < max_iterations:
        best_state = current
        best_sequence = sequence
        for i in range(len(sequence)):
            for j in range(i+1,len(sequence)):
                candidate = sequence.copy()
                candidate[i],candidate[j] = candidate[j],candidate[i]
                outcome = evaluate_sequence(problem,candidate)
                if outcome is not None and outcome.km < best_state.km - 1e-9:
                    best_state,best_sequence = outcome,candidate
        iterations += 1
        if best_sequence == sequence:
            break
        current,sequence = best_state,best_sequence
    return Result("hill_climbing",True,current.km,iterations,time.perf_counter()-start,
                  list(current.route),n,n)


def csp_backtracking(problem, max_nodes=50000):
    """CSP: variáveis = sequência de operações; domínios = ações ainda possíveis;
    restrições = precedência, capacidade e janelas. Retorna primeira solução."""
    start = time.perf_counter()
    nodes = 0
    backtracks = 0
    n = len(problem.requests)
    def dfs(state):
        nonlocal nodes,backtracks
        if len(state.delivered) == n:
            return finish(problem,state)
        if nodes >= max_nodes:
            return None
        nodes += 1
        choices = sorted(actions(problem,state),key=lambda s:(s.moment,s.km))
        for nxt in choices:
            answer = dfs(nxt)
            if answer is not None:
                return answer
        backtracks += 1
        return None
    answer = dfs(initial(problem))
    if answer is None:
        return Result("csp",False,math.inf,nodes,time.perf_counter()-start,
                      [],0,n,f"Sem solução; backtracks={backtracks}")
    return Result("csp",True,answer.km,nodes,time.perf_counter()-start,
                  list(answer.route),n,n,f"backtracks={backtracks}")


def summary(results):
    return pd.DataFrame([{"algoritmo":r.algorithm,"viavel":r.feasible,
        "distancia_km":r.distance_km if r.feasible else None,
        "nos_ou_iteracoes":r.expanded,"tempo_s":r.runtime_s,
        "artigos_entregues":r.delivered,"total_artigos":r.total,
        "observacoes":r.message} for r in results])

# ================== EXTENSÃO: FROTA DE DOIS VEÍCULOS ==================
# Mantém as funções anteriores para compatibilidade com os testes do notebook.

@dataclass
class FleetProblem:
    vehicles: list[Vehicle]
    requests: list[Request]
    distance: dict
    duration: dict
    start_time: pd.Timestamp
    rejected: list[tuple[int,str]] = field(default_factory=list)
    stop_service_min: float = 0.0
    warehouse_handling_min: float = 0.0
    horizon_end: pd.Timestamp | None = None

@dataclass
class FleetResult:
    algorithm: str
    feasible: bool
    distance_km: float
    expanded: int
    runtime_s: float
    routes: dict[int,list[tuple[str,int,int,str]]]
    delivered: int
    total: int
    message: str = ""


def load_fleet_problem(folder="data", start_time=None):
    """Carrega os dois veículos e todos os artigos temporalmente elegíveis.
    Não rejeita artigos em função da capacidade de apenas uma carrinha.
    """
    folder = Path(folder)
    dfv = pd.read_csv(folder / "vehicles.csv")
    base = load_problem(folder, vehicle_id=int(dfv.iloc[dfv.capacity_kg.argmax()].vehicle_id),
                        start_time=start_time)
    locs = pd.read_csv(folder / "locations.csv")
    vehicles = []
    for v in dfv.itertuples(index=False):
        rows = locs[(locs.location_type == "collection_center") & (locs.ref_id == v.center_id)]
        if len(rows) != 1:
            raise ValueError(f"Centro não encontrado para veículo {v.vehicle_id}")
        vehicles.append(Vehicle(int(v.vehicle_id), float(v.capacity_kg), int(rows.iloc[0].location_id)))
    return FleetProblem(vehicles, base.requests, base.distance, base.duration,
                        base.start_time, base.rejected, base.stop_service_min,
                        base.warehouse_handling_min, base.horizon_end)


def _fleet_candidates(problem, states, item_ids, check_time_windows=True):
    """Gera ações de dois veículos, com serviço, janelas e armazém."""
    lookup = {r.item_id: r for r in problem.requests}
    for vi, state in enumerate(states):
        vehicle = problem.vehicles[vi]
        pending = state.picked - state.warehouse_passed
        if pending:
            loc = vehicle.center_location
            km = 0. if state.location == loc else problem.distance[(state.location, loc)]
            minutes = 0. if state.location == loc else problem.duration[(state.location, loc)]
            arrival = state.moment + pd.Timedelta(minutes=minutes)
            end = _service_end(problem, arrival, "armazem", check_time_windows)
            if end is not None:
                nxt = State(loc, end, state.load, state.picked, state.delivered,
                            state.route + (("armazem", 0, loc, str(arrival)),),
                            state.km + km, state.warehouse_passed | pending)
                yield vi, nxt, km, "armazem", 0
        for item in sorted(item_ids):
            r = lookup[item]
            owner = next((i for i, s in enumerate(states) if item in s.picked), None)
            if owner is None:
                if state.load + r.weight > vehicle.capacity_kg + 1e-9:
                    continue
                loc, kind, delta = r.pickup, "recolha", r.weight
                earliest, latest = r.ready, r.pickup_end
            elif owner == vi and item in state.warehouse_passed and item not in state.delivered:
                loc, kind, delta = r.delivery, "entrega", -r.weight
                earliest, latest = r.delivery_start, r.delivery_end
            else:
                continue
            km = 0. if state.location == loc else problem.distance[(state.location, loc)]
            minutes = 0. if state.location == loc else problem.duration[(state.location, loc)]
            arrival = state.moment + pd.Timedelta(minutes=minutes)
            if check_time_windows:
                arrival = max(arrival, earliest)
                if arrival > latest:
                    continue
            end = _service_end(problem, arrival, kind, check_time_windows)
            if end is None:
                continue
            nxt = State(loc, end, max(0., state.load + delta),
                        state.picked | ({item} if kind == "recolha" else set()),
                        state.delivered | ({item} if kind == "entrega" else set()),
                        state.route + ((kind, item, loc, str(arrival)),),
                        state.km + km, state.warehouse_passed)
            yield vi, nxt, km, kind, item


def _fleet_complete(problem, states, delivered_count, expanded, start, algorithm, message=""):
    routes = {}
    distance = 0.0
    for vehicle,state in zip(problem.vehicles,states):
        km = 0.0 if state.location == vehicle.center_location else problem.distance[(state.location,vehicle.center_location)]
        minutes = 0.0 if state.location == vehicle.center_location else problem.duration[(state.location,vehicle.center_location)]
        arrival = state.moment + pd.Timedelta(minutes=minutes)
        routes[vehicle.vehicle_id] = list(state.route) + [("regresso",0,vehicle.center_location,str(arrival))]
        distance += state.km + km
    return FleetResult(algorithm,delivered_count==len(problem.requests),distance,expanded,
                       time.perf_counter()-start,routes,delivered_count,len(problem.requests),message)


def fleet_greedy(problem, heuristic_variant=1, max_steps=10000):
    """Constrói uma solução conjunta para os veículos, sem duplicar artigos.
    h1 privilegia deslocação curta; h2 privilegia próxima janela a fechar.
    O resultado pode ser parcial; feasible=True só se todas as entregas ocorrerem.
    """
    start = time.perf_counter()
    states = tuple(State(v.center_location,problem.start_time) for v in problem.vehicles)
    item_ids = {r.item_id for r in problem.requests}
    expanded = 0
    for _ in range(max_steps):
        delivered = set().union(*(s.delivered for s in states))
        if delivered == item_ids:
            return _fleet_complete(problem,states,len(delivered),expanded,start,"fleet_greedy")
        candidates = list(_fleet_candidates(problem,states,item_ids-delivered))
        expanded += 1
        if not candidates:
            return _fleet_complete(problem,states,len(delivered),expanded,start,"fleet_greedy",
                                   "Sem próximas ações válidas; solução parcial")
        if heuristic_variant == 1:
            selected = min(candidates,key=lambda c:(c[2],c[1].moment, c[0]))
        elif heuristic_variant == 2:
            lookup = {r.item_id:r for r in problem.requests}
            selected = min(candidates,key=lambda c:(
                (lookup[c[4]].pickup_end if c[3]=='recolha' else lookup[c[4]].delivery_end)-c[1].moment if c[3] != "armazem" else pd.Timedelta(hours=48),
                c[2],c[0]))
        else:
            raise ValueError("heuristic_variant deve ser 1 ou 2")
        vi,nxt,_,_,_ = selected
        states = tuple(nxt if j==vi else s for j,s in enumerate(states))
    delivered = set().union(*(s.delivered for s in states))
    return _fleet_complete(problem,states,len(delivered),expanded,start,"fleet_greedy","Limite de passos")


def fleet_gbfs(problem, heuristic_variant=1, max_expansions=5000):
    """Greedy Best-First Search para a frota, com fronteira de prioridades.

    A fronteira é ordenada APENAS por h(n), sem incluir g(n):
      H1: menor distância para a próxima operação viável (km);
      H2: menor folga temporal para a próxima operação viável (minutos).
    A função procura uma solução completa; pode atingir o limite de expansões.
    Nem H1 nem H2 garantem a menor distância global.
    """
    if heuristic_variant not in (1, 2):
        raise ValueError("heuristic_variant deve ser 1 ou 2")
    if max_expansions < 1:
        raise ValueError("max_expansions deve ser positivo")

    start = time.perf_counter()
    ids = frozenset(r.item_id for r in problem.requests)
    by_id = {r.item_id: r for r in problem.requests}
    root = tuple(State(v.center_location, problem.start_time)
                 for v in problem.vehicles)

    def delivered_ids(states):
        return frozenset().union(*(s.delivered for s in states))

    def h(states):
        pending = ids - delivered_ids(states)
        if not pending:
            return 0.0
        candidates = list(_fleet_candidates(problem, states, pending))
        if not candidates:
            return math.inf
        if heuristic_variant == 1:
            return min(km for _, _, km, _, _ in candidates)
        return min(
            ((by_id[item].pickup_end if kind == 'recolha'
              else by_id[item].delivery_end) - nxt.moment).total_seconds() / 60.0
            if kind != "armazem" else 0.0
            for _, nxt, _, kind, item in candidates
        )

    counter = itertools.count()
    frontier = [(h(root), next(counter), root)]
    visited = set()
    expanded = 0
    best_partial = root

    while frontier and expanded < max_expansions:
        _, _, states = heapq.heappop(frontier)
        completed = delivered_ids(states)
        if len(completed) > len(delivered_ids(best_partial)):
            best_partial = states
        if completed == ids:
            return _fleet_complete(problem, states, len(completed),
                                   expanded, start, "fleet_gbfs")
        key = tuple((s.location, s.picked, s.warehouse_passed, s.delivered,
                     round(s.load, 5), s.moment) for s in states)
        if key in visited:
            continue
        visited.add(key)
        expanded += 1
        for vi, nxt, _, _, _ in _fleet_candidates(problem, states, ids - completed):
            successors = tuple(nxt if j == vi else s
                               for j, s in enumerate(states))
            heapq.heappush(frontier, (h(successors), next(counter), successors))

    partial_count = len(delivered_ids(best_partial))
    result = _fleet_complete(
        problem, best_partial, partial_count, expanded, start, "fleet_gbfs",
        "Pesquisa sem solução completa (limite de expansões ou fronteira esgotada)"
    )
    result.feasible = False
    return result



def fleet_astar(problem, heuristic_variant=1, max_expansions=10000):
    """A* para dois veículos, respeitando capacidade, precedência e horários.

    H1: limite inferior 0 (custo uniforme, admissível).
    H2: menor deslocação até uma ação necessária, limite inferior admissível
        para distâncias não negativas, sem garantia de consistência.
    O limite de expansões impede garantir solução ou otimalidade se atingido.
    """
    if heuristic_variant not in (1, 2):
        raise ValueError("heuristic_variant deve ser 1 ou 2")
    if max_expansions < 1:
        raise ValueError("max_expansions deve ser positivo")
    start = time.perf_counter()
    root = tuple(State(v.center_location, problem.start_time) for v in problem.vehicles)
    ids = frozenset(r.item_id for r in problem.requests)
    counter = itertools.count()

    def delivered_ids(states):
        return frozenset().union(*(s.delivered for s in states))

    def heuristic(states):
        if heuristic_variant == 1 or delivered_ids(states) == ids:
            return 0.0
        choices = list(_fleet_candidates(problem, states, ids - delivered_ids(states),
                                         check_time_windows=True))
        return min((km for _, _, km, _, _ in choices), default=0.0)

    queue = [(heuristic(root), next(counter), root)]
    # Guardar tempos no estado: dois estados com a mesma posição/carga mas
    # horários diferentes não têm necessariamente os mesmos sucessores.
    best_cost = {}
    expanded = 0
    best_partial = root
    while queue and expanded < max_expansions:
        _, _, states = heapq.heappop(queue)
        delivered = delivered_ids(states)
        if (len(delivered), -sum(s.km for s in states)) > (len(delivered_ids(best_partial)), -sum(s.km for s in best_partial)):
            best_partial = states
        if delivered == ids:
            return _fleet_complete(problem, states, len(delivered), expanded,
                                   start, "fleet_astar")
        key = tuple((s.location, s.picked, s.warehouse_passed, s.delivered,
                     round(s.load, 5), s.moment) for s in states)
        cost = sum(s.km for s in states)
        if cost >= best_cost.get(key, math.inf) - 1e-9:
            continue
        best_cost[key] = cost
        expanded += 1
        for vi, nxt, _, _, _ in _fleet_candidates(problem, states, ids - delivered,
                                                  check_time_windows=True):
            successors = tuple(nxt if j == vi else s for j, s in enumerate(states))
            g = sum(s.km for s in successors)
            heapq.heappush(queue, (g + heuristic(successors), next(counter), successors))
    delivered = delivered_ids(best_partial)
    result = _fleet_complete(problem, best_partial, len(delivered), expanded,
                             start, "fleet_astar",
                             "Sem solução completa dentro do limite de expansões ou fronteira esgotada")
    result.feasible = False
    return result


def validar_rota_frota(problem, result, tolerance_km=0.02):
    """Reexecuta operações da rota com todas as restrições ativas.

    Retorna (válida, mensagens). Uma rota parcial não é considerada completa.
    """
    errors = []
    if not result.routes:
        return False, ["Resultado sem rotas"]
    sequences = []
    for vehicle in problem.vehicles:
        route = result.routes.get(vehicle.vehicle_id)
        if route is None:
            errors.append(f"Veículo {vehicle.vehicle_id}: rota em falta")
            sequences.append([])
            continue
        if not route or route[-1][0] != "regresso" or route[-1][2] != vehicle.center_location:
            errors.append(f"Veículo {vehicle.vehicle_id}: regresso ao centro em falta")
        sequences.append([(kind, item) for kind, item, _, _ in route if kind != "regresso"])
    if errors:
        return False, errors
    replay = evaluate_fleet_sequences(problem, sequences)
    if replay is None:
        return False, ["A sequência viola precedência, capacidade, horários ou cobertura"]
    for vehicle in problem.vehicles:
        original = result.routes[vehicle.vehicle_id]
        expected = replay.routes[vehicle.vehicle_id]
        if len(original) != len(expected):
            errors.append(f"Veículo {vehicle.vehicle_id}: número de operações divergente")
            continue
        for i, (old, new) in enumerate(zip(original, expected), start=1):
            if old[:3] != new[:3] or pd.Timestamp(old[3]) != pd.Timestamp(new[3]):
                errors.append(f"Veículo {vehicle.vehicle_id}: operação {i} não coincide com a simulação")
                break
    if abs(replay.distance_km - result.distance_km) > tolerance_km:
        errors.append("Distância declarada diferente da distância recalculada")
    if result.delivered != len(problem.requests) or not result.feasible:
        errors.append("Resultado não assinalado como completo")
    return not errors, errors


def fleet_summary(results):
    return pd.DataFrame([{
        "algoritmo":r.algorithm,"viavel":r.feasible,"distancia_km":round(r.distance_km,2),
        "nos_ou_passos":r.expanded,"tempo_s":r.runtime_s,
        "artigos_entregues":r.delivered,"total_artigos":r.total,
        "observacoes":r.message
    } for r in results])


# ================== HILL CLIMBING E CSP PARA A FROTA ==================

def evaluate_fleet_sequences(problem, sequences):
    """Valida rotas completas com recolha, passagem pelo centro e entrega."""
    start = time.perf_counter()
    by_id = {r.item_id: r for r in problem.requests}
    states = [State(v.center_location, problem.start_time) for v in problem.vehicles]
    for vi, sequence in enumerate(sequences):
        for kind, item in sequence:
            options = _fleet_candidates(problem, tuple(states), set(by_id),
                                        check_time_windows=True)
            found = next((nxt for j, nxt, _, action, article in options
                          if j == vi and action == kind and article == item), None)
            if found is None:
                return None
            states[vi] = found
    done = set().union(*(s.delivered for s in states))
    if done != set(by_id):
        return None
    return _fleet_complete(problem, tuple(states), len(done), 0, start,
                           "fleet_evaluation")


def fleet_hill_climbing(problem, initial_result, max_iterations=100):
    """Trocas de operações e realocações entre veículos; preserva viabilidade.
    history regista o custo após cada iteração para os gráficos do relatório.
    """
    start = time.perf_counter()
    count = len(problem.requests)
    if not initial_result.feasible:
        return FleetResult("fleet_hill_climbing", False, math.inf, 0,
                           time.perf_counter()-start, {}, 0, count,
                           "É necessária uma solução inicial completa")
    sequences = [
        [(kind,item) for kind,item,_,_ in initial_result.routes[v.vehicle_id]
         if kind != "regresso"]
        for v in problem.vehicles
    ]
    current = evaluate_fleet_sequences(problem, sequences)
    if current is None:
        return FleetResult("fleet_hill_climbing", False, math.inf, 0,
                           time.perf_counter()-start, {}, 0, count,
                           "A rota inicial não passou na validação")
    history = [current.distance_km]
    iterations = 0
    for _ in range(max_iterations):
        best = current
        best_sequences = sequences
        # Permuta pares de operações na mesma rota.
        for vi, seq in enumerate(sequences):
            for i in range(len(seq)):
                for j in range(i+1,len(seq)):
                    candidate = [list(s) for s in sequences]
                    candidate[vi][i], candidate[vi][j] = candidate[vi][j], candidate[vi][i]
                    outcome = evaluate_fleet_sequences(problem, candidate)
                    if outcome and outcome.distance_km < best.distance_km - 1e-9:
                        best, best_sequences = outcome, candidate
        # Move o par recolha/entrega de um veículo para outro (no final).
        for vi, seq in enumerate(sequences):
            items = {item for kind,item in seq if kind == "recolha"}
            for item in items:
                for vj in range(len(sequences)):
                    if vj == vi:
                        continue
                    candidate = [[op for op in s if not (j == vi and op[1] == item)]
                                 for j,s in enumerate(sequences)]
                    candidate[vj].extend([("recolha",item), ("entrega",item)])
                    outcome = evaluate_fleet_sequences(problem, candidate)
                    if outcome and outcome.distance_km < best.distance_km - 1e-9:
                        best, best_sequences = outcome, candidate
        iterations += 1
        if best_sequences is sequences:
            break
        sequences, current = best_sequences, best
        history.append(current.distance_km)
    current.algorithm = "fleet_hill_climbing"
    current.expanded = iterations
    current.runtime_s = time.perf_counter()-start
    current.history = history
    return current


def fleet_csp(problem, max_nodes=50000):
    """Procura por backtracking para encontrar atribuições/ordens válidas.
    Variáveis: ações a atribuir a veículos e sua ordem.
    Domínios: ações permitidas por veículo.
    Restrições: precedência, capacidade e janelas temporais.
    """
    start = time.perf_counter()
    root = tuple(State(v.center_location, problem.start_time)
                 for v in problem.vehicles)
    ids = frozenset(r.item_id for r in problem.requests)
    nodes, backtracks = 0, 0
    def dfs(states):
        nonlocal nodes, backtracks
        delivered = frozenset().union(*(s.delivered for s in states))
        if delivered == ids:
            return states
        if nodes >= max_nodes:
            return None
        nodes += 1
        choices = list(_fleet_candidates(problem, states, ids-delivered))
        choices.sort(key=lambda x: (x[1].moment, x[2], x[0]))
        for vi,nxt,_,_,_ in choices:
            new_states = tuple(nxt if j == vi else s
                               for j,s in enumerate(states))
            answer = dfs(new_states)
            if answer is not None:
                return answer
            if nodes >= max_nodes:
                break
        backtracks += 1
        return None
    answer = dfs(root)
    if answer is None:
        return FleetResult("fleet_csp",False,math.inf,nodes,
                           time.perf_counter()-start,{},0,len(ids),
                           f"Sem solução completa ou limite atingido; backtracks={backtracks}")
    return _fleet_complete(problem,answer,len(ids),nodes,start,"fleet_csp",
                           f"backtracks={backtracks}")


def validar_passagem_centro(rotas, centers=None, expected_items=None):
    """Confirma recolha -> centro -> entrega em cada rota.

    centers: mapeamento vehicle_id -> location_id, para verificar a localização.
    expected_items: conjunto de artigos que devem ser entregues exatamente uma vez.
    """
    delivered_global = set()
    for vehicle_id, route in rotas.items():
        picked = set()
        passed = set()
        for kind, item, loc, _ in route:
            if kind == "recolha":
                if item in picked or item in delivered_global:
                    return False
                picked.add(item)
            elif kind == "armazem":
                if centers is not None and loc != centers[vehicle_id]:
                    return False
                passed.update(picked)
            elif kind == "entrega":
                if item not in passed or item in delivered_global:
                    return False
                delivered_global.add(item)
            elif kind != "regresso":
                return False
    if expected_items is not None and delivered_global != set(expected_items):
        return False
    return True


if __name__ == "__main__":
    p = load_fleet_problem("data")
    p.requests = p.requests[:3]
    g = fleet_greedy(p)
    a = fleet_astar(p, max_expansions=2000)
    h = fleet_hill_climbing(p, a if a.feasible else g)
    c = fleet_csp(p, max_nodes=2000)
    print(fleet_summary([g,a,h,c]).to_string(index=False))


