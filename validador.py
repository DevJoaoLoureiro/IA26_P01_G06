"""Validador independente das rotas da frota.

Lê diretamente os CSV do dataset e verifica uma solução SEM reutilizar
nenhuma função de algoritmos.py. Serve como auditoria: se os algoritmos e o
modelo tivessem um erro, este validador apanhava-o.

Uma solução é um dicionário {vehicle_id: rota}, em que a rota é uma lista de
operações (tipo, artigo, location_id, início_do_serviço) com tipo em
{"recolha", "armazem", "entrega", "regresso"}, tal como é produzida pelos
algoritmos (FleetResult.routes).

Regras verificadas (rules.csv e enunciado):
  * cada artigo é recolhido e entregue exatamente uma vez, pelo mesmo veículo;
  * a recolha é feita no produtor do produto e a entrega no local da encomenda;
  * fluxo produtor -> armazém (centro do veículo) -> local de entrega;
  * janelas: o INÍCIO do serviço respeita a janela de disponibilidade
    (e não é anterior à data da encomenda) e a janela de entrega;
  * tempos: viagem da matriz + stop_service_min por paragem +
    warehouse_handling_min por visita ao armazém; operações seguidas do mesmo
    tipo, no mesmo local e com o mesmo início formam uma única paragem;
  * capacidade do veículo nunca excedida;
  * todas as operações terminam dentro do horizonte de planeamento;
  * a rota termina com regresso ao centro;
  * a distância total declarada coincide com a recalculada.
"""
from pathlib import Path

import pandas as pd


def _carregar(pasta):
    pasta = Path(pasta)
    ler = lambda nome: pd.read_csv(pasta / f"{nome}.csv")
    regras = dict(ler("rules")[["rule", "value"]].values)
    locs = ler("locations")
    local = {(r.location_type, int(r.ref_id)): int(r.location_id)
             for r in locs.itertuples(index=False)}
    itens = (ler("order_items")
             .merge(ler("orders"), on="order_id")
             .merge(ler("products")[["product_id", "producer_id"]], on="product_id"))
    artigos = {}
    for r in itens.itertuples(index=False):
        artigos[int(r.order_item_id)] = dict(
            recolha=local[("producer", int(r.producer_id))],
            entrega=local[("delivery_location", int(r.delivery_location_id))],
            peso=float(r.weight_kg),
            encomenda=pd.Timestamp(r.order_datetime),
            disp_ini=pd.Timestamp(r.availability_start),
            disp_fim=pd.Timestamp(r.availability_end),
            ent_ini=pd.Timestamp(r.delivery_start),
            ent_fim=pd.Timestamp(r.delivery_end))
    veiculos = {int(v.vehicle_id): dict(capacidade=float(v.capacity_kg),
                                        centro=local[("collection_center", int(v.center_id))])
                for v in ler("vehicles").itertuples(index=False)}
    matriz = ler("distance_matrix")
    km = {(int(r.origin_location_id), int(r.destination_location_id)): float(r.distance_km)
          for r in matriz.itertuples(index=False)}
    minutos = {(int(r.origin_location_id), int(r.destination_location_id)): float(r.duration_min)
               for r in matriz.itertuples(index=False)}
    inicio = pd.Timestamp(regras["start_date"])
    fim = inicio + pd.Timedelta(days=int(regras["horizon_days"]))
    return dict(artigos=artigos, veiculos=veiculos, km=km, minutos=minutos,
                servico=float(regras["stop_service_min"]),
                armazem=float(regras["warehouse_handling_min"]),
                inicio=inicio, fim=fim)


def validar(pasta, rotas, distancia_declarada=None, artigos_esperados=None,
            tolerancia_km=0.05):
    """Devolve (válida, lista_de_erros, distância_recalculada_km)."""
    d = _carregar(pasta)
    erros = []
    distancia = 0.0
    entregues = {}
    recolhidos = {}
    viagem_km = lambda a, b: 0.0 if a == b else d["km"][(a, b)]
    viagem_min = lambda a, b: pd.Timedelta(minutes=0.0 if a == b else d["minutos"][(a, b)])

    for vid, rota in rotas.items():
        v = d["veiculos"][vid]
        local, livre = v["centro"], d["inicio"]
        carga, a_bordo, no_armazem = 0.0, set(), set()
        anterior = None
        for n, (tipo, artigo, loc, inicio_txt) in enumerate(rota, start=1):
            ref = f"veículo {vid}, operação {n} ({tipo} {artigo})"
            inicio = pd.Timestamp(inicio_txt)
            mesma_paragem = (anterior is not None and tipo in ("recolha", "entrega")
                             and anterior[0] == tipo and anterior[1] == loc
                             and anterior[2] == inicio)
            if not mesma_paragem:
                distancia += viagem_km(local, loc)
                chegada = livre + viagem_min(local, loc)
                if inicio < chegada - pd.Timedelta(seconds=1):
                    erros.append(f"{ref}: começa antes de o veículo poder chegar")
            if tipo == "regresso":
                if loc != v["centro"]:
                    erros.append(f"{ref}: regresso fora do centro")
                if n != len(rota):
                    erros.append(f"{ref}: há operações depois do regresso")
                local = loc
                break
            if tipo == "armazem":
                if loc != v["centro"]:
                    erros.append(f"{ref}: armazém diferente do centro do veículo")
                no_armazem |= a_bordo
                livre = inicio + pd.Timedelta(minutes=d["armazem"])
            else:
                a = d["artigos"].get(artigo)
                if a is None:
                    erros.append(f"{ref}: artigo inexistente")
                    continue
                if tipo == "recolha":
                    if loc != a["recolha"]:
                        erros.append(f"{ref}: local de recolha errado")
                    if artigo in recolhidos:
                        erros.append(f"{ref}: artigo recolhido mais de uma vez")
                    if not (max(a["disp_ini"], a["encomenda"]) <= inicio <= a["disp_fim"]):
                        erros.append(f"{ref}: fora da janela de recolha")
                    recolhidos[artigo] = vid
                    a_bordo.add(artigo)
                    carga += a["peso"]
                    if carga > v["capacidade"] + 1e-6:
                        erros.append(f"{ref}: capacidade excedida ({carga:.1f} kg)")
                else:
                    if loc != a["entrega"]:
                        erros.append(f"{ref}: local de entrega errado")
                    if recolhidos.get(artigo) != vid:
                        erros.append(f"{ref}: entregue sem recolha pelo mesmo veículo")
                    if artigo not in no_armazem:
                        erros.append(f"{ref}: entregue sem passar pelo armazém")
                    if artigo in entregues:
                        erros.append(f"{ref}: artigo entregue mais de uma vez")
                    if not (a["ent_ini"] <= inicio <= a["ent_fim"]):
                        erros.append(f"{ref}: fora da janela de entrega")
                    entregues[artigo] = vid
                    a_bordo.discard(artigo)
                    carga -= a["peso"]
                if not mesma_paragem:
                    livre = inicio + pd.Timedelta(minutes=d["servico"])
            if livre > d["fim"]:
                erros.append(f"{ref}: termina depois do horizonte")
            local = loc
            anterior = (tipo, loc, inicio)
        if not rota or rota[-1][0] != "regresso":
            erros.append(f"veículo {vid}: a rota não termina com regresso ao centro")

    esperados = set(d["artigos"]) if artigos_esperados is None else set(artigos_esperados)
    em_falta = esperados - set(entregues)
    if em_falta:
        erros.append(f"{len(em_falta)} artigos por entregar: {sorted(em_falta)[:10]}")
    if distancia_declarada is not None and abs(distancia - distancia_declarada) > tolerancia_km:
        erros.append(f"distância declarada {distancia_declarada:.2f} km "
                     f"≠ recalculada {distancia:.2f} km")
    return not erros, erros, distancia
