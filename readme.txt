Projeto 1 — Inteligência Artificial 2026/27 — Grupo 06
Otimização de rotas para distribuição de alimentos locais
GitHub: https://github.com/DevJoaoLoureiro/IA26_P01_G06

GRUPO
Guilherme Araujo 30146 · Rodrigo Lopes 31525 · Tomas Gomes 31501 · João Loureiro 31551

CONTEÚDO
01_eda_tiny.ipynb       Análise exploratória do dataset final (80 artigos).
01_Project01_G06.ipynb  Relatório principal: modelação, procura informada,
                        CSP, procura local, resultados, análise e conclusão.
algoritmos.py           Classes do modelo (Problem, Request, Vehicle, State)
                        e todos os algoritmos.
model.py                Interface do modelo: leitura das instâncias,
                        função de custo e verificador de viabilidade.
validador.py            Validador independente das soluções (lê os CSV
                        diretamente, sem usar algoritmos.py).
test_validacao.py       Testes com rotas escritas à mão e testes dos
                        algoritmos (pequenas instâncias e instância completa).
data/                   Os 12 ficheiros CSV do dataset.

EXECUÇÃO
Requisitos: Python 3.10+, pandas, matplotlib, Jupyter.
  pip install pandas matplotlib jupyter
Abrir os notebooks a partir desta pasta e executar todas as células por
ordem (o notebook principal demora alguns minutos).
Testes: python test_validacao.py
No Google Colab, se os ficheiros não estiverem presentes, a primeira
célula de cada notebook clona o repositório.

RESULTADOS PRINCIPAIS (80 artigos, 2 veículos)
Todos os algoritmos abaixo entregam os 80 artigos; as soluções são
confirmadas pelo validador independente.
  Greedy H1 + reparação por inserção   solução inicial completa
  CSP (MRV + forward checking)         solução completa e viável
  Hill Climbing (relocate)             melhora a solução do Greedy
  Simulated Annealing                  melhor solução encontrada
GBFS e A* (duas heurísticas cada) são avaliados em instâncias de 3, 5 e 8
artigos: o espaço de estados cresce exponencialmente e torna a procura
sistemática impraticável para os 80 artigos.

LIMITAÇÕES
O A* e o GBFS têm limite de expansões; acima de 5 artigos não terminam.
A procura local encontra ótimos locais, sem garantia de ótimo global.
O modelo mantém a carga no veículo depois do armazém (permanência de 0 h,
dentro do máximo de 24 h); o armazenamento entre viagens não é explorado.
