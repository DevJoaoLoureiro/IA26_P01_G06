Projeto 1 — Inteligência Artificial 2026/27 — Grupo 06
GitHub: https://github.com/DevJoaoLoureiro/IA26_P01_G06

CONTEÚDO
01_Project01_G06.ipynb — notebook principal com metodologia, resultados e gráficos.
01_eda_tiny.ipynb — análise exploratória.
02_relatorio.ipynb — notebook original, preservado para rastreabilidade.
algoritmos.py — classes, algoritmos e validador de rotas.
test_validacao.py — testes de regressão executáveis.
data/ — 12 ficheiros CSV de entrada.

EXECUÇÃO
No Google Colab: abrir 01_Project01_G06.ipynb e executar as células.
O notebook original clona o repositório GitHub: é necessário atualizar o
repositório com estes ficheiros antes de o executar, ou ajustar o bloco de
importação para utilizar os ficheiros carregados na sessão Colab.
Localmente: pip install pandas matplotlib; python test_validacao.py.

LIMITAÇÕES
O A* tem limite de expansões e não garante ótimo quando interrompido.
H1 do A* equivale a custo uniforme; H2 usa heurística de distância.
Hill Climbing exige solução inicial Greedy completa.
Os textos de resultados históricos no notebook devem ser atualizados após
nova execução integral; não devem ser apresentados como resultados novos.
O dataset completo de 80 artigos não teve solução completa confirmada.
