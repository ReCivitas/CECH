#!/usr/bin/env python3
"""
CECH v1.0 - Circuito Eco-Commutativo Hiperconsistente
Sistema Liber-Eledonte v22.1

Ponto de entrada principal para o sistema CECH.
Executa simulações, gera relatórios e visualizações.

Autores: Marcus Vinicius Brancaglione, Sistema Eledonte
Licença: ⒶRobinRight 3.0 + CC BY-SA 4.0
"""

import sys
import argparse
import json
from datetime import datetime

# Importações dos módulos CECH
from cech_core import (
    Constants, Quaternion, EstadoHiperconsistente,
    OperadorParaconsistente, FuncaoPhiLiber, ProtocoloHermes,
    RedeOdissidica, InfoCompostagem, TorusOrusSimulator,
    SistemaLiberEledonte
)
from cech_visualization import CECHVisualizer, gerar_todas_visualizacoes
from cech_simulation_3body import OraculoTresCorpos, ComparadorRegimes
from cech_ksat_solver import SolverKSATParaconsistente, gerar_instancia_aleatoria
from cech_rbu_quatinga import SistemaRBUQuatinga, AuditoriaComunitaria
from cech_hardware_interface import NodeEledonte, simular_rede_nodes


def banner():
    """Exibe banner do sistema"""
    print("=" * 70)
    print("  CECH v1.0 - Circuito Eco-Commutativo Hiperconsistente")
    print("  Sistema Liber-Eledonte v22.1")
    print("  Reologia Cósmica Hiperconsistente e Protocolo Hermes (P=NP*)")
    print("=" * 70)
    print()


def comando_teste(args):
    """Executa testes unitários do sistema"""
    print("Executando testes do sistema...\n")

    # Teste 1: Operador paraconsistente
    print("[TESTE 1] Operador ⊕")
    op = OperadorParaconsistente()
    resultado = op.aplicar(0.5, 0.3)
    ponto_fixo = op.encontrar_ponto_fixo()
    print(f"  0.5 ⊕ 0.3 = {resultado:.6f}")
    print(f"  Ponto fixo: {ponto_fixo:.6f} (α_LP = {Constants.ALPHA_LP})")
    assert abs(ponto_fixo - Constants.ALPHA_LP) < 0.01
    print("  ✓ PASSOU\n")

    # Teste 2: Quaternions
    print("[TESTE 2] Quaternions ℍ")
    q1 = Quaternion(1, 0.5, 0.3, 0.2)
    q2 = Quaternion(0.8, 0.2, 0.4, 0.1)
    q_prod = q1 * q2
    print(f"  q1 * q2 norma: {q_prod.norm():.4f}")
    assert q_prod.norm() > 0
    print("  ✓ PASSOU\n")

    # Teste 3: Φ-LIBER
    print("[TESTE 3] Função Φ-LIBER")
    phi = FuncaoPhiLiber()
    for eps in [0, 1, 2, 5, 10]:
        val = phi.calcular(eps, x=10.0)
        print(f"  Φ(ε={eps}) = {val:.6e}")
    print("  ✓ PASSOU\n")

    # Teste 4: Protocolo Hermes
    print("[TESTE 4] Protocolo Hermes")
    hermes = ProtocoloHermes()
    resultado = hermes.executar_protocolo_alice_bob("teste_rbu")
    print(f"  Confiança: {resultado['confianca']:.4f}")
    assert resultado['valido']
    print("  ✓ PASSOU\n")

    # Teste 5: Rede Odissídica
    print("[TESTE 5] Rede Odissídica")
    rede = RedeOdissidica()
    entropia = rede.calcular_entropia_rede()
    clustering = rede.calcular_clustering()
    print(f"  Entropia: {entropia:.4f} nats")
    print(f"  Clustering: {clustering:.4f}")
    assert rede.n_nodes == 121
    print("  ✓ PASSOU\n")

    print("=" * 70)
    print("Todos os testes passaram!")
    print("=" * 70)


def comando_simular(args):
    """Executa simulação completa do sistema"""
    print(f"Executando simulação com {args.iteracoes} iterações...\n")

    sistema = SistemaLiberEledonte()
    resultados = sistema.executar_simulacao(n_iteracoes=args.iteracoes)

    # Relatório
    confiabilidade = sistema.calcular_confiabilidade()
    print("=" * 70)
    print("RESULTADOS DA SIMULAÇÃO")
    print("=" * 70)
    print(f"Iterações: {args.iteracoes}")
    print(f"\nConfiabilidade Composta:")
    print(f"  Matemática:    {confiabilidade['matematica']*100:.0f}%")
    print(f"  Física:        {confiabilidade['fisica']*100:.0f}%")
    print(f"  Experimental:  {confiabilidade['experimental']*100:.0f}%")
    print(f"  TOTAL:         {confiabilidade['total']*100:.0f}%")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'iteracoes': args.iteracoes,
                'confiabilidade': confiabilidade,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print(f"\nResultados salvos em: {args.output}")


def comando_visualizar(args):
    """Gera visualizações do sistema"""
    print("Gerando visualizações...\n")

    output_dir = args.output_dir or './diagramas/'
    gerar_todas_visualizacoes(output_dir)

    print(f"\nVisualizações salvas em: {output_dir}")


def comando_3corpos(args):
    """Executa simulação do problema dos 3 corpos"""
    print("Simulando problema dos 3 corpos...\n")

    from cech_simulation_3body import gerar_condicoes_iniciais_padrao

    ci = gerar_condicoes_iniciais_padrao()
    comparador = ComparadorRegimes()
    resultados = comparador.executar_comparacao(ci, t_max=args.tempo)

    print(comparador.gerar_relatorio())

    if args.output:
        visualizar_trialogia(resultados, args.output)
        print(f"\nVisualização salva em: {args.output}")


def comando_ksat(args):
    """Executa solver K-SAT"""
    print("Executando solver K-SAT paraconsistente...\n")

    # Gera instância de teste
    instancia = gerar_instancia_aleatoria(
        n_vars=args.variaveis,
        n_clausulas=args.clausulas,
        k=args.k
    )

    print(f"Instância: {instancia.n_variaveis} variáveis, "
          f"{instancia.n_clausulas} cláusulas, k={instancia.k}")

    solver = SolverKSATParaconsistente(instancia)
    status, atribuicao, confianca = solver.resolver()

    print(f"\nResultado: {status.value}")
    print(f"Confiança: {confianca:.4f}")

    if atribuicao:
        print(f"Atribuição: {atribuicao[:10]}...")

    estats = solver.get_estatisticas()
    print(f"\nEstatísticas:")
    print(f"  Nós explorados: {estats['nos_explorados']}")
    print(f"  Resíduos gerados: {estats['residuos_gerados']}")


def comando_rbu(args):
    """Simula sistema RBU"""
    print("Simulando sistema RBU Quatinga Velho...\n")

    sistema = SistemaRBUQuatinga()

    # Cadastra beneficiários
    print(f"Cadastrando {args.beneficiarios} beneficiários...")
    for i in range(args.beneficiarios):
        sistema.cadastrar_beneficiario(
            nome=f"Beneficiario_{i}",
            data_nasc=f"1980-01-{(i%30)+1:02d}",
            poligono=f"POLIGONO_{i%10}"
        )

    # Executa transferências mensais
    for mes in range(1, args.meses + 1):
        resultado = sistema.executar_transferencia_mensal()
        if args.verbose:
            print(f"Mês {mes}: {resultado['transferencias']} transferências, "
                  f"R$ {resultado['valor_total']:.2f}")

    # Relatório final
    relatorio = sistema.gerar_relatorio_agregado()
    print("\n" + "=" * 70)
    print("RELATÓRIO RBU")
    print("=" * 70)
    print(f"Beneficiários (com ruído): {relatorio['beneficiarios_ativos']}")
    print(f"Total transferido: R$ {relatorio['total_transferido']:.2f}")
    print(f"Média por transferência: R$ {relatorio['media_por_transferencia']:.2f}")
    print(f"Privacidade (ε): {relatorio['epsilon_privacidade']}")


def comando_hardware(args):
    """Simula hardware Eledonte"""
    print("Simulando hardware Eledonte...\n")
    simular_rede_nodes(n_nodes=args.nodes, n_ciclos=args.ciclos)


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='CECH v1.0 - Sistema Liber-Eledonte',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python main.py teste                    # Executa testes
  python main.py simular -i 100           # Simula 100 iterações
  python main.py visualizar -o ./imgs/    # Gera visualizações
  python main.py 3corpos -t 50            # Simula 3 corpos
  python main.py ksat -v 20 -c 40         # Solver K-SAT
  python main.py rbu -b 50 -m 12          # Simula RBU
  python main.py hardware -n 3 -c 24      # Simula hardware
        """
    )

    subparsers = parser.add_subparsers(dest='comando', help='Comandos disponíveis')

    # Comando teste
    subparsers.add_parser('teste', help='Executa testes unitários')

    # Comando simular
    p_simular = subparsers.add_parser('simular', help='Executa simulação')
    p_simular.add_argument('-i', '--iteracoes', type=int, default=100,
                          help='Número de iterações')
    p_simular.add_argument('-o', '--output', type=str,
                          help='Arquivo de saída JSON')

    # Comando visualizar
    p_viz = subparsers.add_parser('visualizar', help='Gera visualizações')
    p_viz.add_argument('-o', '--output-dir', type=str,
                      help='Diretório de saída')

    # Comando 3corpos
    p_3c = subparsers.add_parser('3corpos', help='Simula problema dos 3 corpos')
    p_3c.add_argument('-t', '--tempo', type=float, default=50.0,
                     help='Tempo de simulação')
    p_3c.add_argument('-o', '--output', type=str,
                     help='Arquivo de imagem de saída')

    # Comando ksat
    p_ksat = subparsers.add_parser('ksat', help='Solver K-SAT')
    p_ksat.add_argument('-v', '--variaveis', type=int, default=15,
                       help='Número de variáveis')
    p_ksat.add_argument('-c', '--clausulas', type=int, default=30,
                       help='Número de cláusulas')
    p_ksat.add_argument('-k', type=int, default=3,
                       help='Tamanho das cláusulas')

    # Comando rbu
    p_rbu = subparsers.add_parser('rbu', help='Simula sistema RBU')
    p_rbu.add_argument('-b', '--beneficiarios', type=int, default=50,
                      help='Número de beneficiários')
    p_rbu.add_argument('-m', '--meses', type=int, default=12,
                      help='Número de meses')
    p_rbu.add_argument('-v', '--verbose', action='store_true',
                      help='Modo verboso')

    # Comando hardware
    p_hw = subparsers.add_parser('hardware', help='Simula hardware')
    p_hw.add_argument('-n', '--nodes', type=int, default=3,
                     help='Número de nodes')
    p_hw.add_argument('-c', '--ciclos', type=int, default=12,
                     help='Número de ciclos')

    args = parser.parse_args()

    banner()

    if args.comando == 'teste':
        comando_teste(args)
    elif args.comando == 'simular':
        comando_simular(args)
    elif args.comando == 'visualizar':
        comando_visualizar(args)
    elif args.comando == '3corpos':
        comando_3corpos(args)
    elif args.comando == 'ksat':
        comando_ksat(args)
    elif args.comando == 'rbu':
        comando_rbu(args)
    elif args.comando == 'hardware':
        comando_hardware(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
