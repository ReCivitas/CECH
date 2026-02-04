"""
CECH v1.0 - Simulação do Problema dos 3 Corpos
Oráculo de Tríada Hiperconsistente
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class OraculoTresCorpos:
    """
    Simulação do problema dos 3 corpos com métrica reológica hiperconsistente.
    Implementa o Oráculo de Tríada: Caos → Ordem → Hiperconsistência
    """

    def __init__(self, massas: np.ndarray = None, 
                 viscosidade: float = 0.0,
                 dt: float = 0.01):
        """
        Args:
            massas: Array [m1, m2, m3] em unidades arbitrárias
            viscosidade: Parâmetro η da reologia (0 = caos puro)
            dt: Passo de integração
        """
        self.massas = massas if massas is not None else np.array([1.0, 1.0, 1.0])
        self.eta = viscosidade
        self.dt = dt
        self.G = 1.0  # Constante gravitacional normalizada

    def _derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calcula derivadas para integração.
        y = [x1, y1, z1, x2, y2, z2, x3, y3, z3, 
             vx1, vy1, vz1, vx2, vy2, vz2, vx3, vy3, vz3]
        """
        n = 3
        pos = y[:3*n].reshape(n, 3)
        vel = y[3*n:].reshape(n, 3)

        # Acelerações gravitacionais
        acc = np.zeros((n, 3))
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_ij = pos[j] - pos[i]
                    dist = np.linalg.norm(r_ij)
                    if dist > 1e-10:
                        acc[i] += self.G * self.massas[j] * r_ij / dist**3

        # Termo de viscosidade (amortecimento reológico)
        if self.eta > 0:
            acc -= self.eta * vel

        # Retorna [velocidades, acelerações]
        return np.concatenate([vel.flatten(), acc.flatten()])

    def simular(self, condicoes_iniciais: np.ndarray, 
                t_max: float = 100.0,
                metodo: str = 'RK45') -> Dict:
        """
        Executa simulação completa.

        Args:
            condicoes_iniciais: [posições (9), velocidades (9)]
            t_max: Tempo máximo de simulação
            metodo: Método de integração

        Returns:
            Dict com trajetórias e métricas
        """
        t_span = (0, t_max)
        t_eval = np.arange(0, t_max, self.dt)

        sol = solve_ivp(
            self._derivatives,
            t_span,
            condicoes_iniciais,
            method=metodo,
            t_eval=t_eval,
            dense_output=True
        )

        # Extrair trajetórias
        n = 3
        trajetorias = []
        for i in range(n):
            traj = {
                'x': sol.y[i*3],
                'y': sol.y[i*3 + 1],
                'z': sol.y[i*3 + 2],
                'vx': sol.y[9 + i*3],
                'vy': sol.y[9 + i*3 + 1],
                'vz': sol.y[9 + i*3 + 2]
            }
            trajetorias.append(traj)

        # Calcular métricas
        energia = self._calcular_energia(sol.y)
        entropia = self._calcular_entropia(sol.y)

        return {
            't': sol.t,
            'trajetorias': trajetorias,
            'energia': energia,
            'entropia': entropia,
            'sucesso': sol.success,
            'mensagem': sol.message
        }

    def _calcular_energia(self, y: np.ndarray) -> np.ndarray:
        """Calcula energia total do sistema ao longo do tempo"""
        n = 3
        n_pontos = y.shape[1]
        energia = np.zeros(n_pontos)

        for k in range(n_pontos):
            pos = y[:9, k].reshape(n, 3)
            vel = y[9:, k].reshape(n, 3)

            # Energia cinética
            ec = 0.5 * sum(self.massas[i] * np.linalg.norm(vel[i])**2 
                          for i in range(n))

            # Energia potencial gravitacional
            ep = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    r_ij = np.linalg.norm(pos[i] - pos[j])
                    if r_ij > 1e-10:
                        ep -= self.G * self.massas[i] * self.massas[j] / r_ij

            energia[k] = ec + ep

        return energia

    def _calcular_entropia(self, y: np.ndarray) -> np.ndarray:
        """Calcula entropia de posições (medida de dispersão)"""
        n_pontos = y.shape[1]
        entropia = np.zeros(n_pontos)

        for k in range(n_pontos):
            pos = y[:9, k].reshape(3, 3)
            # Entropia baseada na variância das posições
            variancia = np.var(pos, axis=0).sum()
            entropia[k] = np.log(variancia + 1)

        return entropia

    def classificar_regime(self, entropia: np.ndarray) -> str:
        """
        Classifica o regime do sistema:
        - CAOS: entropia alta e crescente
        - ORDEM: entropia baixa e estável
        - HIPERCONSISTÊNCIA: entropia oscilante controlada
        """
        entropia_media = np.mean(entropia)
        entropia_std = np.std(entropia)

        if entropia_media > 5 and entropia_std > 2:
            return "CAOS"
        elif entropia_media < 2 and entropia_std < 0.5:
            return "ORDEM"
        elif entropia_std > 1 and entropia_std < 3:
            return "HIPERCONSISTÊNCIA"
        else:
            return "TRANSICAO"


class ComparadorRegimes:
    """Compara os três regimes: Caos, Ordem, Hiperconsistência"""

    def __init__(self):
        self.resultados = {}

    def executar_comparacao(self, condicoes_iniciais: np.ndarray,
                           t_max: float = 50.0) -> Dict:
        """Executa simulações nos três regimes"""

        # Regime 1: Caos (baixa viscosidade)
        print("Simulando regime CAOS (η=0)...")
        caos = OraculoTresCorpos(viscosidade=0.0, dt=0.01)
        self.resultados['caos'] = caos.simular(condicoes_iniciais, t_max)

        # Regime 2: Ordem (viscosidade moderada)
        print("Simulando regime ORDEM (η=0.1)...")
        ordem = OraculoTresCorpos(viscosidade=0.1, dt=0.01)
        self.resultados['ordem'] = ordem.simular(condicoes_iniciais, t_max)

        # Regime 3: Hiperconsistência (alta viscosidade logarítmica)
        print("Simulando regime HIPERCONSISTÊNCIA (η=1.0)...")
        hiper = OraculoTresCorpos(viscosidade=1.0, dt=0.01)
        self.resultados['hiperconsistencia'] = hiper.simular(condicoes_iniciais, t_max)

        return self.resultados

    def gerar_relatorio(self) -> str:
        """Gera relatório comparativo"""
        relatorio = []
        relatorio.append("=" * 60)
        relatorio.append("RELATÓRIO COMPARATIVO: TRIALOGIA DOS REGIMES")
        relatorio.append("=" * 60)

        for regime, dados in self.resultados.items():
            relatorio.append(f"\n{regime.upper()}:")
            relatorio.append(f"  Energia média: {np.mean(dados['energia']):.6f}")
            relatorio.append(f"  Entropia média: {np.mean(dados['entropia']):.4f}")
            relatorio.append(f"  Desvio entropia: {np.std(dados['entropia']):.4f}")
            relatorio.append(f"  Classificação: {OraculoTresCorpos().classificar_regime(dados['entropia'])}")

        return "\n".join(relatorio)


def gerar_condicoes_iniciais_padrao() -> np.ndarray:
    """Gera condições iniciais padrão para simulação"""
    # Posições iniciais (triângulo equilátero no plano xy)
    pos = np.array([
        1.0, 0.0, 0.0,    # Corpo 1
        -0.5, 0.866, 0.0, # Corpo 2
        -0.5, -0.866, 0.0 # Corpo 3
    ])

    # Velocidades iniciais (rotação aproximadamente circular)
    vel = np.array([
        0.0, 0.5, 0.0,
        -0.433, -0.25, 0.0,
        0.433, -0.25, 0.0
    ])

    return np.concatenate([pos, vel])


def visualizar_trialogia(resultados: Dict, save_path: str = None):
    """Visualiza as três simulações lado a lado"""
    fig = plt.figure(figsize=(18, 6))

    regimes = ['caos', 'ordem', 'hiperconsistencia']
    titulos = ['CAOS (P≠NP)', 'ORDEM (P→NP)', 'HIPERCONSISTÊNCIA (P=NP*)']
    cores = ['#e74c3c', '#27ae60', '#8e44ad']

    for idx, (regime, titulo, cor) in enumerate(zip(regimes, titulos, cores)):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        dados = resultados[regime]

        for i, traj in enumerate(dados['trajetorias']):
            ax.plot(traj['x'], traj['y'], traj['z'], 
                   color=cor, alpha=0.7, linewidth=1)
            ax.scatter(traj['x'][0], traj['y'][0], traj['z'][0],
                      color='green', s=50, marker='o')
            ax.scatter(traj['x'][-1], traj['y'][-1], traj['z'][-1],
                      color='red', s=50, marker='s')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(titulo, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.suptitle('Oráculo de Tríada: Problema dos 3 Corpos', 
                fontsize=16, fontweight='bold', y=1.02)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("ORÁCULO DE TRÍADA: Problema dos 3 Corpos")
    print("Simulação com Reologia Hiperconsistente")
    print("=" * 60)

    # Condições iniciais
    ci = gerar_condicoes_iniciais_padrao()

    # Comparar regimes
    comparador = ComparadorRegimes()
    resultados = comparador.executar_comparacao(ci, t_max=30.0)

    # Relatório
    print(comparador.gerar_relatorio())

    # Visualização
    print("\nGerando visualização...")
    visualizar_trialogia(resultados, 'oraculo_triada.png')

    print("\nSimulação concluída!")
