"""
CECH v1.0 - Módulo de Visualização
Geração de gráficos, diagramas e análises visuais
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple, Optional
import io
import base64

# Configuração de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

class CECHVisualizer:
    """Gerador de visualizações do sistema CECH"""

    def __init__(self):
        self.colors = {
            'primary': '#1a5276',
            'secondary': '#7f8c8d', 
            'accent': '#8e44ad',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'light': '#ecf0f1',
            'dark': '#2c3e50'
        }

    def plot_phi_liber_surface(self, save_path: str = None) -> str:
        """
        Superfície 3D da função Φ-LIBER
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        epsilon = np.linspace(0, 10, 100)
        x = np.linspace(1, 20, 100)
        Epsilon, X = np.meshgrid(epsilon, x)

        # Φ(ε, x) = (4π · e^(-ε²/2) · c²) / (3γ · x · ln(x))
        c = 299792458
        gamma = 0.5
        epsilon_eff = 2.5 * np.tanh(Epsilon / 2.5)
        Phi = (4 * np.pi * np.exp(-epsilon_eff**2 / 2) * c**2) /               (3 * gamma * X * np.log(X))

        # Clip para visualização
        Phi = np.clip(Phi, 0, 1e30)

        surf = ax.plot_surface(Epsilon, X, np.log10(Phi + 1), 
                               cmap='viridis', alpha=0.8)

        ax.set_xlabel('ε (Liberdade)')
        ax.set_ylabel('x (Variável de estado)')
        ax.set_zlabel('log₁₀(Φ)')
        ax.set_title('Função Φ-LIBER: Campo Reológico Hiperconsistente')

        fig.colorbar(surf, ax=ax, label='log₁₀(Φ)')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            return save_path

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        return img_str

    def plot_rede_odissidica(self, adj_matrix: np.ndarray, 
                             save_path: str = None) -> str:
        """
        Visualização da Rede Odissídica 11×11
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        n = int(np.sqrt(len(adj_matrix)))
        pos = {}
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                # Layout em grade com perturbação para small-world
                theta = 2 * np.pi * idx / (n * n)
                r = 3 + 0.5 * np.sin(5 * theta)
                pos[idx] = (j + 0.1 * np.random.randn(), 
                           i + 0.1 * np.random.randn())

        # Desenha arestas
        for i in range(len(adj_matrix)):
            for j in range(i+1, len(adj_matrix)):
                if adj_matrix[i, j] > 0:
                    x = [pos[i][0], pos[j][0]]
                    y = [pos[i][1], pos[j][1]]
                    ax.plot(x, y, 'gray', alpha=0.3, linewidth=0.5)

        # Desenha nodos
        for idx, (x, y) in pos.items():
            # Cor baseada na posição (gradiente)
            color = plt.cm.plasma(idx / (n * n))
            circle = Circle((x, y), 0.15, color=color, 
                          ec='black', linewidth=1, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, str(idx), ha='center', va='center',
                   fontsize=6, color='white', fontweight='bold')

        ax.set_xlim(-1, n)
        ax.set_ylim(-1, n)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Rede Odissídica 11×11: Topologia de Dádiva', 
                    fontsize=14, pad=20)

        # Legenda
        legend_elements = [
            mpatches.Patch(color='gray', alpha=0.3, label='Conexões'),
            mpatches.Circle((0, 0), 0.1, color=plt.cm.plasma(0.5), 
                          label='Nodos (gradiente de atividade)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            return save_path

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        return img_str

    def plot_torus_orus_contraction(self, historico: List[Dict],
                                    save_path: str = None) -> str:
        """
        Visualização da contração Torus → Orus
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        passos = [h['passo'] for h in historico]
        volumes = [h['volume'] for h in historico]
        lambdas = [h['lambda_liber'] for h in historico]
        r_majors = [h['r_major'] for h in historico]
        r_minors = [h['r_minor'] for h in historico]

        # Gráfico 1: Volume ao longo do tempo
        ax1 = axes[0, 0]
        ax1.semilogy(passos, volumes, color=self.colors['primary'], linewidth=2)
        ax1.set_xlabel('Passo de Contração')
        ax1.set_ylabel('Volume (escala log)')
        ax1.set_title('Contração do Volume Toroidal')
        ax1.grid(True, alpha=0.3)

        # Gráfico 2: Força Liber
        ax2 = axes[0, 1]
        ax2.plot(passos, lambdas, color=self.colors['accent'], linewidth=2)
        ax2.set_xlabel('Passo de Contração')
        ax2.set_ylabel('Λ (Força Liber)')
        ax2.set_title('Compensação da Força Liber')
        ax2.grid(True, alpha=0.3)

        # Gráfico 3: Raios
        ax3 = axes[1, 0]
        ax3.plot(passos, r_majors, label='R Major', 
                color=self.colors['primary'], linewidth=2)
        ax3.plot(passos, r_minors, label='r Minor', 
                color=self.colors['warning'], linewidth=2)
        ax3.set_xlabel('Passo de Contração')
        ax3.set_ylabel('Raio')
        ax3.set_title('Evolução dos Raios')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Gráfico 4: Representação geométrica
        ax4 = axes[1, 1]
        theta = np.linspace(0, 2*np.pi, 100)

        # Torus inicial
        R, r = r_majors[0], r_minors[0]
        x_torus = (R + r * np.cos(theta)) * np.cos(3*theta)
        y_torus = (R + r * np.cos(theta)) * np.sin(3*theta)
        ax4.plot(x_torus, y_torus, 'b--', alpha=0.5, label='Torus Inicial')

        # Torus final
        R, r = r_majors[-1], r_minors[-1]
        x_torus = (R + r * np.cos(theta)) * np.cos(3*theta)
        y_torus = (R + r * np.cos(theta)) * np.sin(3*theta)
        ax4.plot(x_torus, y_torus, 'r-', linewidth=2, label='Orus Final')

        # Orus (esfera)
        r_orus = (R * r**2)**(1/3)
        circle = Circle((0, 0), r_orus, fill=False, 
                       color='green', linewidth=2, linestyle=':', label='Orus')
        ax4.add_patch(circle)

        ax4.set_aspect('equal')
        ax4.set_title('Transformação Geométrica')
        ax4.legend()
        ax4.axis('off')

        plt.tight_layout()
        plt.suptitle('Simulação Torus → Orus', fontsize=16, y=1.02)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            return save_path

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        return img_str

    def plot_operador_paraconsistente(self, save_path: str = None) -> str:
        """
        Visualização do operador ⊕ em 3D
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        a = np.linspace(-2, 2, 100)
        b = np.linspace(-2, 2, 100)
        A, B = np.meshgrid(a, b)

        # a ⊕ b = (a + b) / (1 + |ab|)
        Z = (A + B) / (1 + np.abs(A * B))

        surf = ax.plot_surface(A, B, Z, cmap='RdBu_r', alpha=0.9)

        # Linha de ponto fixo
        alpha = 0.047
        x_fix = np.linspace(-1, 1, 50)
        y_fix = -x_fix + alpha
        z_fix = (x_fix + y_fix) / (1 + np.abs(x_fix * y_fix))
        ax.plot(x_fix, y_fix, z_fix, 'k-', linewidth=3, label='Ponto Fixo')

        ax.set_xlabel('a')
        ax.set_ylabel('b')
        ax.set_zlabel('a ⊕ b')
        ax.set_title('Operador Paraconsistente ⊕')

        fig.colorbar(surf, ax=ax, label='a ⊕ b')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            return save_path

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        return img_str

    def plot_triologia_regimes(self, save_path: str = None) -> str:
        """
        Visualização da trialogia: Caos → Ordem → Hiperconsistência
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Caos (trajetória irregular)
        ax1 = axes[0]
        np.random.seed(42)
        t = np.linspace(0, 10, 1000)
        x_caos = np.cumsum(np.random.randn(1000) * 0.1)
        y_caos = np.cumsum(np.random.randn(1000) * 0.1)
        ax1.plot(x_caos, y_caos, color=self.colors['danger'], alpha=0.7, linewidth=0.8)
        ax1.scatter(x_caos[::100], y_caos[::100], c=t[::100], cmap='Reds', s=20)
        ax1.set_title('CAOS (P≠NP)
Viscosidade η→0', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.axis('equal')

        # Ordem (trajetória suave)
        ax2 = axes[1]
        t = np.linspace(0, 4*np.pi, 1000)
        x_ordem = np.sin(t) * np.exp(-t/10)
        y_ordem = np.cos(t) * np.exp(-t/10)
        ax2.plot(x_ordem, y_ordem, color=self.colors['success'], linewidth=2)
        ax2.scatter(x_ordem[::100], y_ordem[::100], c=t[::100], cmap='Greens', s=20)
        ax2.set_title('ORDEM (P→NP)
η Moderada', fontsize=12, fontweight='bold')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.axis('equal')

        # Hiperconsistência (ciclo fechado)
        ax3 = axes[2]
        theta = np.linspace(0, 2*np.pi, 1000)
        r = 1 + 0.1 * np.sin(10*theta)
        x_hiper = r * np.cos(theta)
        y_hiper = r * np.sin(theta)
        ax3.plot(x_hiper, y_hiper, color=self.colors['accent'], linewidth=2)
        ax3.scatter(x_hiper[::100], y_hiper[::100], c=theta[::100], 
                   cmap='Purples', s=30, edgecolors='gold', linewidths=1)
        ax3.set_title('HIPERCONSISTÊNCIA (P=NP*)
η→∞ (Limite Log)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.axis('equal')

        plt.tight_layout()
        plt.suptitle('Trialogia dos Regimes Computacionais', 
                    fontsize=16, fontweight='bold', y=1.05)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            return save_path

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        return img_str

    def plot_arquitetura_stack(self, save_path: str = None) -> str:
        """
        Diagrama da arquitetura em 5 camadas
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')

        camadas = [
            {'nome': '5. Aplicação Social (ReCivitas)', 'cor': '#e8f6f3', 
             'desc': 'RBU via Protocolo Hermes', 'y': 10},
            {'nome': '4. Processamento Metacognitivo', 'cor': '#d5f5e3',
             'desc': 'Operador ⊕, Kernel K(τ,τ')', 'y': 8},
            {'nome': '3. Álgebra Fundamental', 'cor': '#abebc6',
             'desc': 'Clifford Cℓ₄,₁, Spinores Majorana-Weyl', 'y': 6},
            {'nome': '2. Campo Φ-LIBER', 'cor': '#82e0aa',
             'desc': 'Viscosidade η(Φ) = η₀ ln(1+κΦ)', 'y': 4},
            {'nome': '1. Infraestrutura', 'cor': '#58d68d',
             'desc': 'CUDA (block 256-512), precisão mista', 'y': 2},
        ]

        for camada in camadas:
            rect = mpatches.FancyBboxPatch(
                (1, camada['y']-0.8), 8, 1.5,
                boxstyle="round,pad=0.05,rounding_size=0.2",
                facecolor=camada['cor'], edgecolor='black', linewidth=2
            )
            ax.add_patch(rect)
            ax.text(5, camada['y']+0.2, camada['nome'], 
                   ha='center', va='center', fontsize=12, fontweight='bold')
            ax.text(5, camada['y']-0.3, camada['desc'],
                   ha='center', va='center', fontsize=9, style='italic')

        # Setas de fluxo
        for i in range(len(camadas)-1):
            ax.annotate('', xy=(5, camadas[i+1]['y']+0.8), 
                       xytext=(5, camadas[i]['y']-0.8),
                       arrowprops=dict(arrowstyle='<->', color='gray', lw=2))

        ax.set_title('Stack Arquitetural Liber-Eledonte', 
                    fontsize=16, fontweight='bold', pad=20)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            return save_path

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        return img_str


# Função para gerar todas as visualizações
def gerar_todas_visualizacoes(output_dir: str = './diagramas/'):
    """Gera todas as visualizações do sistema"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    viz = CECHVisualizer()

    # Gerar todas as figuras
    viz.plot_phi_liber_surface(f'{output_dir}/01_phi_liber_surface.png')
    viz.plot_operador_paraconsistente(f'{output_dir}/02_operador_paraconsistente.png')
    viz.plot_triologia_regimes(f'{output_dir}/03_trialogia_regimes.png')
    viz.plot_arquitetura_stack(f'{output_dir}/04_arquitetura_stack.png')

    print(f"Diagramas gerados em: {output_dir}")
    return output_dir


if __name__ == "__main__":
    gerar_todas_visualizacoes()
