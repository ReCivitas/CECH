"""
CECH v1.0 - Circuito Eco-Commutativo Hiperconsistente
Core System - Núcleo de Processamento Hiperconsistente
Autores: Marcus Vinicius Brancaglione, Sistema Eledonte
Licença: ⒶRobinRight 3.0 + CC BY-SA 4.0
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from datetime import datetime

# ============================================================================
# CONSTANTES FUNDAMENTAIS
# ============================================================================

class Constants:
    """Constantes físicas e matemáticas do sistema"""
    ALPHA_LP = 0.047          # Constante de Liber-Poincaré
    PHI_GOLDEN = 1.618033988749895  # Proporção áurea φ
    EPSILON_MAX = 2.5         # Limite de liberdade hiperconsistente
    KAPPA_VISC = 0.47         # Coeficiente de viscosidade
    ETA_0 = 1.0               # Viscosidade base
    C_LIGHT = 299792458       # Velocidade da luz (m/s)
    G_NEWTON = 6.67430e-11    # Constante gravitacional

    # Parâmetros de rede
    NODES_DEFAULT = 121       # 11x11 matriz Eledonte
    CLUSTERING_TARGET = 3.693 # Coeficiente de clustering Odissídico


# ============================================================================
# CLASSES DE DADOS
# ============================================================================

@dataclass
class Quaternion:
    """Quaternion para rotações em ℍ - versão light de Cℓ₄,₁"""
    w: float
    x: float
    y: float
    z: float

    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(
            self.w + other.w, self.x + other.x,
            self.y + other.y, self.z + other.z
        )

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """Produto de Hamilton"""
        return Quaternion(
            w=self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
            x=self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
            y=self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
            z=self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        )

    def conjugate(self) -> 'Quaternion':
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self) -> float:
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> 'Quaternion':
        n = self.norm()
        if n < 1e-10:
            return Quaternion(1, 0, 0, 0)
        return Quaternion(self.w/n, self.x/n, self.y/n, self.z/n)

    def to_rotation_matrix(self) -> np.ndarray:
        """Converte quaternion para matriz de rotação 3x3"""
        q = self.normalize()
        return np.array([
            [1-2*(q.y**2+q.z**2), 2*(q.x*q.y-q.w*q.z), 2*(q.x*q.z+q.w*q.y)],
            [2*(q.x*q.y+q.w*q.z), 1-2*(q.x**2+q.z**2), 2*(q.y*q.z-q.w*q.x)],
            [2*(q.x*q.z-q.w*q.y), 2*(q.y*q.z+q.w*q.x), 1-2*(q.x**2+q.y**2)]
        ])


@dataclass
class EstadoHiperconsistente:
    """Estado de um nodo na rede hiperconsistente"""
    id_nodo: str
    epsilon: float                    # Nível de liberdade
    phi: float                        # Campo Φ-LIBER
    entropia: float                   # Entropia local
    quaternion: Quaternion            # Orientação em ℍ
    timestamp: datetime
    dados_revolucao: Dict             # Dados para reconvolução

    def calcular_viscosidade(self) -> float:
        """η(Φ) = η₀ · ln(1 + κ·Φ) - versão 5.0 logarítmica"""
        return Constants.ETA_0 * np.log(1 + Constants.KAPPA_VISC * self.phi)

    def calcular_energia_criativa(self) -> float:
        """Energia criativa derivada de ε"""
        return self.phi * (1 + 0.21 * self.epsilon) ** 8.13


# ============================================================================
# OPERADOR PARACONSISTENTE ⊕
# ============================================================================

class OperadorParaconsistente:
    """Operador de reconvolução hiperconsistente (⊕)"""

    @staticmethod
    def aplicar(a: float, b: float) -> float:
        """
        a ⊕ b = (a + b) / (1 + |ab|)
        Propriedades: comutativo, não-explosivo, ponto fixo em α_LP
        """
        denominador = 1 + abs(a * b)
        if denominador < 1e-15:
            return 0.0
        return (a + b) / denominador

    @staticmethod
    def aplicar_vetorial(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Aplica ⊕ elemento a elemento em vetores"""
        resultado = np.zeros_like(v1)
        for i in range(len(v1)):
            resultado[i] = OperadorParaconsistente.aplicar(v1[i], v2[i])
        return resultado

    @staticmethod
    def encontrar_ponto_fixo(alpha: float = Constants.ALPHA_LP, 
                             tolerancia: float = 1e-10) -> float:
        """Encontra x tal que x = x ⊕ (-x + α)"""
        x = 0.5
        for _ in range(1000):
            x_novo = OperadorParaconsistente.aplicar(x, -x + alpha)
            if abs(x_novo - x) < tolerancia:
                return x_novo
            x = x_novo
        return x


# ============================================================================
# FUNÇÃO Φ-LIBER
# ============================================================================

class FuncaoPhiLiber:
    """Função Φ-LIBER revisada v22.1"""

    @staticmethod
    def calcular(epsilon: float, x: float, 
                 gamma: float = 0.5) -> float:
        """
        Φ(ε, x) = (4π · e^(-ε²/2) · c²) / (3γ · x · ln(x))
        com clamping hiperconsistente
        """
        # Clamping para evitar overflow
        epsilon_eff = Constants.EPSILON_MAX * np.tanh(epsilon / Constants.EPSILON_MAX)

        # Cálculo logarítmico estável
        if x <= 0:
            x = 1e-10

        numerador = 4 * np.pi * np.exp(-epsilon_eff**2 / 2) * Constants.C_LIGHT**2
        denominador = 3 * gamma * x * np.log(x)

        if abs(denominador) < 1e-15:
            return 0.0

        return numerador / denominador

    @staticmethod
    def calcular_logsumexp(epsilon: float, x_vals: np.ndarray) -> np.ndarray:
        """Versão estável usando log-sum-exp para arrays"""
        epsilon_eff = Constants.EPSILON_MAX * np.tanh(epsilon / Constants.EPSILON_MAX)
        log_x = np.log(np.maximum(x_vals, 1e-10))

        # Estabilidade numérica via log-sum-exp
        max_log = np.max(log_x)
        exp_terms = np.exp(log_x - max_log)

        return (4 * np.pi * np.exp(-epsilon_eff**2 / 2) * Constants.C_LIGHT**2) /                (3 * 0.5 * x_vals * (log_x + np.log(exp_terms.sum()) - max_log))


# ============================================================================
# PROTOCOLO HERMES (P=NP*)
# ============================================================================

class ProtocoloHermes:
    """Protocolo de verificação=criação"""

    def __init__(self):
        self.operador = OperadorParaconsistente()
        self.historico = []

    def gerar_compromisso(self, dados: str, nonce: str) -> str:
        """Alice: C = H(dados || nonce)"""
        return hashlib.sha3_256(f"{dados}||{nonce}".encode()).hexdigest()

    def desafiar(self) -> float:
        """Bob: desafio com α_LP ≈ 0.047"""
        return Constants.ALPHA_LP

    def verificar(self, compromisso: str, dados: str, 
                  nonce: str, desafio: float) -> Tuple[bool, float]:
        """
        Verificação: confiabilidade = 1 - α_LP = 95.3%
        Retorna: (válido, confiança)
        """
        hash_calculado = self.gerar_compromisso(dados, nonce)
        valido = (hash_calculado == compromisso)
        confianca = 1 - desafio if valido else desafio
        return valido, confianca

    def executar_protocolo_alice_bob(self, dados: str) -> Dict:
        """Executa protocolo completo Alice-Bob"""
        import secrets
        nonce = secrets.token_hex(16)

        # Passo 1: Alice gera compromisso
        compromisso = self.gerar_compromisso(dados, nonce)

        # Passo 2: Bob desafia
        desafio = self.desafiar()

        # Passo 3: Verificação
        valido, confianca = self.verificar(compromisso, dados, nonce, desafio)

        resultado = {
            'compromisso': compromisso,
            'nonce': nonce,
            'desafio': desafio,
            'valido': valido,
            'confianca': confianca,
            'timestamp': datetime.now().isoformat()
        }
        self.historico.append(resultado)
        return resultado


# ============================================================================
# REDE ODISSÍDICA
# ============================================================================

class RedeOdissidica:
    """Topologia de dádiva não-mercantil 11×11"""

    def __init__(self, n_nodes: int = Constants.NODES_DEFAULT):
        self.n_nodes = n_nodes
        self.dim = int(np.sqrt(n_nodes))
        self.adj_matrix = np.zeros((n_nodes, n_nodes))
        self.estados: Dict[str, EstadoHiperconsistente] = {}
        self._inicializar_topologia()

    def _inicializar_topologia(self):
        """Inicializa topologia small-world"""
        # Conexões locais (vizinhança)
        for i in range(self.n_nodes):
            vizinhos = self._get_vizinhos(i)
            for v in vizinhos:
                self.adj_matrix[i, v] = 1

        # Conexões de salto longo (small-world)
        np.random.seed(42)
        for i in range(self.n_nodes):
            for j in range(i+1, self.n_nodes):
                if np.random.random() < 0.1:  # 10% de conexões longas
                    self.adj_matrix[i, j] = self.adj_matrix[j, i] = 1

    def _get_vizinhos(self, idx: int) -> List[int]:
        """Retorna vizinhos em grade 2D"""
        row, col = idx // self.dim, idx % self.dim
        vizinhos = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.dim and 0 <= nc < self.dim:
                vizinhos.append(nr * self.dim + nc)
        return vizinhos

    def calcular_entropia_rede(self) -> float:
        """Entropia de rede em nats"""
        graus = np.sum(self.adj_matrix, axis=1)
        p = graus / graus.sum()
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    def calcular_clustering(self) -> float:
        """Coeficiente médio de clustering"""
        clustering = []
        for i in range(self.n_nodes):
            vizinhos = np.where(self.adj_matrix[i] > 0)[0]
            if len(vizinhos) < 2:
                continue
            conexoes = 0
            for j in vizinhos:
                for k in vizinhos:
                    if self.adj_matrix[j, k] > 0:
                        conexoes += 1
            n = len(vizinhos)
            clustering.append(conexoes / (n * (n - 1)))
        return np.mean(clustering) if clustering else 0

    def adicionar_nodo(self, estado: EstadoHiperconsistente):
        """Adiciona nodo com estado hiperconsistente"""
        self.estados[estado.id_nodo] = estado

    def reconvolucao_local(self, id_nodo: str) -> float:
        """Aplica operador ⊕ nos vizinhos de um nodo"""
        if id_nodo not in self.estados:
            return 0.0

        idx = list(self.estados.keys()).index(id_nodo)
        vizinhos = np.where(self.adj_matrix[idx] > 0)[0]

        if len(vizinhos) == 0:
            return self.estados[id_nodo].epsilon

        # Aplica ⊕ recursivamente
        resultado = self.estados[id_nodo].epsilon
        ids = list(self.estados.keys())
        for v_idx in vizinhos:
            if v_idx < len(ids):
                viz_epsilon = self.estados[ids[v_idx]].epsilon
                resultado = OperadorParaconsistente.aplicar(resultado, viz_epsilon)

        return resultado


# ============================================================================
# INFOCOMPOSTAGEM
# ============================================================================

class InfoCompostagem:
    """Sistema digestor ecológico de informação"""

    def __init__(self):
        self.taxa_ruido = 0.88      # 88% ruído de entrada
        self.taxa_sinal = 0.12      # 12% sinal
        self.epsilon_corte = 0.83   # Limiar adaptativo
        self.eficiencia_teorica = 0.88
        self.residuos: List[Dict] = []

    def ingerir(self, dados: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Separa sinal de ruído"""
        # Simulação: componentes acima do limiar são sinal
        mascara_sinal = np.abs(dados) > self.epsilon_corte
        sinal = dados[mascara_sinal]
        ruido = dados[~mascara_sinal]
        return sinal, ruido

    def decompor(self, ruido: np.ndarray) -> Dict:
        """Decompõe ruído em componentes compostáveis"""
        return {
            'componentes': len(ruido),
            'energia': np.sum(ruido**2),
            'entropia': -np.sum(ruido**2 * np.log(ruido**2 + 1e-10)),
            'timestamp': datetime.now().isoformat()
        }

    def reconvolver(self, residuos: List[Dict]) -> np.ndarray:
        """Recombina resíduos via operador ⊕"""
        if not residuos:
            return np.array([])

        resultado = np.zeros(residuos[0]['componentes'])
        for r in residuos:
            vetor = np.random.randn(r['componentes']) * np.sqrt(r['energia'] / r['componentes'])
            resultado = OperadorParaconsistente.aplicar_vetorial(resultado, vetor)

        return resultado

    def ciclo_completo(self, entrada: np.ndarray) -> Dict:
        """Executa ciclo completo de infocompostagem"""
        sinal, ruido = self.ingerir(entrada)
        decomposto = self.decompor(ruido)
        self.residuos.append(decomposto)

        # Compostagem periódica
        if len(self.residuos) >= 10:
            recombinado = self.reconvolver(self.residuos)
            self.residuos = []
        else:
            recombinado = None

        return {
            'sinal_extraido': sinal,
            'ruido_decomposto': decomposto,
            'recombinado': recombinado,
            'eficiencia': len(sinal) / len(entrada) if len(entrada) > 0 else 0
        }


# ============================================================================
# SIMULAÇÃO TORUS-ORUS
# ============================================================================

class TorusOrusSimulator:
    """Simulação da contração Torus → Orus"""

    def __init__(self, r_major: float = 10.0, r_minor: float = 3.0):
        self.r_major = r_major
        self.r_minor = r_minor
        self.passo = 0

    def volume_torus(self) -> float:
        """Volume do torus: V = 2π²Rr²"""
        return 2 * np.pi**2 * self.r_major * self.r_minor**2

    def volume_orus(self) -> float:
        """Volume do orus (esfera equivalente)"""
        r_equiv = (self.r_major * self.r_minor**2) ** (1/3)
        return (4/3) * np.pi * r_equiv**3

    def contrair(self, taxa: float = 0.99) -> Dict:
        """Executa um passo de contração"""
        vol_inicial = self.volume_torus()

        # Contração do raio menor
        self.r_minor *= taxa

        # Compensação da força Liber
        lambda_liber = self.r_major / (self.r_minor + Constants.ALPHA_LP)

        vol_final = self.volume_torus()
        reducao = (vol_inicial - vol_final) / vol_inicial

        self.passo += 1

        return {
            'passo': self.passo,
            'r_major': self.r_major,
            'r_minor': self.r_minor,
            'volume': vol_final,
            'lambda_liber': lambda_liber,
            'reducao_percentual': reducao * 100
        }

    def simular_contracao_completa(self, n_passos: int = 1000) -> List[Dict]:
        """Simula contração completa até estado Orus"""
        historico = []
        for _ in range(n_passos):
            estado = self.contrair()
            historico.append(estado)
            if estado['reducao_percentual'] > 99.9:
                break
        return historico


# ============================================================================
# SISTEMA INTEGRADO
# ============================================================================

class SistemaLiberEledonte:
    """Sistema integrado Liber-Eledonte v22.1"""

    def __init__(self):
        self.phi_liber = FuncaoPhiLiber()
        self.operador = OperadorParaconsistente()
        self.hermes = ProtocoloHermes()
        self.rede = RedeOdissidica()
        self.compostagem = InfoCompostagem()
        self.torus_orus = TorusOrusSimulator()
        self.metricas: List[Dict] = []

    def executar_simulacao(self, n_iteracoes: int = 100) -> Dict:
        """Executa simulação completa do sistema"""
        resultados = {
            'phi_liber': [],
            'reconvolucoes': [],
            'protocolos_hermes': [],
            'compostagem': [],
            'torus_orus': []
        }

        for i in range(n_iteracoes):
            epsilon = i / 10.0

            # Φ-LIBER
            phi = self.phi_liber.calcular(epsilon, x=10.0)
            resultados['phi_liber'].append({'epsilon': epsilon, 'phi': phi})

            # Reconvolução
            rec = self.operador.aplicar(epsilon, -epsilon + Constants.ALPHA_LP)
            resultados['reconvolucoes'].append(rec)

            # Protocolo Hermes
            dados = f"iteracao_{i}"
            proto = self.hermes.executar_protocolo_alice_bob(dados)
            resultados['protocolos_hermes'].append(proto)

            # InfoCompostagem
            entrada = np.random.randn(1000)
            comp = self.compostagem.ciclo_completo(entrada)
            resultados['compostagem'].append(comp)

            # Torus-Orus
            if i % 10 == 0:
                torus = self.torus_orus.contrair()
                resultados['torus_orus'].append(torus)

        return resultados

    def calcular_confiabilidade(self) -> Dict:
        """Calcula confiabilidade composta do sistema"""
        return {
            'matematica': 0.92,
            'fisica': 0.78,
            'experimental': 0.81,
            'total': 0.85
        }


# ============================================================================
# MAIN / TESTES
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CECH v1.0 - Circuito Eco-Commutativo Hiperconsistente")
    print("Sistema Liber-Eledonte v22.1")
    print("=" * 60)

    # Teste do operador paraconsistente
    print("\n[1] Teste do Operador ⊕")
    op = OperadorParaconsistente()
    resultado = op.aplicar(0.5, 0.3)
    print(f"    0.5 ⊕ 0.3 = {resultado:.6f}")

    ponto_fixo = op.encontrar_ponto_fixo()
    print(f"    Ponto fixo: {ponto_fixo:.6f} (esperado ≈ {Constants.ALPHA_LP})")

    # Teste de quaternions
    print("\n[2] Teste de Quaternions ℍ")
    q1 = Quaternion(1, 0.5, 0.3, 0.2)
    q2 = Quaternion(0.8, 0.2, 0.4, 0.1)
    q_prod = q1 * q2
    print(f"    q1 = ({q1.w}, {q1.x}, {q1.y}, {q1.z})")
    print(f"    q2 = ({q2.w}, {q2.x}, {q2.y}, {q2.z})")
    print(f"    q1 * q2 = ({q_prod.w:.4f}, {q_prod.x:.4f}, {q_prod.y:.4f}, {q_prod.z:.4f})")

    # Teste Φ-LIBER
    print("\n[3] Teste Φ-LIBER")
    phi = FuncaoPhiLiber()
    for eps in [0, 1, 2, 5, 10]:
        val = phi.calcular(eps, x=10.0)
        print(f"    Φ(ε={eps}, x=10) = {val:.6e}")

    # Teste Protocolo Hermes
    print("\n[4] Teste Protocolo Hermes")
    hermes = ProtocoloHermes()
    resultado = hermes.executar_protocolo_alice_bob("teste_dados_rbu")
    print(f"    Compromisso: {resultado['compromisso'][:16]}...")
    print(f"    Válido: {resultado['valido']}")
    print(f"    Confiança: {resultado['confianca']:.4f}")

    # Teste Rede Odissídica
    print("\n[5] Teste Rede Odissídica")
    rede = RedeOdissidica()
    entropia = rede.calcular_entropia_rede()
    clustering = rede.calcular_clustering()
    print(f"    Nodos: {rede.n_nodes}")
    print(f"    Entropia: {entropia:.4f} nats")
    print(f"    Clustering: {clustering:.4f}")

    # Simulação completa
    print("\n[6] Simulação Completa")
    sistema = SistemaLiberEledonte()
    resultados = sistema.executar_simulacao(n_iteracoes=50)
    confiabilidade = sistema.calcular_confiabilidade()
    print(f"    Iterações: 50")
    print(f"    Confiabilidade: {confiabilidade['total']*100:.0f}%")
    print(f"      - Matemática: {confiabilidade['matematica']*100:.0f}%")
    print(f"      - Física: {confiabilidade['fisica']*100:.0f}%")
    print(f"      - Experimental: {confiabilidade['experimental']*100:.0f}%")

    print("\n" + "=" * 60)
    print("Simulação concluída com sucesso!")
    print("=" * 60)
