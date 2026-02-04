"""
CECH v1.0 - Interface de Hardware
Controle de GPIO, sensores e comunicação LoRa
"""

import numpy as np
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass
from enum import Enum
import time
import json

class TipoSensor(Enum):
    """Tipos de sensores suportados"""
    TEMPERATURA = "temperatura"
    UMIDADE = "umidade"
    LUMINOSIDADE = "luminosidade"
    TENSAO = "tensao"
    CORRENTE = "corrente"

@dataclass
class LeituraSensor:
    """Leitura de um sensor"""
    tipo: TipoSensor
    valor: float
    timestamp: float
    unidade: str

    def to_dict(self) -> Dict:
        return {
            'tipo': self.tipo.value,
            'valor': self.valor,
            'timestamp': self.timestamp,
            'unidade': self.unidade
        }


class SimuladorGPIO:
    """Simulador de GPIO para Raspberry Pi"""

    BCM = "BCM"
    BOARD = "BOARD"

    IN = "IN"
    OUT = "OUT"

    HIGH = 1
    LOW = 0

    def __init__(self):
        self.modo = None
        self.pinos: Dict[int, Dict] = {}
        self._callbacks: Dict[int, List[Callable]] = {}

    def setmode(self, modo: str):
        """Define modo de numeração de pinos"""
        self.modo = modo

    def setup(self, pino: int, direcao: str, pull_up_down: str = None):
        """Configura um pino"""
        self.pinos[pino] = {
            'direcao': direcao,
            'estado': self.LOW,
            'pull': pull_up_down
        }

    def output(self, pino: int, estado: int):
        """Define estado de saída"""
        if pino in self.pinos:
            self.pinos[pino]['estado'] = estado

    def input(self, pino: int) -> int:
        """Lê estado de entrada"""
        if pino in self.pinos:
            return self.pinos[pino]['estado']
        return self.LOW

    def cleanup(self):
        """Limpa configuração de pinos"""
        self.pinos.clear()


class GerenciadorEnergia:
    """Gerenciamento de energia solar e bateria"""

    def __init__(self, capacidade_bateria_ah: float = 10.0,
                 tensao_nominal: float = 12.0):
        self.capacidade_ah = capacidade_bateria_ah
        self.tensao = tensao_nominal
        self.capacidade_wh = capacidade_bateria_ah * tensao_nominal

        self.carga_atual_ah = capacidade_bateria_ah * 0.8  # 80% inicial
        self.geracao_solar_w = 0.0
        self.consumo_w = 0.0

        # Histórico
        self.historico: List[Dict] = []

    def simular_geracao_solar(self, irradiancia: float, 
                              eficiencia_painel: float = 0.18) -> float:
        """
        Simula geração solar baseada em irradiância (W/m²).
        Painel típico: 10W (0.06m²)
        """
        area_painel = 0.06  # m²
        self.geracao_solar_w = irradiancia * area_painel * eficiencia_painel
        return self.geracao_solar_w

    def calcular_consumo(self, componentes_ativos: Dict[str, float]) -> float:
        """
        Calcula consumo total.
        componentes: {nome: potencia_w}
        """
        self.consumo_w = sum(componentes_ativos.values())
        return self.consumo_w

    def atualizar_estado(self, dt_horas: float = 1.0) -> Dict:
        """Atualiza estado da bateria"""
        # Energia líquida
        energia_liquida_wh = (self.geracao_solar_w - self.consumo_w) * dt_horas

        # Converte para Ah
        delta_ah = energia_liquida_wh / self.tensao
        self.carga_atual_ah = np.clip(
            self.carga_atual_ah + delta_ah,
            0,
            self.capacidade_ah
        )

        estado = {
            'timestamp': time.time(),
            'carga_ah': self.carga_atual_ah,
            'carga_percentual': 100 * self.carga_atual_ah / self.capacidade_ah,
            'geracao_solar_w': self.geracao_solar_w,
            'consumo_w': self.consumo_w,
            'autonomia_horas': self._calcular_autonomia()
        }

        self.historico.append(estado)
        return estado

    def _calcular_autonomia(self) -> float:
        """Calcula autonomia em horas"""
        if self.consumo_w <= 0:
            return float('inf')
        energia_restante_wh = self.carga_atual_ah * self.tensao
        return energia_restante_wh / self.consumo_w

    def modo_economia(self) -> bool:
        """Retorna True se deve entrar em modo de economia"""
        return self.carga_atual_ah / self.capacidade_ah < 0.2


class InterfaceLoRa:
    """Interface de comunicação LoRa"""

    def __init__(self, frequencia_mhz: float = 915.0,
                 potencia_dbm: int = 14):
        self.frequencia = frequencia_mhz
        self.potencia = potencia_dbm
        self.mensagens_enviadas = 0
        self.mensagens_recebidas = 0
        self.buffer: List[Dict] = []

    def enviar(self, dados: Dict, destino: str = "broadcast") -> bool:
        """Simula envio de mensagem LoRa"""
        mensagem = {
            'tipo': 'lora',
            'destino': destino,
            'dados': dados,
            'timestamp': time.time(),
            'tamanho_bytes': len(json.dumps(dados).encode())
        }

        self.buffer.append(mensagem)
        self.mensagens_enviadas += 1

        # Simula latência
        time.sleep(0.01)
        return True

    def receber(self, timeout_ms: int = 1000) -> Optional[Dict]:
        """Simula recebimento de mensagem"""
        if self.buffer:
            msg = self.buffer.pop(0)
            self.mensagens_recebidas += 1
            return msg
        return None

    def calcular_alcance_teorico(self) -> float:
        """Calcula alcance teórico em km"""
        # Simplificado: potência em dBm → km aproximado
        return 10 ** ((self.potencia - 100) / 20)


class NodeEledonte:
    """Nodo completo Eledonte 1-E (embarcado)"""

    def __init__(self, node_id: str):
        self.id = node_id
        self.gpio = SimuladorGPIO()
        self.energia = GerenciadorEnergia()
        self.lora = InterfaceLoRa()
        self.sensores: Dict[TipoSensor, LeituraSensor] = {}

        self.estado = "inicializando"
        self.tarefas_agendadas: List[Dict] = []

    def inicializar(self):
        """Inicializa hardware"""
        self.gpio.setmode(self.gpio.BCM)

        # Configura pinos
        self.gpio.setup(17, self.gpio.OUT)  # LED status
        self.gpio.setup(27, self.gpio.IN)   # Sensor
        self.gpio.setup(22, self.gpio.OUT)  # LoRa CS

        self.estado = "pronto"
        return True

    def ler_sensores(self) -> Dict:
        """Lê todos os sensores"""
        timestamp = time.time()

        # Simula leituras
        self.sensores[TipoSensor.TEMPERATURA] = LeituraSensor(
            tipo=TipoSensor.TEMPERATURA,
            valor=25.0 + np.random.randn() * 2,
            timestamp=timestamp,
            unidade="C"
        )

        self.sensores[TipoSensor.UMIDADE] = LeituraSensor(
            tipo=TipoSensor.UMIDADE,
            valor=60.0 + np.random.randn() * 5,
            timestamp=timestamp,
            unidade="%"
        )

        self.sensores[TipoSensor.TENSAO] = LeituraSensor(
            tipo=TipoSensor.TENSAO,
            valor=self.energia.tensao,
            timestamp=timestamp,
            unidade="V"
        )

        self.sensores[TipoSensor.CORRENTE] = LeituraSensor(
            tipo=TipoSensor.CORRENTE,
            valor=self.energia.consumo_w / self.energia.tensao,
            timestamp=timestamp,
            unidade="A"
        )

        return {k.value: v.to_dict() for k, v in self.sensores.items()}

    def executar_ciclo(self) -> Dict:
        """Executa um ciclo completo de operação"""
        # Lê sensores
        dados_sensores = self.ler_sensores()

        # Atualiza energia
        self.energia.simular_geracao_solar(irradiancia=800)
        self.energia.calcular_consumo({
            'raspberry_pi': 5.0,
            'lora': 0.1,
            'sensores': 0.5
        })
        estado_energia = self.energia.atualizar_estado()

        # Transmite dados
        pacote = {
            'node_id': self.id,
            'sensores': dados_sensores,
            'energia': estado_energia,
            'estado': self.estado
        }
        self.lora.enviar(pacote)

        # LED status
        if self.energia.modo_economia():
            self.gpio.output(17, self.gpio.LOW)
        else:
            self.gpio.output(17, self.gpio.HIGH)

        return pacote

    def shutdown(self):
        """Desliga o nodo de forma segura"""
        self.estado = "desligando"
        self.gpio.cleanup()
        self.estado = "desligado"


def simular_rede_nodes(n_nodes: int = 3, n_ciclos: int = 24):
    """Simula uma rede de nodes Eledonte"""
    print("=" * 60)
    print("SIMULAÇÃO DE REDE ELEDONTE")
    print("=" * 60)

    nodes = [NodeEledonte(f"NODE_{i:03d}") for i in range(n_nodes)]

    # Inicializa
    for node in nodes:
        node.inicializar()

    # Executa ciclos
    for ciclo in range(n_ciclos):
        print(f"\nCiclo {ciclo + 1}:")

        for node in nodes:
            resultado = node.executar_ciclo()
            energia = resultado['energia']
            print(f"  {node.id}: {energia['carga_percentual']:.1f}% "
                  f"({energia['consumo_w']:.1f}W)")

    # Estatísticas
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS FINAIS")
    print("=" * 60)

    for node in nodes:
        print(f"\n{node.id}:")
        print(f"  Mensagens LoRa: {node.lora.mensagens_enviadas}")
        print(f"  Carga final: {node.energia.carga_atual_ah:.2f}Ah")
        print(f"  Autonomia: {node.energia._calcular_autonomia():.1f}h")
        node.shutdown()


if __name__ == "__main__":
    simular_rede_nodes(n_nodes=3, n_ciclos=12)
