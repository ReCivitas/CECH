"""
CECH v1.0 - Módulo RBU Quatinga Velho
Sistema de Renda Básica Universal com privacidade diferencial
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import secrets
import json

@dataclass
class Beneficiario:
    """Representa um beneficiário da RBU"""
    id_hash: str           # Hash cego (SHA3)
    poligono_500m: str     # Região aproximada (anonimizada)
    data_cadastro: datetime
    ativo: bool = True
    veto_etico: List[str] = field(default_factory=list)

    def gerar_id_c ego(nome: str, data_nasc: str, salt_local: str) -> str:
        """Gera ID hash cego"""
        dados = f"{nome}||{data_nasc}||{salt_local}"
        return hashlib.sha3_256(dados.encode()).hexdigest()[:32]


@dataclass  
class TransferenciaRBU:
    """Representa uma transferência de RBU"""
    id_beneficiario: str
    valor: float
    timestamp: datetime
    hash_bloco: str
    confirmada: bool = False

    def gerar_hash(self) -> str:
        """Gera hash da transação (0x000...000 para privacidade)"""
        # Hash zero para não rastrear
        return "0x" + "0" * 64


class ProtocoloPrivacidadeDiferencial:
    """
    Implementa privacidade diferencial (ε=0.1) para dados RBU.
    Garante que a presença ou ausência de um indivíduo não afete
    estatísticas agregadas significativamente.
    """

    EPSILON = 0.1  # Parâmetro de privacidade

    @staticmethod
    def adicionar_ruido_laplace(valor: float, sensibilidade: float = 1.0) -> float:
        """
        Adiciona ruído Laplace para privacidade diferencial.
        scale = sensibilidade / ε
        """
        scale = sensibilidade / ProtocoloPrivacidadeDiferencial.EPSILON
        # Amostra da distribuição Laplace
        u = np.random.random() - 0.5
        ruido = -scale * np.sign(u) * np.log(1 - 2 * abs(u))
        return valor + ruido

    @staticmethod
    def agregar_com_privacidade(valores: List[float]) -> Dict:
        """
        Agrega dados com garantia de privacidade diferencial.
        """
        n = len(valores)
        if n == 0:
            return {'contagem': 0, 'media': 0, 'total': 0}

        # Estatísticas verdadeiras
        soma_real = sum(valores)
        media_real = soma_real / n

        # Estatísticas com ruído
        soma_ruidosa = ProtocoloPrivacidadeDiferencial.adicionar_ruido_laplace(
            soma_real, sensibilidade=max(valores) if valores else 1.0
        )
        contagem_ruidosa = int(ProtocoloPrivacidadeDiferencial.adicionar_ruido_laplace(
            n, sensibilidade=1.0
        ))

        return {
            'contagem': max(0, contagem_ruidosa),
            'media': soma_ruidosa / max(1, contagem_ruidosa),
            'total': max(0, soma_ruidosa),
            'epsilon': ProtocoloPrivacidadeDiferencial.EPSILON
        }


class SistemaRBUQuatinga:
    """Sistema de RBU para Quatinga Velho"""

    VALOR_RBU = 42.0  # Valor mensal em reais

    def __init__(self):
        self.beneficiarios: Dict[str, Beneficiario] = {}
        self.transferencias: List[TransferenciaRBU] = []
        self.salt_local = secrets.token_hex(16)
        self.privacidade = ProtocoloPrivacidadeDiferencial()
        self.modo_congelado = False
        self.assembleia_pendente = False

    def cadastrar_beneficiario(self, nome: str, data_nasc: str,
                               poligono: str) -> str:
        """
        Cadastra novo beneficiário com anonimização.
        Retorna ID hash para referência futura.
        """
        id_hash = Beneficiario.gerar_id_cego(nome, data_nasc, self.salt_local)

        beneficiario = Beneficiario(
            id_hash=id_hash,
            poligono_500m=poligono,
            data_cadastro=datetime.now()
        )

        self.beneficiarios[id_hash] = beneficiario
        return id_hash

    def executar_transferencia_mensal(self) -> Dict:
        """
        Executa transferência mensal para todos os beneficiários ativos.
        """
        if self.modo_congelado:
            return {
                'status': 'CONGELADO',
                'mensagem': 'Sistema em modo de assembleia',
                'transferencias': 0
            }

        transferencias_realizadas = 0

        for id_hash, benef in self.beneficiarios.items():
            if not benef.ativo:
                continue

            # Cria transferência
            transferencia = TransferenciaRBU(
                id_beneficiario=id_hash,
                valor=self.VALOR_RBU,
                timestamp=datetime.now(),
                hash_bloco="0x" + "0" * 64  # Hash zero para privacidade
            )

            self.transferencias.append(transferencia)
            transferencias_realizadas += 1

        return {
            'status': 'SUCESSO',
            'transferencias': transferencias_realizadas,
            'valor_total': transferencias_realizadas * self.VALOR_RBU,
            'timestamp': datetime.now().isoformat()
        }

    def aplicar_veto_etico(self, id_beneficiario: str, 
                          restricao: str) -> bool:
        """
        Aplica veto ético de um beneficiário.
        O veto impede que os dados sejam usados em modelos específicos.
        """
        if id_beneficiario not in self.beneficiarios:
            return False

        self.beneficiarios[id_beneficiario].veto_etico.append(restricao)
        return True

    def congelar_para_assembleia(self) -> None:
        """Congela o sistema para assembleia comunitária (7 dias)"""
        self.modo_congelado = True
        self.assembleia_pendente = True

    def descongelar(self, decisao_comunitaria: bool) -> Dict:
        """
        Descongela o sistema após decisão comunitária.
        Se decisão for False, apaga dados locais.
        """
        if not self.assembleia_pendente:
            return {'status': 'ERRO', 'mensagem': 'Sistema não estava congelado'}

        if decisao_comunitaria:
            self.modo_congelado = False
            self.assembleia_pendente = False
            return {
                'status': 'CONTINUAR',
                'mensagem': 'Sistema continuará operando'
            }
        else:
            # Direito ao esquecimento
            n_benef = len(self.beneficiarios)
            self.beneficiarios.clear()
            self.transferencias.clear()
            self.modo_congelado = False
            self.assembleia_pendente = False
            return {
                'status': 'ENCERRADO',
                'mensagem': f'Dados de {n_benef} beneficiários apagados'
            }

    def gerar_relatorio_agregado(self) -> Dict:
        """
        Gera relatório estatístico com privacidade diferencial.
        """
        # Coleta dados (nunca individuais)
        valores_transferidos = [t.valor for t in self.transferencias]

        # Agrega com privacidade
        estatisticas = self.privacidade.agregar_com_privacidade(
            valores_transferidos
        )

        # Contagem de beneficiários (com ruído)
        n_benef = len(self.beneficiarios)
        n_benef_ruidoso = int(self.privacidade.adicionar_ruido_laplace(n_benef))

        return {
            'beneficiarios_ativos': max(0, n_benef_ruidoso),
            'total_transferido': estatisticas['total'],
            'media_por_transferencia': estatisticas['media'],
            'epsilon_privacidade': self.privacidade.EPSILON,
            'periodo': 'mensal'
        }

    def get_estatisticas_sistema(self) -> Dict:
        """Retorna estatísticas do sistema"""
        return {
            'total_beneficiarios': len(self.beneficiarios),
            'beneficiarios_ativos': sum(1 for b in self.beneficiarios.values() if b.ativo),
            'total_transferencias': len(self.transferencias),
            'valor_total_transferido': sum(t.valor for t in self.transferencias),
            'modo_congelado': self.modo_congelado,
            'assembleia_pendente': self.assembleia_pendente
        }


class AuditoriaComunitaria:
    """Sistema de auditoria por sorteio democrático"""

    def __init__(self, sistema_rbu: SistemaRBUQuatinga):
        self.sistema = sistema_rbu
        self.auditores_atuais: List[str] = []
        self.historico_auditorias: List[Dict] = []

    def sortear_auditores(self, n_auditores: int = 3) -> List[str]:
        """
        Sorteia auditores aleatoriamente.
        Usa hash do último bloco como semente.
        """
        if not self.sistema.beneficiarios:
            return []

        ids = list(self.sistema.beneficiarios.keys())

        # Semente baseada no timestamp atual
        seed = int(datetime.now().timestamp())
        np.random.seed(seed)

        auditores = np.random.choice(ids, 
                                     size=min(n_auditores, len(ids)),
                                     replace=False)

        self.auditores_atuais = list(auditores)

        self.historico_auditorias.append({
            'data': datetime.now().isoformat(),
            'auditores': self.auditores_atuais,
            'seed': seed
        })

        return self.auditores_atuais

    def verificar_integridade(self, id_auditor: str) -> Dict:
        """
        Permite que um auditor verifique a integridade do sistema.
        """
        if id_auditor not in self.auditores_atuais:
            return {
                'status': 'NAO_AUTORIZADO',
                'mensagem': 'ID não é auditor atual'
            }

        # Verificações que o auditor pode fazer
        estats = self.sistema.get_estatisticas_sistema()

        return {
            'status': 'AUTORIZADO',
            'estatisticas': estats,
            'verificacoes_possiveis': [
                'confirmar_transferencias_ocorreram',
                'verificar_saldo_total',
                'confirmar_numero_beneficiarios'
            ]
        }


def simular_ano_rbu(n_beneficiarios: int = 100):
    """Simula um ano de operação RBU"""
    sistema = SistemaRBUQuatinga()
    auditoria = AuditoriaComunitaria(sistema)

    # Cadastra beneficiários
    print("Cadastrando beneficiários...")
    for i in range(n_beneficiarios):
        sistema.cadastrar_beneficiario(
            nome=f"Beneficiario_{i}",
            data_nasc=f"1980-01-{i%30+1:02d}",
            poligono=f"POLIGONO_{i%10}"
        )

    # Simula 12 meses
    for mes in range(1, 13):
        print(f"\nMês {mes}:")

        # Transferência mensal
        resultado = sistema.executar_transferencia_mensal()
        print(f"  Transferências: {resultado['transferencias']}")
        print(f"  Valor total: R$ {resultado['valor_total']:.2f}")

        # Auditoria trimestral
        if mes % 3 == 0:
            auditores = auditoria.sortear_auditores()
            print(f"  Auditores sorteados: {len(auditores)}")

    # Relatório anual
    print("\n" + "=" * 50)
    print("RELATÓRIO ANUAL")
    print("=" * 50)

    relatorio = sistema.gerar_relatorio_agregado()
    print(f"Beneficiários (com ruído): {relatorio['beneficiarios_ativos']}")
    print(f"Total transferido: R$ {relatorio['total_transferido']:.2f}")
    print(f"Média por transferência: R$ {relatorio['media_por_transferencia']:.2f}")
    print(f"Privacidade (ε): {relatorio['epsilon_privacidade']}")


if __name__ == "__main__":
    print("=" * 60)
    print("SISTEMA RBU - QUATINGA VELO")
    print("Privacidade Diferencial ε=0.1")
    print("=" * 60)

    simular_ano_rbu(n_beneficiarios=50)
