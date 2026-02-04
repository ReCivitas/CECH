"""
CECH v1.0 - Solver K-SAT Paraconsistente
Estratégia K-SAT Compostável à Prova de Falha
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import random

class StatusSAT(Enum):
    """Status da resolução K-SAT"""
    SAT = "satisfazível"
    UNSAT = "insatisfazível"
    UNDEFINED = "indefinido"
    PARTIAL = "parcial"

@dataclass
class InstanciaKSAT:
    """Representa uma instância K-SAT"""
    n_variaveis: int
    n_clausulas: int
    k: int  # Tamanho de cada cláusula
    clausulas: List[List[int]]  # Cada cláusula é lista de literais

    def __post_init__(self):
        self.atribuicao = [None] * self.n_variaveis

    def avaliar_clausula(self, clausula: List[int], 
                         atribuicao: List[Optional[bool]]) -> Optional[bool]:
        """Avalia uma cláusula dada uma atribuição parcial"""
        valores = []
        for literal in clausula:
            var_idx = abs(literal) - 1
            if atribuicao[var_idx] is None:
                return None  # Indeterminado
            valor = atribuicao[var_idx] if literal > 0 else not atribuicao[var_idx]
            valores.append(valor)
        return any(valores) if valores else None

    def avaliar(self, atribuicao: List[Optional[bool]]) -> Tuple[StatusSAT, float]:
        """
        Avalia a fórmula completa.
        Retorna: (status, confiança)
        """
        satisfeitas = 0
        indefinidas = 0

        for clausula in self.clausulas:
            resultado = self.avaliar_clausula(clausula, atribuicao)
            if resultado is None:
                indefinidas += 1
            elif resultado:
                satisfeitas += 1

        if indefinidas > 0:
            confianca = satisfeitas / (self.n_clausulas - indefinidas) if (self.n_clausulas - indefinidas) > 0 else 0
            return StatusSAT.PARTIAL, confianca

        if satisfeitas == self.n_clausulas:
            return StatusSAT.SAT, 1.0

        return StatusSAT.UNSAT, satisfeitas / self.n_clausulas


@dataclass
class ResiduoComputacional:
    """Resíduo de tentativa falha de resolução"""
    atribuicao_parcial: List[Optional[bool]]
    variaveis_livres: List[int]
    clausulas_pendentes: List[int]
    confianca: float
    profundidade: int

    def tamanho(self) -> int:
        return len(self.variaveis_livres)


class OperadorReconvolucaoKSAT:
    """Operador ⊕ para combinar resíduos K-SAT"""

    ALPHA_LP = 0.047

    @staticmethod
    def aplicar(r1: ResiduoComputacional, 
                r2: ResiduoComputacional) -> ResiduoComputacional:
        """
        Combina dois resíduos via operador ⊕.
        Mescla atribuições parciais compatíveis.
        """
        n = len(r1.atribuicao_parcial)
        atribuicao_mesclada = [None] * n

        for i in range(n):
            a1, a2 = r1.atribuicao_parcial[i], r2.atribuicao_parcial[i]
            if a1 is not None and a2 is not None:
                # Conflito: usa operador ⊕ para decidir
                v1, v2 = float(a1), float(a2)
                resultado = (v1 + v2) / (1 + abs(v1 * v2))
                atribuicao_mesclada[i] = resultado > 0.5 if resultado != 0.5 else random.choice([True, False])
            elif a1 is not None:
                atribuicao_mesclada[i] = a1
            elif a2 is not None:
                atribuicao_mesclada[i] = a2

        # Variáveis livres = interseção
        vars_livres = list(set(r1.variaveis_livres) & set(r2.variaveis_livres))

        # Clausulas pendentes = união
        clausulas = list(set(r1.clausulas_pendentes) | set(r2.clausulas_pendentes))

        # Confiança combinada via ⊕
        c1, c2 = r1.confianca, r2.confianca
        confianca = (c1 + c2) / (1 + abs(c1 * c2))

        return ResiduoComputacional(
            atribuicao_parcial=atribuicao_mesclada,
            variaveis_livres=vars_livres,
            clausulas_pendentes=clausulas,
            confianca=confianca,
            profundidade=max(r1.profundidade, r2.profundidade) + 1
        )


class SolverKSATParaconsistente:
    """Solver K-SAT com backtracking paraconsistente"""

    def __init__(self, instancia: InstanciaKSAT,
                 epsilon_corte: float = 0.83,
                 max_profundidade: int = 100):
        self.instancia = instancia
        self.epsilon_corte = epsilon_corte
        self.max_profundidade = max_profundidade
        self.residuos: List[ResiduoComputacional] = []
        self.nos_explorados = 0
        self.operador = OperadorReconvolucaoKSAT()

    def resolver(self) -> Tuple[StatusSAT, Optional[List[bool]], float]:
        """
        Resolve a instância K-SAT.
        Retorna: (status, atribuição, confiança)
        """
        atribuicao_inicial = [None] * self.instancia.n_variaveis
        resultado = self._backtrack(atribuicao_inicial, 0, [])

        if resultado[0] == StatusSAT.SAT:
            return resultado

        # Se falhou, tenta compostagem de resíduos
        if self.residuos:
            return self._tentar_compostagem()

        return resultado

    def _backtrack(self, atribuicao: List[Optional[bool]], 
                   profundidade: int,
                   clausulas_sat: List[int]) -> Tuple[StatusSAT, Optional[List[bool]], float]:
        """Backtracking com poda paraconsistente"""
        self.nos_explorados += 1

        if profundidade > self.max_profundidade:
            # Salva resíduo
            vars_livres = [i for i, a in enumerate(atribuicao) if a is None]
            clausulas_pend = [i for i in range(self.instancia.n_clausulas) 
                            if i not in clausulas_sat]

            residuo = ResiduoComputacional(
                atribuicao_parcial=atribuicao.copy(),
                variaveis_livres=vars_livres,
                clausulas_pendentes=clausulas_pend,
                confianca=0.5,
                profundidade=profundidade
            )
            self.residuos.append(residuo)
            return StatusSAT.UNDEFINED, None, 0.5

        # Verifica estado atual
        status, confianca = self.instancia.avaliar(atribuicao)

        if status == StatusSAT.SAT:
            return StatusSAT.SAT, [a if a is not None else False for a in atribuicao], 1.0

        if status == StatusSAT.UNSAT and confianca < 0.3:
            # Falha graciosa - salva resíduo
            vars_livres = [i for i, a in enumerate(atribuicao) if a is None]
            clausulas_pend = list(range(self.instancia.n_clausulas))

            residuo = ResiduoComputacional(
                atribuicao_parcial=atribuicao.copy(),
                variaveis_livres=vars_livres,
                clausulas_pendentes=clausulas_pend,
                confianca=confianca,
                profundidade=profundidade
            )
            self.residuos.append(residuo)
            return StatusSAT.UNDEFINED, None, confianca

        # Escolhe próxima variável (heurística: mais frequente)
        var_idx = self._escolher_variavel(atribuicao)
        if var_idx is None:
            return status, None, confianca

        # Tenta True
        atribuicao[var_idx] = True
        resultado = self._backtrack(atribuicao, profundidade + 1, clausulas_sat)
        if resultado[0] == StatusSAT.SAT:
            return resultado

        # Tenta False
        atribuicao[var_idx] = False
        resultado = self._backtrack(atribuicao, profundidade + 1, clausulas_sat)
        if resultado[0] == StatusSAT.SAT:
            return resultado

        # Backtrack
        atribuicao[var_idx] = None
        return StatusSAT.UNSAT, None, confianca

    def _escolher_variavel(self, atribuicao: List[Optional[bool]]) -> Optional[int]:
        """Heurística de escolha de variável"""
        vars_livres = [i for i, a in enumerate(atribuicao) if a is None]
        if not vars_livres:
            return None

        # Conta frequência em cláusulas não satisfeitas
        frequencias = {v: 0 for v in vars_livres}
        for clausula in self.instancia.clausulas:
            for lit in clausula:
                var = abs(lit) - 1
                if var in frequencias:
                    frequencias[var] += 1

        return max(frequencias, key=frequencias.get)

    def _tentar_compostagem(self) -> Tuple[StatusSAT, Optional[List[bool]], float]:
        """Tenta resolver via compostagem de resíduos"""
        if len(self.residuos) < 2:
            return StatusSAT.UNSAT, None, 0.0

        # Combina resíduos via ⊕
        combinado = self.residuos[0]
        for residuo in self.residuos[1:]:
            combinado = self.operador.aplicar(combinado, residuo)

        # Tenta resolver com atribuição combinada
        status, confianca = self.instancia.avaliar(combinado.atribuicao_parcial)

        if status == StatusSAT.SAT:
            return StatusSAT.SAT, [a if a is not None else False 
                                   for a in combinado.atribuicao_parcial], confianca

        if status == StatusSAT.PARTIAL and confianca > self.epsilon_corte:
            return StatusSAT.PARTIAL, combinado.atribuicao_parcial, confianca

        return StatusSAT.UNDEFINED, combinado.atribuicao_parcial, combinado.confianca

    def get_estatisticas(self) -> Dict:
        """Retorna estatísticas da resolução"""
        return {
            'nos_explorados': self.nos_explorados,
            'residuos_gerados': len(self.residuos),
            'taxa_reaproveitamento': len(self.residuos) / max(self.nos_explorados, 1)
        }


def gerar_instancia_aleatoria(n_vars: int, n_clausulas: int, 
                               k: int = 3) -> InstanciaKSAT:
    """Gera instância K-SAT aleatória"""
    clausulas = []
    for _ in range(n_clausulas):
        vars_clausula = random.sample(range(1, n_vars + 1), k)
        clausula = [v if random.random() > 0.5 else -v for v in vars_clausula]
        clausulas.append(clausula)

    return InstanciaKSAT(
        n_variaveis=n_vars,
        n_clausulas=n_clausulas,
        k=k,
        clausulas=clausulas
    )


def benchmark_solver(n_testes: int = 100):
    """Benchmark do solver paraconsistente"""
    resultados = {
        'SAT': 0, 'UNSAT': 0, 'UNDEFINED': 0, 'PARTIAL': 0
    }

    for _ in range(n_testes):
        instancia = gerar_instancia_aleatoria(
            n_vars=random.randint(10, 20),
            n_clausulas=random.randint(20, 40),
            k=3
        )

        solver = SolverKSATParaconsistente(instancia)
        status, _, _ = solver.resolver()
        resultados[status.name] += 1

    print("=" * 50)
    print("BENCHMARK K-SAT PARACONSISTENTE")
    print("=" * 50)
    for status, count in resultados.items():
        print(f"{status}: {count}/{n_testes} ({100*count/n_testes:.1f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("SOLVER K-SAT PARACONSISTENTE")
    print("Protocolo Hermes - P=NP*")
    print("=" * 60)

    # Teste simples
    print("\nTeste com instância simples...")
    instancia = InstanciaKSAT(
        n_variaveis=4,
        n_clausulas=3,
        k=2,
        clausulas=[[1, 2], [-2, 3], [-3, 4]]
    )

    solver = SolverKSATParaconsistente(instancia)
    status, atribuicao, confianca = solver.resolver()

    print(f"Status: {status.value}")
    print(f"Confiança: {confianca:.4f}")
    if atribuicao:
        print(f"Atribuição: {atribuicao}")

    estats = solver.get_estatisticas()
    print(f"\nEstatísticas:")
    print(f"  Nós explorados: {estats['nos_explorados']}")
    print(f"  Resíduos gerados: {estats['residuos_gerados']}")

    # Benchmark
    print("\n" + "=" * 60)
    benchmark_solver(n_testes=50)
