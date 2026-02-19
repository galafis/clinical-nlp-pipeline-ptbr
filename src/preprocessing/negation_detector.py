"""
Deteccao de negacao em textos clinicos em portugues brasileiro.

A negacao e um aspecto critico em NLP clinico — um diagnostico negado
tem significado oposto ao de um diagnostico confirmado. Este modulo
implementa deteccao de negacao baseada em padroes linguisticos
especificos do portugues clinico.

Referencia: Adaptado de NegEx (Chapman et al., 2001) para PT-BR
"""

import re
from typing import Dict, List, Tuple


class NegationDetector:
    """
    Detecta marcadores de negacao em textos clinicos em portugues.

    Identifica expressoes de negacao e determina o escopo de cada
    negacao (quais termos estao sendo negados). Essencial para
    diferenciar "paciente com diabetes" de "paciente nega diabetes".

    Padroes suportados:
    - Pre-negacao: "nega", "sem", "ausencia de", "nao apresenta"
    - Pos-negacao: "foi descartado", "excluido", "negativo"
    - Pseudo-negacao: "sem melhora" (nao e negacao real da condicao)

    Example:
        >>> detector = NegationDetector()
        >>> negations = detector.detect(
        ...     "Paciente nega tabagismo e etilismo. Sem dispneia."
        ... )
        >>> for neg in negations:
        ...     print(f"'{neg['cue']}' em posicao {neg['start']}-{neg['end']}")
        'nega' em posicao 9-13
        'Sem' em posicao 38-41
    """

    # Marcadores de negacao em portugues clinico
    PRE_NEGATION_CUES = [
        # Verbos de negacao
        r"\bnega\b",
        r"\bnegou\b",
        r"\bnegando\b",
        r"\bnega[- ]se\b",
        r"\bnao\s+(?:apresenta|refere|relata|possui|tem|ha|houve)",
        r"\bnao\b",
        r"\bnunca\b",
        r"\bjamais\b",
        # Preposicoes/adverbios de ausencia
        r"\bsem\b",
        r"\bausencia\s+de\b",
        r"\bausente\b",
        r"\bnenhum[a]?\b",
        # Expressoes clinicas
        r"\bdescarta[- ]se\b",
        r"\bexclu[ií][- ]se\b",
        r"\bafasta[- ]se\b",
        r"\bnao\s+evidenci",
        r"\bnao\s+identific",
        r"\bnao\s+observ",
        r"\bnao\s+detect",
        r"\bnao\s+visualiz",
    ]

    # Marcadores de negacao que aparecem APOS o termo
    POST_NEGATION_CUES = [
        r"\bfoi\s+descartad[oa]\b",
        r"\bfoi\s+excluid[oa]\b",
        r"\bfoi\s+afastad[oa]\b",
        r"\bfoi\s+negad[oa]\b",
        r"\bnegativ[oa]\b",
        r"\bausen[ct]e\b",
        r"\bnao\s+confirmad[oa]\b",
        r"\bimprovavel\b",
    ]

    # Pseudo-negacao (parece negacao mas nao e)
    PSEUDO_NEGATION = [
        r"\bsem\s+melhora\b",
        r"\bsem\s+sucesso\b",
        r"\bsem\s+resposta\b",
        r"\bnao\s+melhor",
        r"\bnao\s+respondeu\b",
        r"\bnao\s+tolerou\b",
        r"\bapesar\s+de\s+nao\b",
    ]

    # Delimitadores de escopo (quebram o escopo da negacao)
    SCOPE_DELIMITERS = [
        r"\bmas\b",
        r"\bporem\b",
        r"\bentretanto\b",
        r"\bcontudo\b",
        r"\bapresenta\b",
        r"\brefere\b",
        r"\brelata\b",
        r"\bcom\b",
        r"[.;:]",
    ]

    def __init__(self, scope_window: int = 50):
        """
        Args:
            scope_window: Numero maximo de caracteres que uma negacao
                abrange apos o marcador (default: 50)
        """
        self.scope_window = scope_window

        # Compilar padroes
        self._pre_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PRE_NEGATION_CUES
        ]
        self._post_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.POST_NEGATION_CUES
        ]
        self._pseudo_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.PSEUDO_NEGATION
        ]
        self._delimiter_pattern = re.compile(
            "|".join(self.SCOPE_DELIMITERS), re.IGNORECASE
        )

    def detect(self, text: str) -> List[Dict]:
        """
        Detecta marcadores de negacao no texto.

        Args:
            text: Texto clinico

        Returns:
            Lista de dicts com info de cada negacao:
            [{"cue": str, "start": int, "end": int,
              "scope_start": int, "scope_end": int, "type": str}]
        """
        negations = []

        # Verificar pseudo-negacoes primeiro (para ignorar depois)
        pseudo_spans = set()
        for pattern in self._pseudo_patterns:
            for match in pattern.finditer(text):
                pseudo_spans.add((match.start(), match.end()))

        # Detectar pre-negacoes
        for pattern in self._pre_patterns:
            for match in pattern.finditer(text):
                # Verificar se nao e pseudo-negacao
                is_pseudo = any(
                    ps <= match.start() <= pe for ps, pe in pseudo_spans
                )
                if is_pseudo:
                    continue

                scope_end = self._find_scope_end(text, match.end())
                negations.append({
                    "cue": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "scope_start": match.start(),
                    "scope_end": scope_end,
                    "type": "pre-negation",
                })

        # Detectar pos-negacoes
        for pattern in self._post_patterns:
            for match in pattern.finditer(text):
                scope_start = self._find_scope_start(text, match.start())
                negations.append({
                    "cue": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "scope_start": scope_start,
                    "scope_end": match.end(),
                    "type": "post-negation",
                })

        # Ordenar por posicao
        negations.sort(key=lambda x: x["start"])

        return negations

    def _find_scope_end(self, text: str, neg_end: int) -> int:
        """Encontra o fim do escopo de uma pre-negacao."""
        search_text = text[neg_end : neg_end + self.scope_window]
        delimiter = self._delimiter_pattern.search(search_text)

        if delimiter:
            return neg_end + delimiter.start()
        return min(neg_end + self.scope_window, len(text))

    def _find_scope_start(self, text: str, neg_start: int) -> int:
        """Encontra o inicio do escopo de uma pos-negacao."""
        search_start = max(0, neg_start - self.scope_window)
        search_text = text[search_start:neg_start]

        # Procurar ultimo delimitador antes da negacao
        last_delim = 0
        for match in self._delimiter_pattern.finditer(search_text):
            last_delim = match.end()

        return search_start + last_delim

    def is_negated(self, entity_start: int, entity_end: int, negations: List[Dict]) -> bool:
        """
        Verifica se uma entidade em [start, end] esta no escopo de alguma negacao.

        Args:
            entity_start: Posicao inicial da entidade
            entity_end: Posicao final da entidade
            negations: Lista de negacoes detectadas

        Returns:
            True se a entidade esta negada
        """
        for neg in negations:
            if (entity_start >= neg["scope_start"] and
                    entity_start <= neg["scope_end"]):
                return True
        return False
