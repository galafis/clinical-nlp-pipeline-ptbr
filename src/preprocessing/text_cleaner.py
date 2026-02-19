"""
Limpeza e normalizacao de textos clinicos brasileiros.

Trata os problemas mais comuns encontrados em prontuarios eletronicos:
- Multiplos espacos, tabs, quebras de linha
- Caracteres especiais e encoding inconsistente
- Anonimizacao basica de PHI (Protected Health Information)
- Normalizacao de pontuacao medica
"""

import re
import unicodedata
from typing import Optional


class ClinicalTextCleaner:
    """
    Limpador de texto clinico otimizado para prontuarios brasileiros.

    Preserva informacoes clinicamente relevantes enquanto remove
    ruido tipico de sistemas de prontuario eletronico (PEP).
    """

    # Padroes de PHI para anonimizacao basica
    PHI_PATTERNS = [
        # CPF
        (r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[CPF_REMOVIDO]"),
        # Telefone
        (r"\b\(?\d{2}\)?\s*\d{4,5}-?\d{4}\b", "[TELEFONE_REMOVIDO]"),
        # Email
        (r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[EMAIL_REMOVIDO]"),
        # RG
        (r"\bRG\s*:?\s*[\d./-]+\b", "[RG_REMOVIDO]"),
        # Prontuario (generico)
        (r"\bprontu[aá]rio\s*:?\s*\d+\b", "[PRONTUARIO]", re.IGNORECASE),
    ]

    # Padroes de normalizacao
    NORMALIZE_PATTERNS = [
        # Multiplos espacos
        (r"\s+", " "),
        # Quebras de linha multiplas
        (r"\n{3,}", "\n\n"),
        # Tabs
        (r"\t+", " "),
        # Espacos antes de pontuacao
        (r"\s+([.,;:!?])", r"\1"),
    ]

    def __init__(self, remove_phi: bool = True, normalize_whitespace: bool = True):
        self.remove_phi = remove_phi
        self.normalize_whitespace = normalize_whitespace

    def clean(self, text: str) -> str:
        """
        Limpa e normaliza texto clinico.

        Args:
            text: Texto bruto do prontuario

        Returns:
            Texto limpo e normalizado
        """
        if not text or not text.strip():
            return ""

        result = text

        # 1. Normalizar Unicode (NFC — forma canonica composta)
        result = unicodedata.normalize("NFC", result)

        # 2. Remover caracteres de controle (exceto newline e tab)
        result = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", result)

        # 3. Anonimizar PHI se configurado
        if self.remove_phi:
            result = self._remove_phi(result)

        # 4. Normalizar whitespace
        if self.normalize_whitespace:
            for pattern, replacement in self.NORMALIZE_PATTERNS:
                result = re.sub(pattern, replacement, result)

        # 5. Normalizar pontuacao medica
        result = self._normalize_medical_punctuation(result)

        # 6. Strip final
        result = result.strip()

        return result

    def _remove_phi(self, text: str) -> str:
        """Remove Protected Health Information do texto."""
        result = text
        for item in self.PHI_PATTERNS:
            if len(item) == 3:
                pattern, replacement, flags = item
                result = re.sub(pattern, replacement, result, flags=flags)
            else:
                pattern, replacement = item
                result = re.sub(pattern, replacement, result)
        return result

    def _normalize_medical_punctuation(self, text: str) -> str:
        """Normaliza padroes de pontuacao especificos de textos medicos."""
        result = text

        # Normalizar separadores de dosagem: "500 mg" -> "500mg"
        result = re.sub(r"(\d+)\s+(mg|ml|mcg|g|UI|mEq|mmol)\b", r"\1\2", result)

        # Normalizar fracao de frequencia: "8 / 8h" -> "8/8h"
        result = re.sub(r"(\d+)\s*/\s*(\d+)\s*h", r"\1/\2h", result)

        # Normalizar "x" de frequencia: "2 x ao dia" -> "2x ao dia"
        result = re.sub(r"(\d+)\s*x\s+", r"\1x ", result)

        return result
