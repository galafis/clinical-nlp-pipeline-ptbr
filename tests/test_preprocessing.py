"""
Testes unitarios para o modulo de preprocessamento.
Cobre limpeza de texto, expansao de abreviacoes e deteccao de negacao.
"""

import pytest
from src.preprocessing.text_cleaner import ClinicalTextCleaner
from src.preprocessing.abbreviation_expander import AbbreviationExpander
from src.preprocessing.negation_detector import NegationDetector


# ============================================================
# Testes: ClinicalTextCleaner
# ============================================================

class TestClinicalTextCleaner:
    """Testes para limpeza de texto clinico."""

    def setup_method(self):
        self.cleaner = ClinicalTextCleaner()

    def test_remove_multiple_spaces(self):
        text = "Paciente   com    dor"
        result = self.cleaner.clean(text)
        assert "  " not in result
        assert "Paciente com dor" == result

    def test_normalize_tabs(self):
        text = "Exame:\t\tHemograma"
        result = self.cleaner.clean(text)
        assert "\t" not in result

    def test_remove_cpf(self):
        text = "Paciente CPF 123.456.789-00 com HAS"
        result = self.cleaner.clean(text)
        assert "123.456.789-00" not in result
        assert "[CPF_REMOVIDO]" in result

    def test_remove_phone(self):
        text = "Contato: (11) 99999-8888"
        result = self.cleaner.clean(text)
        assert "99999-8888" not in result
        assert "[TELEFONE_REMOVIDO]" in result

    def test_remove_email(self):
        text = "Email: paciente@hospital.com.br"
        result = self.cleaner.clean(text)
        assert "paciente@hospital.com.br" not in result
        assert "[EMAIL_REMOVIDO]" in result

    def test_normalize_dosage_spacing(self):
        text = "Losartana 50 mg via oral"
        result = self.cleaner.clean(text)
        assert "50mg" in result

    def test_normalize_frequency(self):
        text = "Tomar de 8 / 8h"
        result = self.cleaner.clean(text)
        assert "8/8h" in result

    def test_empty_text(self):
        assert self.cleaner.clean("") == ""
        assert self.cleaner.clean("   ") == ""

    def test_preserves_medical_content(self):
        text = "Paciente com HAS e DM2. PA 130x85mmHg."
        result = self.cleaner.clean(text)
        assert "HAS" in result
        assert "DM2" in result
        assert "130x85mmHg" in result

    def test_phi_removal_can_be_disabled(self):
        cleaner = ClinicalTextCleaner(remove_phi=False)
        text = "CPF 123.456.789-00"
        result = cleaner.clean(text)
        assert "123.456.789-00" in result


# ============================================================
# Testes: AbbreviationExpander
# ============================================================

class TestAbbreviationExpander:
    """Testes para expansao de abreviacoes medicas."""

    def setup_method(self):
        self.expander = AbbreviationExpander()

    def test_expand_pcte(self):
        result = self.expander.expand("pcte com dor")
        assert "paciente" in result

    def test_expand_has(self):
        result = self.expander.expand("Portador de HAS")
        assert "hipertensao arterial sistemica" in result

    def test_expand_dm2(self):
        result = self.expander.expand("Em tratamento para DM2")
        assert "diabetes mellitus tipo 2" in result

    def test_expand_vo(self):
        result = self.expander.expand("Losartana 50mg VO")
        assert "via oral" in result

    def test_expand_ev(self):
        result = self.expander.expand("Dipirona 1g EV")
        assert "endovenoso" in result

    def test_expand_multiple(self):
        result = self.expander.expand("Pcte com HAS e DM, em uso de Losartana VO")
        assert "paciente" in result
        assert "hipertensao arterial sistemica" in result
        assert "diabetes mellitus" in result
        assert "via oral" in result

    def test_no_partial_expansion(self):
        """Nao deve expandir 'has' dentro de 'phases'."""
        result = self.expander.expand("phases of treatment")
        assert "hipertensao" not in result

    def test_get_abbreviations(self):
        abbrs = self.expander.get_abbreviations()
        assert isinstance(abbrs, dict)
        assert len(abbrs) > 50

    def test_add_custom_abbreviation(self):
        self.expander.add_abbreviation("xpto", "exame personalizado")
        result = self.expander.expand("Resultado do XPTO normal")
        assert "exame personalizado" in result

    def test_preserves_non_abbreviations(self):
        text = "O resultado do exame ficou normal"
        result = self.expander.expand(text)
        assert result == text


# ============================================================
# Testes: NegationDetector
# ============================================================

class TestNegationDetector:
    """Testes para deteccao de negacao clinica."""

    def setup_method(self):
        self.detector = NegationDetector()

    def test_detect_nega(self):
        negations = self.detector.detect("Nega febre e calafrios")
        assert len(negations) >= 1
        assert negations[0]["cue"] == "Nega"

    def test_detect_sem(self):
        negations = self.detector.detect("Sem sinais de infeccao")
        assert len(negations) >= 1
        assert "Sem" in negations[0]["cue"]

    def test_detect_ausencia(self):
        negations = self.detector.detect("Ausencia de sopro cardiaco")
        assert len(negations) >= 1

    def test_detect_nao_apresenta(self):
        negations = self.detector.detect("Nao apresenta edema de MMII")
        assert len(negations) >= 1

    def test_no_negation(self):
        negations = self.detector.detect("Paciente com dor toracica")
        assert len(negations) == 0

    def test_multiple_negations(self):
        text = "Nega tabagismo. Nega etilismo. Sem alergias conhecidas."
        negations = self.detector.detect(text)
        assert len(negations) >= 3

    def test_pseudo_negation_ignored(self):
        """'Sem melhora' nao deve ser detectado como negacao."""
        negations = self.detector.detect("Sem melhora com analgesicos")
        # 'Sem' seria detectado, mas 'sem melhora' e pseudo-negacao
        pseudo_found = any(neg["cue"] == "Sem" and "melhora" in
                          negations[0].get("cue", "") for neg in negations)
        # O pseudo deve ser filtrado
        assert len(negations) == 0

    def test_is_negated_method(self):
        text = "Nega dispneia e ortopneia"
        negations = self.detector.detect(text)
        # 'dispneia' esta na posicao 5-13 apos 'Nega'
        assert self.detector.is_negated(5, 13, negations)

    def test_scope_limited_by_delimiter(self):
        text = "Nega febre, mas apresenta tosse"
        negations = self.detector.detect(text)
        # 'tosse' esta apos o delimitador 'mas'
        assert len(negations) >= 1
        # O escopo da negacao deve terminar antes de 'mas'

    def test_post_negation(self):
        negations = self.detector.detect("Diagnostico de pneumonia foi descartado")
        assert len(negations) >= 1
        assert negations[0]["type"] == "post-negation"
