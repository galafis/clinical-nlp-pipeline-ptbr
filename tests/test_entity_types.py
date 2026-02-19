"""
Testes unitarios para tipos de entidades e labels BIO.
"""

import pytest
from src.ner.entity_types import (
    ClinicalEntityType,
    ENTITY_COLORS,
    ENTITY_DESCRIPTIONS,
    BIO_LABELS,
    LABEL2ID,
    ID2LABEL,
    get_bio_labels,
)


class TestClinicalEntityType:
    """Testes para o enum de entidades clinicas."""

    def test_all_entities_defined(self):
        """Deve ter exatamente 13 tipos de entidades."""
        assert len(ClinicalEntityType) == 13

    def test_entity_names(self):
        expected = {
            "MEDICAMENTO", "DOSAGEM", "FREQUENCIA", "VIA_ADMINISTRACAO",
            "DIAGNOSTICO", "PROCEDIMENTO", "SINTOMA", "EXAME",
            "RESULTADO_EXAME", "ANATOMIA", "CONDICAO", "TEMPORAL", "NEGACAO",
        }
        actual = {e.value for e in ClinicalEntityType}
        assert actual == expected

    def test_entity_colors_complete(self):
        """Todas as entidades devem ter cor definida."""
        for entity in ClinicalEntityType:
            assert entity.value in ENTITY_COLORS
            assert ENTITY_COLORS[entity.value].startswith("#")

    def test_entity_descriptions_complete(self):
        """Todas as entidades devem ter descricao."""
        for entity in ClinicalEntityType:
            assert entity.value in ENTITY_DESCRIPTIONS
            desc = ENTITY_DESCRIPTIONS[entity.value]
            assert "pt" in desc
            assert "en" in desc
            assert "exemplos" in desc


class TestBIOLabels:
    """Testes para labels BIO (Begin, Inside, Outside)."""

    def test_bio_labels_count(self):
        """13 entidades x 2 (B + I) + 1 (O) = 27 labels."""
        labels = get_bio_labels()
        assert len(labels) == 27

    def test_o_label_first(self):
        assert BIO_LABELS[0] == "O"

    def test_b_and_i_pairs(self):
        """Cada entidade deve ter B- e I- labels."""
        for entity in ClinicalEntityType:
            assert f"B-{entity.value}" in BIO_LABELS
            assert f"I-{entity.value}" in BIO_LABELS

    def test_label2id_consistency(self):
        """LABEL2ID e ID2LABEL devem ser inversos."""
        for label, idx in LABEL2ID.items():
            assert ID2LABEL[idx] == label

    def test_id2label_consistency(self):
        for idx, label in ID2LABEL.items():
            assert LABEL2ID[label] == idx

    def test_all_labels_have_ids(self):
        for label in BIO_LABELS:
            assert label in LABEL2ID
