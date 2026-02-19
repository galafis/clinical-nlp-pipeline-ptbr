"""
Modulo NER (Named Entity Recognition) clinico.

Fornece o modelo de extracao de entidades, o pipeline completo
e as definicoes de tipos de entidades clinicas.
"""

from src.ner.entity_types import ClinicalEntityType, ENTITY_COLORS


def __getattr__(name):
    """Lazy imports para evitar carregar dependencias pesadas
    (torch, transformers) quando nao necessario."""
    if name == "ClinicalNERModel":
        from src.ner.clinical_ner import ClinicalNERModel
        return ClinicalNERModel
    if name == "ClinicalNERPipeline":
        from src.ner.pipeline import ClinicalNERPipeline
        return ClinicalNERPipeline
    raise AttributeError(f"module 'src.ner' has no attribute {name}")


__all__ = [
    "ClinicalEntityType",
    "ENTITY_COLORS",
    "ClinicalNERModel",
    "ClinicalNERPipeline",
]
