"""
Pipeline integrado de NER Clinico.

Orquestra preprocessamento, inferencia e pos-processamento
em um fluxo unico e configuravel.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

from loguru import logger

from src.ner.entity_types import ClinicalEntityType, ENTITY_COLORS
from src.preprocessing.text_cleaner import ClinicalTextCleaner
from src.preprocessing.abbreviation_expander import AbbreviationExpander
from src.preprocessing.negation_detector import NegationDetector


@dataclass
class ClinicalEntity:
    """Representa uma entidade clinica extraida do texto."""

    text: str
    label: str
    start: int
    end: int
    score: float
    negated: bool = False
    normalized: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PipelineResult:
    """Resultado completo do processamento de um texto clinico."""

    original_text: str
    cleaned_text: str
    entities: List[ClinicalEntity]
    negations: List[Dict]
    processing_time_ms: float
    model_name: str
    entity_count: int = 0
    entity_summary: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.entity_count = len(self.entities)
        self.entity_summary = {}
        for ent in self.entities:
            label = ent.label if isinstance(ent, ClinicalEntity) else ent["label"]
            self.entity_summary[label] = self.entity_summary.get(label, 0) + 1

    def to_dict(self) -> Dict:
        return {
            "original_text": self.original_text,
            "cleaned_text": self.cleaned_text,
            "entities": [
                e.to_dict() if isinstance(e, ClinicalEntity) else e
                for e in self.entities
            ],
            "negations": self.negations,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "model_name": self.model_name,
            "entity_count": self.entity_count,
            "entity_summary": self.entity_summary,
        }


class ClinicalNERPipeline:
    """
    Pipeline completo de extracao de entidades clinicas.

    Integra preprocessamento de texto clinico, inferencia NER via
    Transformer e pos-processamento (deteccao de negacao, normalizacao).

    O pipeline segue estas etapas:
    1. Limpeza do texto (remocao de ruido, normalizacao)
    2. Expansao de abreviacoes medicas
    3. Inferencia NER com modelo Transformer
    4. Deteccao de negacao (nega, ausencia de, sem...)
    5. Normalizacao de entidades
    6. Agregacao de resultados

    Example:
        >>> pipeline = ClinicalNERPipeline()
        >>> pipeline.load()
        >>> result = pipeline.process(
        ...     "Pcte com HAS e DM2, em uso de Losartana 50mg VO 1x/dia. "
        ...     "Nega tabagismo. Hemograma dentro da normalidade."
        ... )
        >>> for ent in result.entities:
        ...     print(f"{ent.text:20s} | {ent.label:20s} | score={ent.score:.2f}")
        HAS                  | CONDICAO             | score=0.97
        DM2                  | CONDICAO             | score=0.96
        Losartana            | MEDICAMENTO          | score=0.99
        50mg                 | DOSAGEM              | score=0.95
        VO                   | VIA_ADMINISTRACAO    | score=0.93
        1x/dia               | FREQUENCIA           | score=0.91
        tabagismo            | CONDICAO             | score=0.88  (negated=True)
        Hemograma            | EXAME                | score=0.94
        dentro da normalidade| RESULTADO_EXAME      | score=0.87
    """

    def __init__(
        self,
        model_name: str = "neuralmind/bert-base-portuguese-cased",
        device: Optional[str] = None,
        expand_abbreviations: bool = True,
        detect_negations: bool = True,
        confidence_threshold: float = 0.5,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.expand_abbreviations = expand_abbreviations
        self.detect_negations = detect_negations

        # Lazy import para evitar dependencia de torch/transformers na importacao
        from src.ner.clinical_ner import ClinicalNERModel

        # Componentes do pipeline
        self.ner_model = ClinicalNERModel(
            model_name=model_name,
            device=device,
        )
        self.text_cleaner = ClinicalTextCleaner()
        self.abbreviation_expander = AbbreviationExpander()
        self.negation_detector = NegationDetector()

        self._is_loaded = False

        logger.info(
            f"ClinicalNERPipeline inicializado | modelo={model_name} | "
            f"abreviacoes={expand_abbreviations} | negacao={detect_negations}"
        )

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        """Carrega todos os componentes do pipeline."""
        self.ner_model.load_model(checkpoint_path)
        self._is_loaded = True
        logger.info("Pipeline carregado e pronto para inferencia")

    def process(self, text: str) -> PipelineResult:
        """
        Processa um texto clinico e extrai entidades.

        Args:
            text: Texto clinico em portugues brasileiro

        Returns:
            PipelineResult com entidades, negacoes e metadados
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline nao carregado. Execute load() primeiro.")

        start_time = datetime.now()

        # 1. Limpeza do texto
        cleaned_text = self.text_cleaner.clean(text)

        # 2. Expansao de abreviacoes
        if self.expand_abbreviations:
            expanded_text = self.abbreviation_expander.expand(cleaned_text)
        else:
            expanded_text = cleaned_text

        # 3. Inferencia NER
        raw_entities = self.ner_model.predict(
            expanded_text,
            threshold=self.confidence_threshold,
        )

        # 4. Deteccao de negacao
        negations = []
        if self.detect_negations:
            negations = self.negation_detector.detect(expanded_text)

        # 5. Criar entidades com metadata
        entities = []
        for ent_dict in raw_entities:
            entity = ClinicalEntity(
                text=ent_dict["text"],
                label=ent_dict["label"],
                start=ent_dict["start"],
                end=ent_dict["end"],
                score=ent_dict["score"],
                negated=self._is_negated(ent_dict, negations),
            )
            entities.append(entity)

        # 6. Calcular tempo
        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        return PipelineResult(
            original_text=text,
            cleaned_text=expanded_text,
            entities=entities,
            negations=negations,
            processing_time_ms=elapsed,
            model_name=self.model_name,
        )

    def process_batch(self, texts: List[str]) -> List[PipelineResult]:
        """Processa multiplos textos clinicos."""
        return [self.process(text) for text in texts]

    def _is_negated(self, entity: Dict, negations: List[Dict]) -> bool:
        """Verifica se uma entidade esta no escopo de uma negacao."""
        for neg in negations:
            neg_end = neg["end"]
            # Janela de negacao: ate 5 tokens apos o marcador
            if entity["start"] >= neg["start"] and entity["start"] <= neg_end + 50:
                return True
        return False
