"""
API REST para o Clinical NLP Pipeline.

Expoe o pipeline de NER clinico via endpoints FastAPI com:
- Documentacao interativa (Swagger UI + ReDoc)
- Validacao de entrada via Pydantic
- Rate limiting e CORS configuravel
- Endpoint de health check e metricas
- Suporte a texto unico e batch processing
"""

import os
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from loguru import logger

from src.ner.pipeline import ClinicalNERPipeline
from src.ner.entity_types import (
    ClinicalEntityType,
    ENTITY_COLORS,
    ENTITY_DESCRIPTIONS,
)


# ============================================================
# Modelos Pydantic (Request / Response)
# ============================================================

class AnalyzeRequest(BaseModel):
    """Requisicao de analise de texto clinico."""

    text: str = Field(
        ...,
        min_length=5,
        max_length=10000,
        description="Texto clinico em portugues brasileiro",
        examples=[
            "Paciente com HAS e DM2, em uso de Losartana 50mg VO 1x/dia. "
            "Nega tabagismo. Hemograma dentro da normalidade."
        ],
    )
    expand_abbreviations: bool = Field(
        default=True,
        description="Expandir abreviacoes medicas automaticamente",
    )
    detect_negations: bool = Field(
        default=True,
        description="Detectar negacoes clinicas",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Score minimo para considerar uma entidade",
    )

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Texto nao pode estar vazio")
        return v.strip()


class BatchAnalyzeRequest(BaseModel):
    """Requisicao de analise em lote."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de textos clinicos (max 100)",
    )
    expand_abbreviations: bool = True
    detect_negations: bool = True
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class EntityResponse(BaseModel):
    """Entidade clinica encontrada."""

    text: str
    label: str
    start: int
    end: int
    score: float
    negated: bool = False


class AnalyzeResponse(BaseModel):
    """Resposta da analise de texto clinico."""

    original_text: str
    cleaned_text: str
    entities: List[EntityResponse]
    negations: List[Dict]
    entity_count: int
    entity_summary: Dict[str, int]
    processing_time_ms: float
    model_name: str


class HealthResponse(BaseModel):
    """Resposta do health check."""

    status: str
    model_loaded: bool
    model_name: str
    version: str
    entities_supported: int
    uptime_seconds: float


class EntityTypeInfo(BaseModel):
    """Informacao sobre um tipo de entidade."""

    name: str
    description_pt: str
    description_en: str
    examples: str
    color: str


# ============================================================
# Aplicacao FastAPI
# ============================================================

# Instancia global do pipeline (carregada no startup)
_pipeline: Optional[ClinicalNERPipeline] = None
_start_time: float = time.time()


def create_app() -> FastAPI:
    """Factory function para criar a aplicacao FastAPI."""

    app = FastAPI(
        title="Clinical NLP Pipeline PT-BR",
        description=(
            "API de Processamento de Linguagem Natural para textos clinicos "
            "em portugues brasileiro. Extrai entidades medicas (medicamentos, "
            "diagnosticos, sintomas, exames, procedimentos) usando modelos "
            "Transformer fine-tunados para o dominio clinico."
        ),
        version="1.0.0",
        contact={
            "name": "Gabriel Demetrios Lafis",
            "url": "https://github.com/galafis",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    # CORS
    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ============================================================
    # Eventos de ciclo de vida
    # ============================================================

    @app.on_event("startup")
    async def startup():
        """Carrega o pipeline NER no startup."""
        global _pipeline, _start_time
        _start_time = time.time()

        model_name = os.getenv(
            "MODEL_NAME", "neuralmind/bert-base-portuguese-cased"
        )
        device = os.getenv("DEVICE", None)
        checkpoint = os.getenv("MODEL_CHECKPOINT", None)

        _pipeline = ClinicalNERPipeline(
            model_name=model_name,
            device=device,
        )

        try:
            _pipeline.load(checkpoint_path=checkpoint)
            logger.info("Pipeline carregado com sucesso no startup")
        except Exception as e:
            logger.warning(f"Pipeline iniciado sem modelo carregado: {e}")

    # ============================================================
    # Endpoints
    # ============================================================

    @app.get("/health", response_model=HealthResponse, tags=["Sistema"])
    async def health_check():
        """Verifica o estado do servico e do modelo."""
        return HealthResponse(
            status="ok" if _pipeline and _pipeline._is_loaded else "degraded",
            model_loaded=_pipeline._is_loaded if _pipeline else False,
            model_name=_pipeline.model_name if _pipeline else "none",
            version="1.0.0",
            entities_supported=len(ClinicalEntityType),
            uptime_seconds=round(time.time() - _start_time, 2),
        )

    @app.get("/entities", response_model=List[EntityTypeInfo], tags=["Referencia"])
    async def list_entity_types():
        """Lista todos os tipos de entidades clinicas suportadas."""
        result = []
        for entity_type in ClinicalEntityType:
            info = ENTITY_DESCRIPTIONS.get(entity_type.value, {})
            result.append(EntityTypeInfo(
                name=entity_type.value,
                description_pt=info.get("pt", ""),
                description_en=info.get("en", ""),
                examples=info.get("exemplos", ""),
                color=ENTITY_COLORS.get(entity_type.value, "#000000"),
            ))
        return result

    @app.post("/analyze", response_model=AnalyzeResponse, tags=["NER"])
    async def analyze_text(request: AnalyzeRequest):
        """
        Analisa um texto clinico e extrai entidades medicas.

        Processa o texto atraves do pipeline completo:
        1. Limpeza e normalizacao
        2. Expansao de abreviacoes (opcional)
        3. Extracao de entidades via Transformer
        4. Deteccao de negacao (opcional)
        5. Agregacao de resultados
        """
        if not _pipeline or not _pipeline._is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Modelo NER nao esta carregado. Tente novamente em instantes.",
            )

        try:
            _pipeline.expand_abbreviations = request.expand_abbreviations
            _pipeline.detect_negations = request.detect_negations
            _pipeline.confidence_threshold = request.confidence_threshold

            result = _pipeline.process(request.text)

            return AnalyzeResponse(
                original_text=result.original_text,
                cleaned_text=result.cleaned_text,
                entities=[
                    EntityResponse(
                        text=e.text,
                        label=e.label,
                        start=e.start,
                        end=e.end,
                        score=round(e.score, 4),
                        negated=e.negated,
                    )
                    for e in result.entities
                ],
                negations=result.negations,
                entity_count=result.entity_count,
                entity_summary=result.entity_summary,
                processing_time_ms=result.processing_time_ms,
                model_name=result.model_name,
            )

        except Exception as e:
            logger.error(f"Erro ao processar texto: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/analyze/batch", tags=["NER"])
    async def analyze_batch(request: BatchAnalyzeRequest):
        """
        Analisa multiplos textos clinicos em lote.

        Processa ate 100 textos por requisicao.
        """
        if not _pipeline or not _pipeline._is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Modelo NER nao esta carregado.",
            )

        try:
            _pipeline.expand_abbreviations = request.expand_abbreviations
            _pipeline.detect_negations = request.detect_negations
            _pipeline.confidence_threshold = request.confidence_threshold

            results = _pipeline.process_batch(request.texts)

            return {
                "results": [r.to_dict() for r in results],
                "total_texts": len(results),
                "total_entities": sum(r.entity_count for r in results),
            }

        except Exception as e:
            logger.error(f"Erro no batch: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


# Instancia da app para uvicorn
app = create_app()
