"""
Testes de integracao para a API FastAPI.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.ner.entity_types import ClinicalEntityType


@pytest.fixture
def client():
    """Cria client de teste FastAPI."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Testes para o endpoint /health."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data
        assert "version" in data
        assert "entities_supported" in data
        assert "uptime_seconds" in data

    def test_health_version(self, client):
        response = client.get("/health")
        assert response.json()["version"] == "1.0.0"

    def test_health_entity_count(self, client):
        response = client.get("/health")
        assert response.json()["entities_supported"] == len(ClinicalEntityType)


class TestEntitiesEndpoint:
    """Testes para o endpoint /entities."""

    def test_entities_returns_200(self, client):
        response = client.get("/entities")
        assert response.status_code == 200

    def test_entities_returns_all_types(self, client):
        response = client.get("/entities")
        data = response.json()
        assert len(data) == len(ClinicalEntityType)

    def test_entity_structure(self, client):
        response = client.get("/entities")
        entity = response.json()[0]
        assert "name" in entity
        assert "description_pt" in entity
        assert "description_en" in entity
        assert "examples" in entity
        assert "color" in entity

    def test_entity_colors_are_hex(self, client):
        response = client.get("/entities")
        for entity in response.json():
            assert entity["color"].startswith("#")


class TestAnalyzeEndpoint:
    """Testes para o endpoint /analyze."""

    def test_analyze_requires_text(self, client):
        response = client.post("/analyze", json={})
        assert response.status_code == 422

    def test_analyze_empty_text_rejected(self, client):
        response = client.post("/analyze", json={"text": ""})
        assert response.status_code == 422

    def test_analyze_short_text_rejected(self, client):
        response = client.post("/analyze", json={"text": "abc"})
        assert response.status_code == 422

    def test_analyze_valid_request_structure(self, client):
        """Verifica que o schema do request e valido."""
        # O modelo pode nao estar carregado no teste, entao 503 e aceitavel
        response = client.post(
            "/analyze",
            json={
                "text": "Paciente com HAS em uso de Losartana 50mg VO",
                "expand_abbreviations": True,
                "detect_negations": True,
                "confidence_threshold": 0.5,
            },
        )
        # 503 = modelo nao carregado (esperado em testes sem GPU)
        # 200 = tudo OK
        assert response.status_code in (200, 503)

    def test_analyze_confidence_threshold_validation(self, client):
        response = client.post(
            "/analyze",
            json={"text": "Teste de validacao", "confidence_threshold": 1.5},
        )
        assert response.status_code == 422

    def test_analyze_confidence_negative_rejected(self, client):
        response = client.post(
            "/analyze",
            json={"text": "Teste de validacao", "confidence_threshold": -0.5},
        )
        assert response.status_code == 422


class TestBatchEndpoint:
    """Testes para o endpoint /analyze/batch."""

    def test_batch_requires_texts(self, client):
        response = client.post("/analyze/batch", json={})
        assert response.status_code == 422

    def test_batch_empty_list_rejected(self, client):
        response = client.post("/analyze/batch", json={"texts": []})
        assert response.status_code == 422

    def test_batch_valid_request(self, client):
        response = client.post(
            "/analyze/batch",
            json={"texts": ["Paciente com dor toracica", "Nega febre"]},
        )
        assert response.status_code in (200, 503)
