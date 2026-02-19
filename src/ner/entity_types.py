"""
Definicao dos tipos de entidades clinicas suportadas pelo pipeline.

O schema de entidades foi inspirado no corpus SemClinBr (PUCPR/HAILab)
e adaptado para cobrir os cenarios mais frequentes em prontuarios
eletronicos brasileiros (PEP), receituarios e laudos de exames.
"""

from enum import Enum
from typing import Dict, List


class ClinicalEntityType(str, Enum):
    """Tipos de entidades clinicas reconhecidas pelo pipeline."""

    MEDICAMENTO = "MEDICAMENTO"
    DOSAGEM = "DOSAGEM"
    FREQUENCIA = "FREQUENCIA"
    VIA_ADMINISTRACAO = "VIA_ADMINISTRACAO"
    DIAGNOSTICO = "DIAGNOSTICO"
    PROCEDIMENTO = "PROCEDIMENTO"
    SINTOMA = "SINTOMA"
    EXAME = "EXAME"
    RESULTADO_EXAME = "RESULTADO_EXAME"
    ANATOMIA = "ANATOMIA"
    CONDICAO = "CONDICAO"
    TEMPORAL = "TEMPORAL"
    NEGACAO = "NEGACAO"


# Cores para visualizacao de entidades (compativel com displaCy e dashboards)
ENTITY_COLORS: Dict[str, str] = {
    "MEDICAMENTO": "#2196F3",       # Azul
    "DOSAGEM": "#4CAF50",           # Verde
    "FREQUENCIA": "#FF9800",        # Laranja
    "VIA_ADMINISTRACAO": "#9C27B0", # Roxo
    "DIAGNOSTICO": "#F44336",       # Vermelho
    "PROCEDIMENTO": "#00BCD4",      # Ciano
    "SINTOMA": "#E91E63",           # Rosa
    "EXAME": "#3F51B5",             # Indigo
    "RESULTADO_EXAME": "#8BC34A",   # Verde claro
    "ANATOMIA": "#795548",          # Marrom
    "CONDICAO": "#FF5722",          # Laranja escuro
    "TEMPORAL": "#607D8B",          # Cinza azulado
    "NEGACAO": "#9E9E9E",           # Cinza
}


# Descricoes das entidades para documentacao e tooltips
ENTITY_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "MEDICAMENTO": {
        "pt": "Farmacos, drogas, principios ativos, nomes comerciais",
        "en": "Drugs, active ingredients, brand names",
        "exemplos": "Losartana, Metformina, Dipirona, Amoxicilina, AAS",
    },
    "DOSAGEM": {
        "pt": "Quantidade e unidade de dose",
        "en": "Dose quantity and unit",
        "exemplos": "500mg, 2 comprimidos, 10ml, 40UI, 1g",
    },
    "FREQUENCIA": {
        "pt": "Posologia e frequencia de administracao",
        "en": "Administration frequency and schedule",
        "exemplos": "8/8h, 2x ao dia, a cada 12 horas, pela manha",
    },
    "VIA_ADMINISTRACAO": {
        "pt": "Via de administracao do medicamento",
        "en": "Route of drug administration",
        "exemplos": "via oral, endovenoso, intramuscular, topico, sublingual",
    },
    "DIAGNOSTICO": {
        "pt": "Diagnosticos clinicos e codigos CID-10",
        "en": "Clinical diagnoses and ICD-10 codes",
        "exemplos": "Hipertensao arterial, Diabetes tipo 2, Pneumonia, I10",
    },
    "PROCEDIMENTO": {
        "pt": "Cirurgias, procedimentos e terapias realizadas",
        "en": "Surgeries, procedures and therapies",
        "exemplos": "Colecistectomia, Hemodialise, Fisioterapia, Intubacao",
    },
    "SINTOMA": {
        "pt": "Queixas e manifestacoes clinicas relatadas",
        "en": "Clinical symptoms and complaints",
        "exemplos": "Cefaleia, Dispneia, Dor toracica, Nauseas, Febre",
    },
    "EXAME": {
        "pt": "Exames laboratoriais e de imagem solicitados",
        "en": "Laboratory and imaging tests",
        "exemplos": "Hemograma, Glicemia, Tomografia, Raio-X, PCR",
    },
    "RESULTADO_EXAME": {
        "pt": "Valores numericos e qualitativos de resultados",
        "en": "Numeric and qualitative test results",
        "exemplos": "135 mg/dL, positivo, dentro da normalidade, 12.5 g/dL",
    },
    "ANATOMIA": {
        "pt": "Partes do corpo, orgaos e sistemas anatomicos",
        "en": "Body parts, organs and anatomical systems",
        "exemplos": "Torax, Abdomen, Membro inferior esquerdo, Figado",
    },
    "CONDICAO": {
        "pt": "Doencas cronicas e comorbidades",
        "en": "Chronic diseases and comorbidities",
        "exemplos": "HAS, DM2, DPOC, Insuficiencia renal cronica, Asma",
    },
    "TEMPORAL": {
        "pt": "Datas, duracoes e marcos temporais",
        "en": "Dates, durations and time references",
        "exemplos": "ha 3 dias, desde 2020, por 7 dias, ontem, cronico",
    },
    "NEGACAO": {
        "pt": "Marcadores de negacao clinica",
        "en": "Clinical negation markers",
        "exemplos": "nega, ausencia de, sem, nao apresenta, descarta",
    },
}


# Mapeamento BIO tags (Begin, Inside, Outside)
def get_bio_labels() -> List[str]:
    """Retorna lista de labels BIO para todas as entidades."""
    labels = ["O"]  # Outside
    for entity in ClinicalEntityType:
        labels.append(f"B-{entity.value}")
        labels.append(f"I-{entity.value}")
    return labels


# Label para ID e vice-versa
BIO_LABELS = get_bio_labels()
LABEL2ID = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(BIO_LABELS)}
