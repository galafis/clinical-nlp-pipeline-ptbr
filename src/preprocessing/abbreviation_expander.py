"""
Expansao de abreviacoes medicas em portugues brasileiro.

Contem dicionario de 60+ abreviacoes comuns em prontuarios eletronicos,
receituarios, laudos e relatorios de alta hospitalar.
"""

import re
from typing import Dict, Optional

import yaml
from loguru import logger


# Dicionario padrao de abreviacoes medicas PT-BR
DEFAULT_ABBREVIATIONS: Dict[str, str] = {
    # Termos gerais de prontuario
    "pcte": "paciente",
    "pct": "paciente",
    "dx": "diagnostico",
    "ddx": "diagnostico diferencial",
    "rx": "prescricao",
    "hda": "historia da doenca atual",
    "hpp": "historia patologica pregressa",
    "hf": "historia familiar",
    "hs": "historia social",
    "ap": "antecedentes pessoais",
    "af": "antecedentes familiares",
    "qp": "queixa principal",
    "qd": "queixa e duracao",
    "ef": "exame fisico",
    "tto": "tratamento",
    "ctl": "controle",
    "prog": "prognostico",
    "cd": "conduta",
    "hdz": "hipotese diagnostica",

    # Vias de administracao
    "vo": "via oral",
    "ev": "endovenoso",
    "iv": "intravenoso",
    "im": "intramuscular",
    "sc": "subcutaneo",
    "sl": "sublingual",
    "vr": "via retal",
    "top": "topico",
    "inal": "inalatorio",

    # Frequencia / posologia
    "bid": "duas vezes ao dia",
    "tid": "tres vezes ao dia",
    "qid": "quatro vezes ao dia",
    "prn": "se necessario",
    "acm": "a criterio medico",
    "sn": "se necessario",

    # Condicoes clinicas
    "has": "hipertensao arterial sistemica",
    "dm": "diabetes mellitus",
    "dm2": "diabetes mellitus tipo 2",
    "dm1": "diabetes mellitus tipo 1",
    "dpoc": "doenca pulmonar obstrutiva cronica",
    "iam": "infarto agudo do miocardio",
    "icc": "insuficiencia cardiaca congestiva",
    "ic": "insuficiencia cardiaca",
    "irc": "insuficiencia renal cronica",
    "itu": "infeccao do trato urinario",
    "avc": "acidente vascular cerebral",
    "tce": "traumatismo cranioencefalico",
    "tvp": "trombose venosa profunda",
    "tep": "tromboembolismo pulmonar",
    "fa": "fibrilacao atrial",
    "fv": "fibrilacao ventricular",
    "pcr": "parada cardiorrespiratoria",
    "bcp": "broncopneumonia",
    "pnm": "pneumonia",
    "ira": "insuficiencia respiratoria aguda",
    "lra": "lesao renal aguda",
    "hiv": "virus da imunodeficiencia humana",
    "tb": "tuberculose",
    "ca": "cancer",
    "neo": "neoplasia",

    # Sinais vitais e exame fisico
    "fc": "frequencia cardiaca",
    "fr": "frequencia respiratoria",
    "pa": "pressao arterial",
    "sat": "saturacao de oxigenio",
    "tax": "temperatura axilar",
    "tax.": "temperatura axilar",
    "beg": "bom estado geral",
    "reg": "regular estado geral",
    "meg": "mau estado geral",
    "lote": "lucido, orientado no tempo e espaco",
    "mvf": "murmorio vesicular fisiologico",
    "mv": "murmorio vesicular",
    "bcnf": "bulhas cardiacas normofonéticas",
    "rnf": "ruidos normofonéticos",
    "rha": "ruidos hidro-aereos",
    "abd": "abdomen",
    "mmii": "membros inferiores",
    "mmss": "membros superiores",
    "msd": "membro superior direito",
    "mse": "membro superior esquerdo",
    "mid": "membro inferior direito",
    "mie": "membro inferior esquerdo",

    # Exames
    "hb": "hemoglobina",
    "ht": "hematocrito",
    "leuco": "leucocitos",
    "plq": "plaquetas",
    "cr": "creatinina",
    "ur": "ureia",
    "gli": "glicemia",
    "hba1c": "hemoglobina glicada",
    "tgo": "transaminase glutamico-oxalacetica",
    "tgp": "transaminase glutamico-piruvica",
    "ecg": "eletrocardiograma",
    "eco": "ecocardiograma",
    "tc": "tomografia computadorizada",
    "rm": "ressonancia magnetica",
    "usg": "ultrassonografia",
    "rx": "raio-x",
    "eas": "elementos anormais e sedimento",
}


class AbbreviationExpander:
    """
    Expande abreviacoes medicas em textos clinicos.

    Usa um dicionario de 90+ abreviacoes comuns em prontuarios
    brasileiros. Suporta carregamento de dicionarios customizados
    via arquivo YAML.
    """

    def __init__(
        self,
        custom_dict_path: Optional[str] = None,
        case_sensitive: bool = False,
    ):
        self.abbreviations = dict(DEFAULT_ABBREVIATIONS)
        self.case_sensitive = case_sensitive

        if custom_dict_path:
            self._load_custom_dict(custom_dict_path)

        logger.debug(
            f"AbbreviationExpander inicializado | "
            f"{len(self.abbreviations)} abreviacoes carregadas"
        )

    def _load_custom_dict(self, path: str) -> None:
        """Carrega dicionario customizado de um arquivo YAML."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                custom = yaml.safe_load(f)
            if isinstance(custom, dict):
                self.abbreviations.update(custom)
                logger.info(f"Dicionario customizado carregado de {path}")
        except Exception as e:
            logger.warning(f"Falha ao carregar dicionario customizado: {e}")

    def expand(self, text: str) -> str:
        """
        Expande abreviacoes no texto.

        Usa word boundaries para evitar expansoes parciais.
        Exemplo: "pcte com HAS" -> "paciente com hipertensao arterial sistemica"

        Args:
            text: Texto clinico com abreviacoes

        Returns:
            Texto com abreviacoes expandidas
        """
        result = text

        for abbr, expansion in self.abbreviations.items():
            if self.case_sensitive:
                pattern = rf"\b{re.escape(abbr)}\b"
                result = re.sub(pattern, expansion, result)
            else:
                pattern = rf"\b{re.escape(abbr)}\b"
                result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)

        return result

    def add_abbreviation(self, abbr: str, expansion: str) -> None:
        """Adiciona uma abreviacao ao dicionario."""
        self.abbreviations[abbr.lower()] = expansion

    def get_abbreviations(self) -> Dict[str, str]:
        """Retorna copia do dicionario de abreviacoes."""
        return dict(self.abbreviations)
