"""
Quickstart — Clinical NLP Pipeline PT-BR

Demonstra o uso basico do pipeline para extrair entidades
clinicas de textos medicos em portugues brasileiro.

Executar:
    python examples/quickstart.py
"""

from src.preprocessing.text_cleaner import ClinicalTextCleaner
from src.preprocessing.abbreviation_expander import AbbreviationExpander
from src.preprocessing.negation_detector import NegationDetector
from src.ner.entity_types import ClinicalEntityType, ENTITY_COLORS


def demo_text_cleaner():
    """Demonstra limpeza de texto clinico."""
    print("=" * 60)
    print("1. LIMPEZA DE TEXTO CLINICO")
    print("=" * 60)

    cleaner = ClinicalTextCleaner()

    texto_bruto = (
        "Pcte masculino,   67a, CPF 123.456.789-00,  portador de  "
        "HAS e DM2.\nContato: (11) 99999-8888.\n\n\n"
        "Em uso de Losartana 50 mg  VO  1x/dia."
    )

    texto_limpo = cleaner.clean(texto_bruto)

    print(f"\nTexto bruto:\n{texto_bruto}")
    print(f"\nTexto limpo:\n{texto_limpo}")
    print()


def demo_abbreviation_expander():
    """Demonstra expansao de abreviacoes medicas."""
    print("=" * 60)
    print("2. EXPANSAO DE ABREVIACOES MEDICAS")
    print("=" * 60)

    expander = AbbreviationExpander()

    textos = [
        "Pcte com HAS e DM2, em uso de Losartana VO",
        "Nega dpoc. AP: iam em 2020. Tto com AAS",
        "EF: PA 130x85, FC 72, FR 18, Sat 97%, Tax 36.5",
        "Hb 12.5, Ht 38%, Leuco 8500, Plq 250k, Cr 1.1",
    ]

    for texto in textos:
        expandido = expander.expand(texto)
        print(f"\nOriginal:  {texto}")
        print(f"Expandido: {expandido}")
    print()


def demo_negation_detector():
    """Demonstra deteccao de negacao clinica."""
    print("=" * 60)
    print("3. DETECCAO DE NEGACAO CLINICA")
    print("=" * 60)

    detector = NegationDetector()

    textos = [
        "Nega tabagismo e etilismo",
        "Sem dispneia ou ortopneia",
        "Ausencia de sopro cardiaco",
        "Nao apresenta edema de MMII",
        "Paciente com febre e calafrios",  # sem negacao
        "Diagnostico de pneumonia foi descartado",
    ]

    for texto in textos:
        negations = detector.detect(texto)
        neg_str = ", ".join(
            [f"'{n['cue']}' (tipo: {n['type']})" for n in negations]
        )
        print(f"\nTexto: {texto}")
        if negations:
            print(f"  Negacoes: {neg_str}")
        else:
            print("  Negacoes: nenhuma")
    print()


def demo_entity_types():
    """Lista todas as entidades clinicas suportadas."""
    print("=" * 60)
    print("4. ENTIDADES CLINICAS SUPORTADAS (13 tipos)")
    print("=" * 60)

    from src.ner.entity_types import ENTITY_DESCRIPTIONS

    for entity in ClinicalEntityType:
        info = ENTITY_DESCRIPTIONS[entity.value]
        print(f"\n  {entity.value}")
        print(f"    PT: {info['pt']}")
        print(f"    EN: {info['en']}")
        print(f"    Exemplos: {info['exemplos']}")
    print()


def demo_full_pipeline_simulation():
    """Simula o pipeline completo (sem modelo carregado)."""
    print("=" * 60)
    print("5. SIMULACAO DO PIPELINE COMPLETO")
    print("=" * 60)

    texto = (
        "Paciente feminina, 45 anos, comparece a emergencia com queixa de "
        "cefaleia intensa ha 3 dias, associada a nauseas e vomitos. "
        "Nega febre. Antecedentes: enxaqueca cronica, em uso de "
        "Sumatriptana 50mg SN. Prescrito Dipirona 1g EV e "
        "Metoclopramida 10mg EV."
    )

    print(f"\nTexto original:\n{texto}")

    # 1. Limpar
    cleaner = ClinicalTextCleaner()
    limpo = cleaner.clean(texto)
    print(f"\n1. Texto limpo:\n{limpo}")

    # 2. Expandir abreviacoes
    expander = AbbreviationExpander()
    expandido = expander.expand(limpo)
    print(f"\n2. Abreviacoes expandidas:\n{expandido}")

    # 3. Detectar negacoes
    detector = NegationDetector()
    negations = detector.detect(expandido)
    print(f"\n3. Negacoes detectadas:")
    for neg in negations:
        print(f"   '{neg['cue']}' -> escopo [{neg['scope_start']}:{neg['scope_end']}]")

    # 4. Entidades esperadas (simulacao — modelo real faria inferencia)
    print(f"\n4. Entidades esperadas (ground truth do dataset):")
    expected = [
        ("cefaleia intensa", "SINTOMA", False),
        ("ha 3 dias", "TEMPORAL", False),
        ("nauseas", "SINTOMA", False),
        ("vomitos", "SINTOMA", False),
        ("febre", "SINTOMA", True),  # negada
        ("enxaqueca cronica", "CONDICAO", False),
        ("Sumatriptana", "MEDICAMENTO", False),
        ("50mg", "DOSAGEM", False),
        ("se necessario", "FREQUENCIA", False),
        ("Dipirona", "MEDICAMENTO", False),
        ("1g", "DOSAGEM", False),
        ("endovenoso", "VIA_ADMINISTRACAO", False),
        ("Metoclopramida", "MEDICAMENTO", False),
        ("10mg", "DOSAGEM", False),
        ("endovenoso", "VIA_ADMINISTRACAO", False),
    ]

    for text, label, negated in expected:
        neg_marker = " [NEGADO]" if negated else ""
        cor = ENTITY_COLORS.get(label, "#000")
        print(f"   {text:25s} | {label:20s}{neg_marker}")

    print(f"\n   Total: {len(expected)} entidades em 8 tipos distintos")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Clinical NLP Pipeline PT-BR — Demonstracao")
    print("=" * 60 + "\n")

    demo_text_cleaner()
    demo_abbreviation_expander()
    demo_negation_detector()
    demo_entity_types()
    demo_full_pipeline_simulation()

    print("=" * 60)
    print("  Para usar o pipeline completo com modelo Transformer:")
    print("  >>> from src.ner.pipeline import ClinicalNERPipeline")
    print("  >>> pipeline = ClinicalNERPipeline()")
    print("  >>> pipeline.load()")
    print('  >>> result = pipeline.process("Paciente com HAS...")')
    print("=" * 60)
