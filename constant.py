from pathlib import Path


CHUNKS_FILE = Path("data/processed/chunks.json")
ENTITIES_FILE = Path("data/interim/entities.json")
RELATIONS_FILE = Path("data/interim/relations.json")


# Domain dictionaries / seed terms


DOMAIN_TERMS = {
    "condition": [
        "hiv",
        "abortion",
        "infertility",
        "pregnancy",
        "complications",
        "infection",
        "haemorrhage",
        "hemorrhage",
        "pain",
        "sepsis",
    ],
    "procedure": [
        "abortion care",
        "post-abortion care",
        "contraception",
        "counselling",
        "counseling",
        "follow-up",
        "screening",
        "testing",
        "treatment",
        "antiretroviral therapy",
    ],
    "actor": [
        "health worker",
        "health workers",
        "clinician",
        "clinicians",
        "women",
        "adolescents",
        "patients",
        "person",
        "people",
    ],
    "organisation": [
        "world health organization",
        "who",
    ],
    "resource": [
        "guideline",
        "guidelines",
        "service",
        "services",
        "care",
        "support",
    ]
}


ALIASES = {
    "who": "world health organization",
    "counselling": "counseling",
    "health workers": "health worker",
    "clinicians": "clinician",
    "guidelines": "guideline",
    "services": "service",
    "women": "woman",
    "adolescents": "adolescent",
    "patients": "patient",
    "complications": "complication",
}
