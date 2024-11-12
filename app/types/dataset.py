from dataclasses import dataclass


@dataclass
class DatasetEntry:
    role: str
    instruction: str
    input: str
    output: str
