from dataclasses import dataclass


@dataclass
class DatasetEntry:
    instruction: str
    input: str
    output: str
