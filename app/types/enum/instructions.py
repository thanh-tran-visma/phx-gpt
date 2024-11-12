from enum import Enum


class InstructionTypes(str, Enum):
    OPERATION = "Handle Operation"
    DEFAULT = "Default"
