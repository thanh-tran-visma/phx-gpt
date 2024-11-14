from enum import Enum


class InstructionEnum(str, Enum):
    DEFAULT = 'Default'
    Anonymize = 'Anonymize the data'
    OPERATION = 'Handle Operation'
    SUITABLE_INSTRUCTION = 'Return the most suitable instruction'
    FLAG_PERSONAL_DATA = "Return True or False. Note that the data may contain inaccuracies in the response"
