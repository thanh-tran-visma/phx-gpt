from enum import Enum


class InstructionEnum(str, Enum):
    DEFAULT = 'Default'
    Assistant_Anonymize_Data = 'Anonymize the data'
    ASSISTANT_SUITABLE_INSTRUCTION = 'Return the most suitable instruction'
    ASSISTANT_FLAG_PERSONAL_DATA = 'Return True or False. Note that the data may contain inaccuracies in the response'
    ASSISTANT_OPERATION_HANDLING = 'Generate and return the operation details in JSON format based on the user’s prompt.'
    USER_OPERATION_HANDLING = 'Provide a helpful and friendly response to guide the user through creating a new operation'
    OPERATION_Instruction = 'Operation instruction'
