from enum import Enum


class InstructionEnum(str, Enum):
    DEFAULT = 'Default'
    ANSWER_QUESTION = 'Answer the question'
    ASSISTANT_ANONYMIZE_DATA = 'Anonymize the data'
    ASSISTANT_SUITABLE_INSTRUCTION = "Determine and return the most suitable instruction: either 'Default' or 'Operation instruction'."
    ASSISTANT_FLAG_PERSONAL_DATA = 'Determine if the provided data contains personal information. Return True if it does and False otherwise. Note: The data might include inaccuracies, so evaluate carefully.'
    ASSISTANT_OPERATION_HANDLING = (
        "Extract operation details from the input, structure them in the following JSON format, and return the result:\n"
        "{\n"
        "  \"name\": \"\",\n"
        "  \"description\": \"\",\n"
        "  \"duration\": 0,\n"
        "  \"forAppointment\": true,\n"
        "  \"vatRate\": 0,\n"
        "  \"invoicing\": false,\n"
        "  \"hourlyRate\": 0,\n"
        "  \"unitPrice\": null,\n"
        "  \"operationRateType\": null,\n"
        "  \"methodsOfConsult\": [\n"
        "    \"TEL\",\n"
        "    \"WEB\",\n"
        "    \"LOC\"\n"
        "  ],\n"
        "  \"wizard\": null\n"
        "}"
    )
    USER_OPERATION_HANDLING = 'Provide a helpful and friendly response to guide the user through creating a new operation'
    OPERATION_INSTRUCTION = 'Operation instruction'
