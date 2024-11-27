from enum import Enum


class TrainingInstructionEnum(str, Enum):
    DEFAULT = 'Default'
    ANSWER_QUESTION = 'Answer the question'
    ASSISTANT_ANONYMIZE_DATA = 'Anonymize the data'
    ASSISTANT_FLAG_PERSONAL_DATA = 'Determine if the provided data contains personal information. Return True if it does and False otherwise. Note: The data might include inaccuracies, so evaluate carefully.'
    ASSISTANT_OPERATION_HANDLING = (
        "You are an advanced AI, tasked to assist the user by calling functions in JSON format. "
        "Extract operation details from the input, structure them in the following strict JSON format, "
        "and return the result with double quotes for all keys and string values:\n"
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
        "    {\n"
        "      \"shortCode\": \"TEL\",\n"
        "      \"name\": \"Phone\"\n"
        "    },\n"
        "    {\n"
        "      \"shortCode\": \"WEB\",\n"
        "      \"name\": \"Web\"\n"
        "    },\n"
        "    {\n"
        "      \"shortCode\": \"LOC\",\n"
        "      \"name\": \"Location\"\n"
        "    }\n"
        "  ],\n"
        "  \"wizard\": null\n"
        "}\n"
        "Fields that are not provided should be set as null where appropriate. Ensure the output strictly follows this JSON structure. "
        "Do not include extra fields, comments, or use single quotes."
    )
    USER_OPERATION_HANDLING = 'Provide a helpful and friendly response to guide the user through creating a new operation'
    OPERATION_INSTRUCTION = 'PHX Operation instruction for appointment'
    ASSISTANT_SUITABLE_INSTRUCTION = (
        "You are an advanced AI, tasked with reviewing the provided conversation history and classifying the user's request. "
        "Extract the relevant details and structure the response in the following strict JSON format, "
        "with double quotes for all keys and string values. Do not include any extra fields, comments, or use single quotes.\n"
        "Classify the user's intent as follows:\n"
        f"- If the user is asking to create or modify an operation, classify it as {OPERATION_INSTRUCTION}.\n"
        f"- If the user is asking about something else, classify it as {DEFAULT}.\n"
        "Determine the CRUD operation for the request:\n"
        "- If the instruction is {InstructionList.DEFAULT.value}, set the 'crud' field to 'NONE'.\n"
        "- If the instruction is not {InstructionList.DEFAULT.value}, then the 'crud' field is required.\n"
        "Assess the presence of sensitive data as follows:\n"
        "- If the instruction is not {InstructionList.DEFAULT.value}, set 'personal_data' to False.\n"
        "- If the instruction is {InstructionList.DEFAULT.value}, evaluate the data as described below.\n"
        "Additionally, assess if the provided data contains personal information under GDPR regulations:\n"
        "- **Who**: Does the text mention any full name or identifiable personal details?\n"
        "- **Where**: Does the text mention specific addresses or locations that could identify an individual?\n"
        "- **How**: Does the text contain contact details like phone numbers or email addresses?\n\n"
        "Flag personal data only if it contains identifiable details such as a full name, address, phone number, or email address.\n"
        "Return True if the text includes identifiable personal information (such as a full name, address, or contact details), and False otherwise. "
        "Evaluate carefully, as data may contain inaccuracies.\n\n"
        "Please note that, only check for the last message from user role only and the data may contain inaccuracies in the response."
    )
