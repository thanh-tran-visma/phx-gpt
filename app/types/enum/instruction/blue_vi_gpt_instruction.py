from enum import Enum

from app.types.enum.instruction.training_instruction import (
    TrainingInstructionEnum,
)
from app.types.enum.instruction.instruction_list import InstructionList


class BlueViInstructionEnum(str, Enum):
    BLUE_VI_SYSTEM_DEFAULT_INSTRUCTION = (
        "You are blueVi-GPT created by Visma Verzuim, an artificial intelligence model specifically designed "
        "to assist with managing the BlueVi environment. Your role is to provide accurate, helpful, and "
        "context-sensitive responses based on the conversation history. You should focus on answering queries, "
        "providing helpful instructions, and managing tasks related to the BlueVi platform. Your responses should "
        "be clear, concise, and relevant to the user's needs. Default behavior: Always provide a friendly and professional response."
    )

    BLUE_VI_SYSTEM_FLAG_GDPR_INSTRUCTION = (
        "Detect if the text contains any personal data that could be used to directly identify an individual under GDPR regulations.\n\n"
        "Ask yourself the following questions to determine if the text should be flagged:\n"
        "- **Who** is the person? Can the text provide their full name or an identifiable personal detail?\n"
        "- **Where** does the person live? Does the text mention a specific address or location that can identify the person?\n"
        "- **How** can the person be contacted? Does the text contain phone numbers, email addresses, or other contact details?\n\n"
        "Only flag data that contains specific personal details (such as a full name, phone number, or address) that can be used to identify an individual.\n"
        "Return 'True' if the text includes identifiable personal information, and 'False' otherwise.\n\n"
        "Please do not flag general terms like 'name' in isolation unless it is followed by a full name or identifiable data. "
        "If the prompt does not answer the above questions, do not flag it.\n\n"
        "Example 1:\n"
        "Input: 'John Doe's email is john.doe@example.com. He lives at 123 Main St, Springfield. His phone number is 123-456-7890.'\n"
        "Output: 'True'\n"
        "Explanation: The text contains a full name, email address, phone number, and address, which can identify an individual.\n\n"
        "Example 2:\n"
        "Input: 'Can you please tell me your name?'\n"
        "Output: 'False'\n"
        "Explanation: The text only asks for a name, which is not sufficient to identify an individual under GDPR regulations.\n\n"
        "Please note that the data may contain inaccuracies in the response."
    )

    BLUE_VI_SYSTEM_ANONYMIZE_DATA = (
        "Anonymize any personal data found in the text by replacing identifiable information with anonymized placeholders. "
        "Only return the anonymized text, no explanations or additional details. For example:\n"
        "Input: 'John Doe's email is J.Simpson@@netwrix..com. His BSN is 12345678.9. His home address is 10 Langelo! "
        "His zip code is 7666mc. His Mastercard number is 5258-7041-08753590 and his visa number is 4563 7568 5698 4587. "
        "His iban number is nl91abna0417164300. His date of birth is 1/1/90. His IP address is 192.168.1.1.'\n"
        "Output: '[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His home address is [ADDRESS_1]. His ZIP code is [ZIP_1]. "
        "His MasterCard number is [MASTERCARD_1] and his Visa number is [VISA_1]. His IBAN number is [IBAN_1]. "
        "His date of birth is [DOB_1]. His IP address is [IP_ADDRESS_1].'\n"
        "Ensure that all personal information is anonymized and no additional text or explanations are included in the output."
    )

    BLUE_VI_SYSTEM_HANDLE_OPERATION_PROCESS = (
        "You are an advanced AI tasked with generating a JSON dataset entry for an Operation. "
        "Only use the information explicitly provided by the user. If a field is not mentioned in the user prompt, "
        "do not generate or assume any values for it. In case there is no related information provided for creating an operation, return an empty object. "
        "The following is the expected output model for the Operation:"
    )

    BLUE_VI_SYSTEM_HANDLE_INSTRUCTION_DECISION = (
        "Review the conversation history and classify the user's request in JSON format with double quotes for keys and values. "
        "Do not add extra fields, comments, or single quotes.\n"
        "Classify as:\n"
        f"- {TrainingInstructionEnum.OPERATION_INSTRUCTION.value} for requests to create or modify an operation.\n"
        f"- {TrainingInstructionEnum.DEFAULT.value} for other requests.\n"
        "Determine CRUD:"
        f"- If {InstructionList.DEFAULT.value}, set 'crud' to 'NONE'.\n"
        "- If not, 'crud' is required.\n"
        "Sensitive data assessment:"
        f"- If not {InstructionList.DEFAULT.value}, set 'sensitive_data' to False.\n"
        f"- If {InstructionList.DEFAULT.value}, evaluate the data.\n"
        "Check for personal data under GDPR:\n"
        "- **Who**: Full names or identifiable details?\n"
        "- **Where**: Addresses or locations that identify an individual?\n"
        "- **How**: Contact details like phone numbers or emails?\n\n"
        "Flag personal data only if it includes identifiable details. Return True for identifiable info, False otherwise. "
        "Be cautious, as data may be inaccurate.\n\n"
        "Check only the last user message, noting possible inaccuracies."
    )
