import json
import logging
from typing import List
from llama_cpp import Llama, LlamaGrammar
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import (
    generate_gbnf_grammar_and_documentation,
)

from app.model import Message
from app.schemas import GptResponseSchema, PhxAppOperation
from app.types.enum.gpt import Role
from app.types.enum.instruction import TrainingInstructionEnum
from app.types.enum.instruction.blue_vi_gpt_instruction_enum import (
    BlueViInstructionEnum,
)
from app.utils import (
    get_blue_vi_response,
    convert_blue_vi_response_to_schema,
    convert_conversation_history_to_tuples,
)


class BlueViGptAssistant:
    def __init__(self, llm: Llama):
        self.llm = llm

    async def check_for_personal_data(self, prompt: str) -> bool:
        """Detect personal data in the prompt relevant to GDPR compliance."""
        response = await get_blue_vi_response(
            self.llm,
            [
                (
                    Role.SYSTEM.value,
                    BlueViInstructionEnum.BLUE_VI_SYSTEM_FLAG_GDPR_INSTRUCTION.value,
                )
            ]
            + [(Role.USER.value, prompt)],
        )

        if not response:
            return False
        result = convert_blue_vi_response_to_schema(response)
        return "True" in result.content

    async def get_anonymized_message(
        self, user_message: str
    ) -> GptResponseSchema:
        """Anonymize the user message."""
        instruction = BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value
        response = await get_blue_vi_response(
            self.llm,
            [Role.SYSTEM.value, instruction]
            + [(Role.USER.value, user_message)],
        )
        return convert_blue_vi_response_to_schema(response)

    async def identify_instruction_type(
        self, conversation_history: List[Message]
    ) -> str:
        """
        Identify the type of instruction based on the conversation history and the prompt context.
        """
        instruction = (
            f"{BlueViInstructionEnum.BLUE_VI_SYSTEM_HANDLE_INSTRUCTION_DECISION.value} \n"
            f"{convert_conversation_history_to_tuples(conversation_history)}"
        )
        # Convert conversation history to tuples and prepare messages
        messages = [
            (Role.SYSTEM.value, instruction)
        ] + convert_conversation_history_to_tuples(conversation_history)

        # Get response from the model
        response = await get_blue_vi_response(self.llm, messages)

        if not response:
            return TrainingInstructionEnum.DEFAULT.value

        # Convert the model's response to the expected schema
        result = convert_blue_vi_response_to_schema(response)

        # Return the appropriate instruction type based on the model's response content
        return (
            TrainingInstructionEnum.OPERATION_INSTRUCTION.value
            if TrainingInstructionEnum.OPERATION_INSTRUCTION.value
            in result.content
            else TrainingInstructionEnum.DEFAULT.value
        )

    async def get_operation_format(
        self, conversation_history: List[Message]
    ) -> PhxAppOperation:
        """Generates an operation schema based on the user's conversation history and model response."""
        # TODO:

        # Define the instruction
        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
            [PhxAppOperation]
        )
        grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)

        instruction = f"{BlueViInstructionEnum.BLUE_VI_SYSTEM_HANDLE_OPERATION_PROCESS.value}: \n\n {documentation}"

        # Create the messages structure with instruction and conversation history
        messages = [
            (Role.SYSTEM.value, instruction)
        ] + convert_conversation_history_to_tuples(conversation_history)

        # Get response from LLM
        response = await get_blue_vi_response(self.llm, messages, grammar)

        # Convert response to schema
        result = convert_blue_vi_response_to_schema(response)

        # Handle JSON decoding
        if not result.content:
            return PhxAppOperation()
        try:
            data: PhxAppOperation = json.loads(result.content)

        except json.JSONDecodeError as e:
            logging.error(
                f"Error decoding JSON: {e}. Content: {result.content}"
            )
            return PhxAppOperation()

        return data if data else PhxAppOperation()
