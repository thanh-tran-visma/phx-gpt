import json
import logging
from typing import List, Optional, Tuple
from llama_cpp import Llama, LlamaGrammar
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import (
    generate_gbnf_grammar_and_documentation,
)

from app.model import Message
from app.schemas import GptResponseSchema, PhxAppOperation, DecisionInstruction
from app.types.enum.gpt import Role
from app.types.enum.http_status import HTTPStatus
from app.types.enum.instruction import CRUD
from app.types.enum.instruction.blue_vi_gpt_instruction import (
    BlueViInstructionEnum,
)
from app.utils import (
    get_blue_vi_response,
    convert_blue_vi_response_to_schema,
    TokenUtils,
)


class BlueViGptAssistant:
    def __init__(self, llm: Llama):
        self.llm = llm
        self.token_utils = TokenUtils(self.llm)

    async def generate_user_response_with_custom_instruction(
        self,
        conversation_history: List[Tuple[str, str]],
        instruction: Optional[str] = None,
    ) -> GptResponseSchema:
        """Generate a response from the model based on conversation history for the user role, optionally with a custom instruction."""
        try:
            logging.info(
                "system_instruction in generate_user_response_with_custom_instruction:"
            )
            system_instruction = (
                instruction
                if instruction
                else BlueViInstructionEnum.BLUE_VI_SYSTEM_DEFAULT_INSTRUCTION.value
            )
            logging.info(system_instruction)
            if isinstance(system_instruction, Message):
                system_instruction = system_instruction.content
            if isinstance(conversation_history, dict):
                role = Role.USER.value
                content_list = conversation_history['content']
                conversation_history = [
                    (role, str(content)) for content in content_list
                ]
            full_conversation_history = [
                (Role.SYSTEM.value, system_instruction)
            ] + conversation_history

            # Log the full conversation history
            logging.info("Full conversation history passed to the model:")
            logging.info(full_conversation_history)

            # Generate the response
            response = await get_blue_vi_response(
                self.llm, full_conversation_history
            )

            # Convert and return the response as GptResponseSchema
            return convert_blue_vi_response_to_schema(response)

        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response in assistant: {e}"
            )
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content="Sorry, something went wrong while generating a response.",
            )

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
    ) -> DecisionInstruction:
        """
        Identify the type of instruction based on the conversation history and the prompt context.
        """
        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
            [DecisionInstruction]
        )
        grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)
        # Convert conversation history to tuples and prepare messages
        instruction = (
            f"{BlueViInstructionEnum.BLUE_VI_SYSTEM_HANDLE_INSTRUCTION_DECISION.value} \n"
            f"{documentation}"
        )
        logging.info('instruction in identify_instruction_type')
        logging.info(instruction)
        messages = [(Role.SYSTEM.value, instruction)] + conversation_history
        # Get response from the model
        response = await get_blue_vi_response(self.llm, messages, grammar)

        # Convert response to schema
        result = convert_blue_vi_response_to_schema(response)

        if not result.content:
            return DecisionInstruction()
        try:
            data: DecisionInstruction = json.loads(result.content)

        except json.JSONDecodeError as e:
            logging.error(
                f"Error decoding JSON: {e}. Content: {result.content}"
            )
            return DecisionInstruction()

        return data if data else DecisionInstruction()

    async def handle_phx_operation(
        self, conversation_history: List[Tuple[str, str]], crud: CRUD
    ) -> PhxAppOperation:
        """Generates an operation schema based on the user's conversation history and model response."""
        # TODO:
        logging.info('crud in handle_phx_operation')
        logging.info('crud in handle_phx_operation')
        # Define the instruction
        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
            [PhxAppOperation]
        )
        grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)

        instruction = f"{BlueViInstructionEnum.BLUE_VI_SYSTEM_HANDLE_OPERATION_PROCESS.value}: \n\n {documentation}"
        # Create the messages structure with instruction and conversation history
        messages = [(Role.SYSTEM.value, instruction)] + conversation_history

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
