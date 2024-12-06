import json
import logging
from typing import List, Optional, Tuple

from langchain_huggingface import ChatHuggingFace
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import (
    generate_gbnf_grammar_and_documentation,
)

from app.model.models import Message
from app.schemas import GptResponseSchema, PhxAppOperation, DecisionInstruction
from app.types.enum.gpt import Role
from app.types.enum.http_status import HTTPStatus
from app.types.enum.instruction import CRUD
from app.types.enum.instruction.blue_vi_gpt_instruction import (
    BlueViInstructionEnum,
)
from app.utils import (
    convert_blue_vi_response_to_schema,
    TokenUtils,
)


class BlueViGptAssistant:
    def __init__(self, llm):
        """Initialize BlueViGptAssistant with LLM and tokenizer."""
        self.llm = llm
        self.token_utils = TokenUtils(self.llm)
        self.chat = ChatHuggingFace(llm=self.llm, verbose=True)

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
            prompt = self.create_prompt(full_conversation_history)

            response = self.chat.invoke(prompt)

            return convert_blue_vi_response_to_schema(response.content)

        except Exception as e:
            logging.error(
                f"Unexpected error while generating chat response in assistant: {e}"
            )
            return GptResponseSchema(
                status=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                content="Sorry, something went wrong while generating a response.",
            )

    @staticmethod
    def create_prompt(conversation_history: List[Tuple[str, str]]) -> str:
        """Convert conversation history to a formatted prompt."""
        return "\n".join(
            [f"{role}: {content}" for role, content in conversation_history]
        )

    async def get_anonymized_message(
        self, user_message: str
    ) -> GptResponseSchema:
        """Anonymize the user message."""
        instruction = BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value
        prompt = self.create_prompt(
            [(Role.SYSTEM.value, instruction), (Role.USER.value, user_message)]
        )
        response = self.chat.invoke(prompt)
        return convert_blue_vi_response_to_schema(response.content)

    async def identify_instruction_type(
        self, conversation_history: List[Message]
    ) -> DecisionInstruction:
        """Identify the type of instruction based on the conversation history and the prompt context."""
        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
            [DecisionInstruction]
        )

        instruction = f"{BlueViInstructionEnum.BLUE_VI_SYSTEM_HANDLE_INSTRUCTION_DECISION.value} \n{documentation}"
        messages = self.create_prompt(
            [(Role.SYSTEM.value, instruction)] + conversation_history
        )
        response = self.chat.invoke(messages)

        result = convert_blue_vi_response_to_schema(response.content)
        logging.info('result in identify_instruction_type')
        logging.info(result)

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
        # Log crud operation
        logging.info('crud in handle_phx_operation')
        logging.info(crud)

        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
            [PhxAppOperation]
        )
        instruction = f"{BlueViInstructionEnum.BLUE_VI_SYSTEM_HANDLE_OPERATION_PROCESS.value}: \n\n{documentation}"
        messages = self.create_prompt(
            [(Role.SYSTEM.value, instruction)] + conversation_history
        )

        response = self.chat.invoke(messages)

        result = convert_blue_vi_response_to_schema(response.content)

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
