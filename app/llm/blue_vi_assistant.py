import json
import logging
from typing import List, Optional, Tuple

from langchain_core.output_parsers import PydanticOutputParser, SimpleJsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace
from pydantic import ValidationError

from app.model import Message
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

    def generate_user_response_with_custom_instruction(
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

    def get_anonymized_message(
        self, user_message: str
    ) -> GptResponseSchema:
        """Anonymize the user message."""
        instruction = BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value
        prompt = self.create_prompt(
            [(Role.SYSTEM.value, instruction), (Role.USER.value, user_message)]
        )
        response = self.chat.invoke(prompt)
        return convert_blue_vi_response_to_schema(response.content)

    def identify_instruction_type(
            self, conversation_history: List[Message]
    ) -> DecisionInstruction:
        """Identify the type of instruction based on the conversation history and the prompt context."""

        # Initialize the PydanticOutputParser with the DecisionInstruction model
        parser = PydanticOutputParser(pydantic_object=DecisionInstruction)

        # Create the prompt template
        prompt = PromptTemplate(
            template="Identify the most suitable instruction type.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        logging.info('Prompt in identify_instruction_type:')
        logging.info(prompt)

        # Initialize the SimpleJsonOutputParser
        json_parser = SimpleJsonOutputParser()

        # Compose the pipeline: Prompt -> Model -> JSON Parser
        prompt_and_model = prompt | self.chat | json_parser

        try:
            # Generate the output by invoking the model with the conversation history
            output = prompt_and_model.invoke({"query": f"{conversation_history}"})

            logging.info('Raw output in identify_instruction_type:')
            logging.info(output)

            # Parse the output using PydanticOutputParser
            parsed_result = parser.parse(output)

            # Log parsed results
            logging.info('Parsed result in identify_instruction_type:')
            logging.info(parsed_result)

            return parsed_result

        except ValidationError as ve:
            logging.error(f"Validation error while parsing output: {ve}")
        except json.JSONDecodeError as jde:
            logging.error(f"JSON decode error: {jde}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

        # Fallback to a default DecisionInstruction object if an error occurs
        return DecisionInstruction()


    def handle_phx_operation(
        self, conversation_history: List[Tuple[str, str]], crud: CRUD
    ) -> PhxAppOperation:
        """Generates an operation schema based on the user's conversation history and model response."""
        # Log crud operation
        logging.info('crud in handle_phx_operation')
        logging.info(crud)
        instruction = f"{BlueViInstructionEnum.BLUE_VI_SYSTEM_HANDLE_OPERATION_PROCESS.value}"
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
