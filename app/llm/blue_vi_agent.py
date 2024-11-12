import logging
from app.model import Message
from app.schemas import GptResponseSchema
from app.types.enum import InstructionTypes


class Agent:
    def __init__(self, model, db_manager, token_utils, history_window_size):
        self.model = model
        self.db_manager = db_manager
        self.token_utils = token_utils
        self.history_window_size = history_window_size

    def flag_personal_data(self, prompt: str) -> bool:
        """Flag personal data in the user prompt using the assistant role."""
        is_personal_data = self.model.assistant_role.check_for_personal_data(
            prompt
        )
        if is_personal_data:
            logging.warning(f"Personal data detected: {prompt}")
        return is_personal_data

    def get_conversation_history(self, user_conversation_id: int):
        """Retrieve and trim the conversation history."""
        conversation_history = (
            self.db_manager.get_messages_by_user_conversation_id(
                user_conversation_id
            )[-self.history_window_size :]
        )
        return self.token_utils.trim_history_to_fit_tokens(
            conversation_history
        )

    def generate_response(self, conversation_history) -> GptResponseSchema:
        """Generate a response using the user role."""
        return self.model.user_role.get_chat_response(conversation_history)

    def handle_operation_instructions(
        self, conversation_history
    ) -> GptResponseSchema:
        """Generate a response for operation instructions using the assistant role."""
        assistant_response = (
            self.model.assistant_role.handle_operation_instructions(
                conversation_history
            )
        )
        logging.info(f"Assistant response: {assistant_response}")
        return self.model.user_role.get_chat_response(conversation_history)

    def handle_conversation(self, message: Message) -> GptResponseSchema:
        """Evaluate the prompt, flag data, retrieve history, and generate a response."""
        # Flag personal data if detected
        if self.flag_personal_data(message.content):
            self.db_manager.flag_message(message.id)

        # Get trimmed conversation history
        conversation_history = self.get_conversation_history(
            message.user_conversation_id
        )

        # Identify if this prompt needs special handling
        instruction_type = self.model.assistant_role.identify_instruction_type(
            message.content
        )

        if instruction_type == InstructionTypes.OPERATION.value:
            return self.handle_operation_instructions(conversation_history)

        return self.generate_response(conversation_history)
