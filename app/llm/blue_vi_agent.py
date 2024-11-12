import logging
from app.model import Message, User
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

        # Ensure we trim history to fit within token limits
        return self.token_utils.trim_history_to_fit_tokens(
            conversation_history
        )

    def generate_response(self, conversation_history) -> GptResponseSchema:
        """Generate a response using the user role."""
        if not conversation_history:
            logging.warning("Empty conversation history provided.")
        return self.model.user_role.get_chat_response(conversation_history)

    def handle_operation_instructions(
        self, uuid: str, user_input: str
    ) -> GptResponseSchema:
        """Generate a response for operation instructions using the assistant role."""
        # Generate the initial operation schema from assistant role
        operation_schema = (
            self.model.assistant_role.handle_operation_instructions(
                uuid, user_input
            )
        )

        logging.info(
            f"Assistant response for operation instructions: {operation_schema}"
        )

        # Use the operation schema for further assistant response generation if necessary
        if operation_schema:
            logging.info(f"operation_schema: {operation_schema}")

            # Call assistant again to fill in missing data
            updated_response = self.model.assistant_role.operation_processing(
                operation_schema
            )

            # Assuming we get the missing fields from the assistant response
            logging.info(f"Updated assistant response: {updated_response}")

        # Return the operation schema or GptResponseSchema
        return GptResponseSchema(content=operation_schema.name)

    def handle_conversation(
        self, user: User, message: Message
    ) -> GptResponseSchema:
        """Evaluate the prompt, flag data, retrieve history, and generate a response."""
        try:
            # Flag personal data if detected
            if self.flag_personal_data(message.content):
                self.db_manager.flag_message(message.id)

            # Get trimmed conversation history
            conversation_history = self.get_conversation_history(
                message.user_conversation_id
            )

            # Identify if this prompt needs special handling
            instruction_type = (
                self.model.assistant_role.identify_instruction_type(
                    message.content
                )
            )

            if instruction_type == InstructionTypes.OPERATION.value:
                # Extract the user input from the last message in the history
                user_input = (
                    conversation_history[-1].content
                    if conversation_history
                    else message.content
                )
                return self.handle_operation_instructions(
                    user.uuid, user_input
                )

            return self.generate_response(conversation_history)

        except Exception as e:
            logging.error(f"Error handling conversation: {e}")
            return GptResponseSchema(
                content="An error occurred while processing the conversation."
            )
