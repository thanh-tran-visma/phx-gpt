import pytest
from app.llm import BlueViGptModel
from app.model.models import Message
from app.types.enum.gpt import Role
from app.types.enum.instruction import BlueViInstructionEnum


@pytest.fixture(scope="class")
def blue_vi_gpt_model():
    return BlueViGptModel()


class TestGetResponse:
    @pytest.mark.asyncio
    async def test_get_response_with_real_model(self, blue_vi_gpt_model):
        user_message = "Hello, how are you?"
        messages = [
            Message(role=Role.USER, content=user_message)
        ]  # Using Message directly
        response = await blue_vi_gpt_model.assistant.generate_user_response_with_custom_instruction(
            messages
        )

        assert (
            response is not None
            and hasattr(response, 'content')
            and len(response.content) > 0
        ), "Model failed to return a valid response"

    @pytest.mark.asyncio
    async def test_get_response_with_blue_vi_answer(self, blue_vi_gpt_model):
        user_message = "what is your name?"
        messages = [Message(role=Role.USER, content=user_message)]
        conversation_history = [(msg.role, msg.content) for msg in messages]
        response = await blue_vi_gpt_model.assistant.generate_user_response_with_custom_instruction(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_DEFAULT_INSTRUCTION.value,
        )

        # Assert that the response content contains references
        assert (
            "blueVi" in response.content
            or "blueVi-GPT" in response.content
            or "Visma Verzuim" in response.content
        ), "Response does not mention the expected terms"


class TestAnonymization:
    @pytest.mark.asyncio
    async def test_get_anonymized_name(self, blue_vi_gpt_model):
        user_message = "John Doe's email is J.Simpson@netwrix.com."
        messages = [Message(role=Role.USER, content=user_message)]
        conversation_history = [(msg.role, msg.content) for msg in messages]
        response = await blue_vi_gpt_model.assistant.generate_user_response_with_custom_instruction(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value,
        )
        assert (
            "[NAME_1]" in response.content or "NAME_1" in response.content
        ), "Test failed: Either '[NAME_1]' token not found or original name is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_email(self, blue_vi_gpt_model):
        user_message = "John Doe's email is J.Simpson@netwrix.com."
        messages = [Message(role=Role.USER, content=user_message)]
        conversation_history = [(msg.role, msg.content) for msg in messages]
        response = await blue_vi_gpt_model.assistant.generate_user_response_with_custom_instruction(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value,
        )

        assert (
            "[EMAIL_1]" in response.content or "EMAIL_1" in response.content
        ), "Test failed: Either '[EMAIL_1]' token not found or original email is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_bsn(self, blue_vi_gpt_model):
        user_message = "His BSN is 123456789."
        messages = [Message(role=Role.USER, content=user_message)]
        conversation_history = [(msg.role, msg.content) for msg in messages]
        response = await blue_vi_gpt_model.assistant.generate_user_response_with_custom_instruction(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value,
        )

        assert (
            "[BSN_1]" in response.content or "BSN_1" in response.content
        ), "Test failed: Either '[BSN_1]' token not found or original BSN is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_address(self, blue_vi_gpt_model):
        user_message = "His home address is 10 Langelo."
        messages = [Message(role=Role.USER, content=user_message)]
        conversation_history = [(msg.role, msg.content) for msg in messages]
        response = await blue_vi_gpt_model.assistant.generate_user_response_with_custom_instruction(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value,
        )

        assert (
            "[ADDRESS_1]" in response.content
            or "ADDRESS_1" in response.content
        ), "Test failed: Either '[ADDRESS_1]' token not found or original address is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_zip(self, blue_vi_gpt_model):
        user_message = "His ZIP code is 7666MC."
        messages = [Message(role=Role.USER, content=user_message)]
        conversation_history = [(msg.role, msg.content) for msg in messages]
        response = await blue_vi_gpt_model.assistant.generate_user_response_with_custom_instruction(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value,
        )

        assert (
            "[ZIP_1]" in response.content or "ZIP_1" in response.content
        ), "Test failed: Either '[ZIP_1]' token not found or original ZIP code is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_mastercard(self, blue_vi_gpt_model):
        user_message = "His MasterCard number is 5258704108753590."
        messages = [Message(role=Role.USER, content=user_message)]
        conversation_history = [(msg.role, msg.content) for msg in messages]
        response = await blue_vi_gpt_model.assistant.generate_user_response_with_custom_instruction(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value,
        )

        assert (
            "[MASTERCARD_1]" in response.content
            or "MASTERCARD_1" in response.content
        ), "Test failed: Either '[MASTERCARD_1]' token not found or original MasterCard number is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_visa(self, blue_vi_gpt_model):
        user_message = "His Visa number is 4563-7568-5698-4587."
        messages = [Message(role=Role.USER, content=user_message)]
        conversation_history = [(msg.role, msg.content) for msg in messages]
        response = await blue_vi_gpt_model.assistant.generate_user_response_with_custom_instruction(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value,
        )

        assert (
            "[VISA_1]" in response.content or "VISA_1" in response.content
        ), "Test failed: Either '[VISA_1]' token not found or original Visa number is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_iban(self, blue_vi_gpt_model):
        user_message = "His IBAN number is NL91ABNA0417164300."
        messages = [Message(role=Role.USER, content=user_message)]
        conversation_history = [(msg.role, msg.content) for msg in messages]
        response = await blue_vi_gpt_model.assistant.generate_user_response_with_custom_instruction(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value,
        )

        assert (
            "[IBAN_1]" in response.content or "IBAN_1" in response.content
        ), "Test failed: Either '[IBAN_1]' token not found or original IBAN number is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_dob(self, blue_vi_gpt_model):
        user_message = "His date of birth is 01/01/1990."
        messages = [Message(role=Role.USER, content=user_message)]
        conversation_history = [(msg.role, msg.content) for msg in messages]
        response = await blue_vi_gpt_model.assistant.generate_user_response_with_custom_instruction(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value,
        )
        assert (
            "[DOB_1]" in response.content or "DOB_1" in response.content
        ), "Test failed: Either '[DOB_1]' token not found or original date of birth is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_ip_address(self, blue_vi_gpt_model):
        user_message = "His IP address is 192.168.1.1."
        messages = [Message(role=Role.USER, content=user_message)]
        conversation_history = [(msg.role, msg.content) for msg in messages]
        response = await blue_vi_gpt_model.assistant.generate_user_response_with_custom_instruction(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value,
        )

        assert (
            "[IP_ADDRESS_1]" in response.content
            or "IP_ADDRESS_1" in response.content
        ), "Test failed: Either '[IP_ADDRESS_1]' token not found or original IP address is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_multiple_fields(self, blue_vi_gpt_model):
        user_message = (
            "John Doe's email is J.Simpson@netwrix.com. "
            "His BSN is 123456789. "
            "His home address is 10 Langelo. "
            "His ZIP code is 7666MC. "
            "His MasterCard number is 5258704108753590. "
            "His Visa number is 4563-7568-5698-4587. "
            "His IBAN number is NL91ABNA0417164300. "
            "His date of birth is 01/01/1990. "
            "His IP address is 192.168.1.1."
        )
        messages = [Message(role=Role.USER, content=user_message)]
        conversation_history = [(msg.role, msg.content) for msg in messages]
        response = await blue_vi_gpt_model.assistant.generate_user_response_with_custom_instruction(
            conversation_history,
            BlueViInstructionEnum.BLUE_VI_SYSTEM_ANONYMIZE_DATA.value,
        )

        assert (
            "[NAME_1]" in response.content or "NAME_1" in response.content
        ), "Test failed for Name."
        assert (
            "[EMAIL_1]" in response.content
            or "ALT_EMAIL_1" in response.content
        ), "Test failed for Email."
        assert (
            "[BSN_1]" in response.content or "ALT_BSN_1" in response.content
        ), "Test failed for BSN."
        assert (
            "[ADDRESS_1]" in response.content
            or "ALT_ADDRESS_1" in response.content
        ), "Test failed for Address."
        assert (
            "[ZIP_1]" in response.content or "ALT_ZIP_1" in response.content
        ), "Test failed for ZIP."
        assert (
            "[MASTERCARD_1]" in response.content
            or "ALT_MASTERCARD_1" in response.content
        ), "Test failed for MasterCard."
        assert (
            "[VISA_1]" in response.content or "ALT_VISA_1" in response.content
        ), "Test failed for Visa."
        assert (
            "[IBAN_1]" in response.content or "ALT_IBAN_1" in response.content
        ), "Test failed for IBAN."
        assert (
            "[DOB_1]" in response.content or "ALT_DOB_1" in response.content
        ), "Test failed for DOB."
        assert (
            "[IP_ADDRESS_1]" in response.content
            or "ALT_IP_ADDRESS_1" in response.content
        ), "Test failed for IP Address."
