import pytest
from app.llm import BlueViGptModel
from app.model.models import Message
from app.types.enum.gpt import Role


@pytest.fixture(scope="class")
def blue_vi_gpt_model():
    return BlueViGptModel()


class TestGetResponse:
    @pytest.mark.asyncio
    async def test_get_response_with_real_model(self, blue_vi_gpt_model):
        user_message = "Hello, how are you?"
        messages = [Message(role=Role.USER, content=user_message)]
        response = blue_vi_gpt_model.assistant.generate_user_response_with_custom_instruction(
            messages
        )

        assert (
            response is not None
            and hasattr(response, 'content')
            and len(response.content) > 0
        ), "Model failed to return a valid response"


class TestAnonymization:
    @pytest.mark.asyncio
    async def test_get_anonymized_name(self, blue_vi_gpt_model):
        user_message = "John Doe's email is J.Simpson@netwrix.com."
        response = blue_vi_gpt_model.assistant.get_anonymized_message(
            user_message
        )

        assert (
            "[NAME_1]" in response.content
            or "John Doe" not in response.content
        ), "Test failed: Either '[NAME_1]' token not found or original name is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_email(self, blue_vi_gpt_model):
        user_message = "John Doe's email is J.Simpson@netwrix.com."
        response = blue_vi_gpt_model.assistant.get_anonymized_message(
            user_message
        )

        assert (
            "[EMAIL_1]" in response.content
            or "J.Simpson@netwrix.com" not in response.content
        ), "Test failed: Either '[EMAIL_1]' token not found or original email is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_bsn(self, blue_vi_gpt_model):
        user_message = "His BSN is 123456789."
        response = blue_vi_gpt_model.assistant.get_anonymized_message(
            user_message
        )

        assert (
            "[BSN_1]" in response.content
            or "123456789" not in response.content
        ), "Test failed: Either '[BSN_1]' token not found or original BSN is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_address(self, blue_vi_gpt_model):
        user_message = "His home address is 10 Langelo."
        response = blue_vi_gpt_model.assistant.get_anonymized_message(
            user_message
        )

        assert (
            "[ADDRESS_1]" in response.content
            or "10 Langelo" not in response.content
        ), "Test failed: Either '[ADDRESS_1]' token not found or original address is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_zip(self, blue_vi_gpt_model):
        user_message = "His ZIP code is 7666MC."
        response = blue_vi_gpt_model.assistant.get_anonymized_message(
            user_message
        )

        assert (
            "[ZIP_1]" in response.content or "7666MC" not in response.content
        ), "Test failed: Either '[ZIP_1]' token not found or original ZIP code is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_mastercard(self, blue_vi_gpt_model):
        user_message = "His MasterCard number is 5258704108753590."
        response = blue_vi_gpt_model.assistant.get_anonymized_message(
            user_message
        )

        assert (
            "[MASTERCARD_1]" in response.content
            or "5258704108753590" not in response.content
        ), "Test failed: Either '[MASTERCARD_1]' token not found or original MasterCard number is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_visa(self, blue_vi_gpt_model):
        user_message = "His Visa number is 4563-7568-5698-4587."
        response = blue_vi_gpt_model.assistant.get_anonymized_message(
            user_message
        )

        assert (
            "[VISA_1]" in response.content
            or "4563-7568-5698-4587" not in response.content
        ), "Test failed: Either '[VISA_1]' token not found or original Visa number is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_iban(self, blue_vi_gpt_model):
        user_message = "His IBAN number is NL91ABNA0417164300."
        response = blue_vi_gpt_model.assistant.get_anonymized_message(
            user_message
        )

        assert (
            "[IBAN_1]" in response.content
            or "NL91ABNA0417164300" not in response.content
        ), "Test failed: Either '[IBAN_1]' token not found or original IBAN number is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_dob(self, blue_vi_gpt_model):
        user_message = "His date of birth is 01/01/1990."
        response = blue_vi_gpt_model.assistant.get_anonymized_message(
            user_message
        )

        assert (
            "[DOB_1]" in response.content
            or "01/01/1990" not in response.content
        ), "Test failed: Either '[DOB_1]' token not found or original date of birth is present."

    @pytest.mark.asyncio
    async def test_get_anonymized_ip_address(self, blue_vi_gpt_model):
        user_message = "His IP address is 192.168.1.1."
        response = blue_vi_gpt_model.assistant.get_anonymized_message(
            user_message
        )

        assert (
            "[IP_ADDRESS_1]" in response.content
            or "192.168.1.1" not in response.content
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
        response = blue_vi_gpt_model.assistant.get_anonymized_message(
            user_message
        )

        assert (
            "[NAME_1]" in response.content
            or "John Doe" not in response.content
        ), "Test failed for Name."
        assert (
            "[EMAIL_1]" in response.content
            or "J.Simpson@netwrix.com" not in response.content
        ), "Test failed for Email."
        assert (
            "[BSN_1]" in response.content
            or "123456789" not in response.content
        ), "Test failed for BSN."
        assert (
            "[ADDRESS_1]" in response.content
            or "10 Langelo" not in response.content
        ), "Test failed for Address."
        assert (
            "[ZIP_1]" in response.content or "7666MC" not in response.content
        ), "Test failed for ZIP."
        assert (
            "[MASTERCARD_1]" in response.content
            or "5258704108753590" not in response.content
        ), "Test failed for MasterCard."
        assert (
            "[VISA_1]" in response.content
            or "4563-7568-5698-4587" not in response.content
        ), "Test failed for Visa."
        assert (
            "[IBAN_1]" in response.content
            or "NL91ABNA0417164300" not in response.content
        ), "Test failed for IBAN."
        assert (
            "[DOB_1]" in response.content
            or "01/01/1990" not in response.content
        ), "Test failed for DOB."
        assert (
            "[IP_ADDRESS_1]" in response.content
            or "192.168.1.1" not in response.content
        ), "Test failed for IP Address."
