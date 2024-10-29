from unittest.mock import patch, MagicMock
import pytest
from app.llm import BlueViGptModel
from app.config.config_env import GGUF_MODEL, MODEL_NAME


@pytest.fixture(scope="class")
def blue_vi_gpt_model():
    return BlueViGptModel()


# Mock test for loading the model
@patch("llama_cpp.Llama.from_pretrained")
def test_load_model(mock_from_pretrained, blue_vi_gpt_model):
    mock_model = MagicMock()
    mock_from_pretrained.return_value = mock_model
    test_model = blue_vi_gpt_model.load_model()

    assert test_model == mock_model
    # Check only relevant arguments
    mock_from_pretrained.assert_called_once()
    called_args, called_kwargs = mock_from_pretrained.call_args
    assert called_kwargs["repo_id"] == MODEL_NAME
    assert called_kwargs["filename"] == GGUF_MODEL


class TestGetResponse:
    def test_get_response_with_real_model(self, blue_vi_gpt_model):
        user_message = "Hello, how are you?"
        messages = [{"role": "user", "content": user_message}]
        response = blue_vi_gpt_model.get_response(messages)
        assert (
            response is not None and len(response) > 0
        ), "Model failed to return a valid response"

    def test_get_response_with_blue_vi_answer(self, blue_vi_gpt_model):
        user_message = "who are you?"
        messages = [{"role": "user", "content": user_message}]
        response = blue_vi_gpt_model.get_response(messages)
        # Assert that the response contains references to "blueVi", "blueVi-GPT", or "Visma Verzuim"
        assert (
            "blueVi" in response
            or "blueVi-GPT" in response
            or "Visma Verzuim" in response
        )


class TestAnonymization:
    def test_get_anonymized_name(self, blue_vi_gpt_model):
        user_message = "John Doe's email is J.Simpson@netwrix.com."
        response = blue_vi_gpt_model.get_anonymized_message(user_message)
        assert (
            "[NAME_1]" in response or "John Doe" not in response
        ), "Test failed: Either '[NAME_1]' token not found or original name is present."

    def test_get_anonymized_email(self, blue_vi_gpt_model):
        user_message = "John Doe's email is J.Simpson@netwrix.com."
        response = blue_vi_gpt_model.get_anonymized_message(user_message)

        assert (
            "[EMAIL_1]" in response or "J.Simpson@netwrix.com" not in response
        ), "Test failed: Either '[EMAIL_1]' token not found or original email is present."

    def test_get_anonymized_bsn(self, blue_vi_gpt_model):
        user_message = "His BSN is 123456789."
        response = blue_vi_gpt_model.get_anonymized_message(user_message)

        assert (
            "[BSN_1]" in response or "123456789" not in response
        ), "Test failed: Either '[BSN_1]' token not found or original BSN is present."

    def test_get_anonymized_address(self, blue_vi_gpt_model):
        user_message = "His home address is 10 Langelo."
        response = blue_vi_gpt_model.get_anonymized_message(user_message)

        assert (
            "[ADDRESS_1]" in response or "10 Langelo" not in response
        ), "Test failed: Either '[ADDRESS_1]' token not found or original address is present."

    def test_get_anonymized_zip(self, blue_vi_gpt_model):
        user_message = "His ZIP code is 7666MC."
        response = blue_vi_gpt_model.get_anonymized_message(user_message)

        assert (
            "[ZIP_1]" in response or "7666MC" not in response
        ), "Test failed: Either '[ZIP_1]' token not found or original ZIP code is present."

    def test_get_anonymized_mastercard(self, blue_vi_gpt_model):
        user_message = "His MasterCard number is 5258704108753590."
        response = blue_vi_gpt_model.get_anonymized_message(user_message)

        assert (
            "[MASTERCARD_1]" in response or "5258704108753590" not in response
        ), "Test failed: Either '[MASTERCARD_1]' token not found or original MasterCard number is present."

    def test_get_anonymized_visa(self, blue_vi_gpt_model):
        user_message = "His Visa number is 4563-7568-5698-4587."
        response = blue_vi_gpt_model.get_anonymized_message(user_message)

        assert (
            "[VISA_1]" in response or "4563-7568-5698-4587" not in response
        ), "Test failed: Either '[VISA_1]' token not found or original Visa number is present."

    def test_get_anonymized_iban(self, blue_vi_gpt_model):
        user_message = "His IBAN number is NL91ABNA0417164300."
        response = blue_vi_gpt_model.get_anonymized_message(user_message)

        assert (
            "[IBAN_1]" in response or "NL91ABNA0417164300" not in response
        ), "Test failed: Either '[IBAN_1]' token not found or original IBAN number is present."

    def test_get_anonymized_dob(self, blue_vi_gpt_model):
        user_message = "His date of birth is 01/01/1990."
        response = blue_vi_gpt_model.get_anonymized_message(user_message)

        assert (
            "[DOB_1]" in response or "01/01/1990" not in response
        ), "Test failed: Either '[DOB_1]' token not found or original date of birth is present."

    def test_get_anonymized_ip_address(self, blue_vi_gpt_model):
        user_message = "His IP address is 192.168.1.1."
        response = blue_vi_gpt_model.get_anonymized_message(user_message)

        assert (
            "[IP_ADDRESS_1]" in response or "192.168.1.1" not in response
        ), "Test failed: Either '[IP_ADDRESS_1]' token not found or original IP address is present."

    def test_get_anonymized_multiple_fields(self, blue_vi_gpt_model):
        user_message = (
            "John Doe's email is J.Simpson@netwrix.com. His BSN is 123456789. His home address is 10 Langelo. His ZIP code is 7666MC. His MasterCard number is 5258704108753590. His Visa number is 4563-7568-5698-4587. His IBAN number is NL91ABNA0417164300. His date of birth is 01/01/1990. His IP address is 192.168.1.1."
        )
        response = blue_vi_gpt_model.get_anonymized_message(user_message)
        assert (
            "[NAME_1]" in response or "John Doe" not in response
        ), "Test failed for Name."
        assert (
            "[EMAIL_1]" in response or "J.Simpson@netwrix.com" not in response
        ), "Test failed for Email."
        assert (
            "[BSN_1]" in response or "123456789" not in response
        ), "Test failed for BSN."
        assert (
            "[ADDRESS_1]" in response or "10 Langelo" not in response
        ), "Test failed for Address."
        assert (
            "[ZIP_1]" in response or "7666MC" not in response
        ), "Test failed for ZIP."
        assert (
            "[MASTERCARD_1]" in response or "5258704108753590" not in response
        ), "Test failed for MasterCard."
        assert (
            "[VISA_1]" in response or "4563-7568-5698-4587" not in response
        ), "Test failed for Visa."
        assert (
            "[IBAN_1]" in response or "NL91ABNA0417164300" not in response
        ), "Test failed for IBAN."
        assert (
            "[DOB_1]" in response or "01/01/1990" not in response
        ), "Test failed for DOB."
        assert (
            "[IP_ADDRESS_1]" in response or "192.168.1.1" not in response
        ), "Test failed for IP Address."
