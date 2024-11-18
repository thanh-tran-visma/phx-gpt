from enum import Enum


class BlueViUnexpectedResponseHandling(str, Enum):
    HANDLE_OPERATION_ERROR = (
        "An error occurred while attempting to create a new operation. The operation could not be created by BlueVi-GPT AI at the moment.\n\n"
        "However, you can still create the operation manually. To do this, follow these steps:\n\n"
        "1. Navigate to the 'Operations' page within the BlueVi environment.\n"
        "2. Click on the 'Create Operation' button.\n"
        "3. Fill in the required details, including:\n"
        "\t* Operation name\n"
        "\t* Description\n"
        "\t* Duration\n"
        "\t* Rate (if applicable)\n"
        "\t* Methods of consultation (e.g., phone, email, in-person)\n"
        "4. Set up invoicing and payment options as needed.\n"
        "5. Assign any necessary workflows, integrations, or custom fields.\n"
        "6. Review and submit the operation for review and approval.\n\n"
        "By following these steps, you'll be able to create the operation manually and have it available for use within the BlueVi environment."
    )
