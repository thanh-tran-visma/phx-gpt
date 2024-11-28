from enum import Enum


class BlueViResponseHandling(str, Enum):
    HANDLE_OPERATION_SUCCESS = (
        "Inform the user in a warm and friendly tone that the {operation} was completed successfully. "
        "Use the user name: {user_name} to personalize the response. "
        "Summarize the main details of the operation in a friendly and engaging way, focusing on key points such as what was done, any important outcomes, and relevant context from the data provided. "
        "For example, mention the operation name, description, duration, and methods of delivery as they relate to the {crud} action. "
        "details: {details}"
        "Ensure the response feels natural and personable, making it easy to understand and engaging, without sounding robotic or formulaic. "
        "Avoid mentioning any information outside the supplied input."
    )
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
