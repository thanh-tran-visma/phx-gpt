import json
from huggingface_hub import HfApi, create_repo
from app.config.config_env import HF_TOKEN
from typing import List
from app.types.dataset import DatasetEntry


class DatasetPreparer:
    def __init__(self):
        # Initialize an empty list to store multiple datasets
        self.data: List[DatasetEntry] = []

    @staticmethod
    def create_entry(instruction: str, data: List[dict]) -> List[DatasetEntry]:
        # Format each entry with the given instruction
        formatted_data = [
            DatasetEntry(
                role=entry.get("role"),
                instruction=instruction,
                input=json.dumps(entry["prompt"], ensure_ascii=False),
                output=entry["response"],
            )
            for entry in data
        ]
        return formatted_data

    def add_data(self, instruction: str, data: List[dict]):
        # Add formatted entries to the main data list
        entries = self.create_entry(instruction, data)
        self.data.extend(entries)

    def save_to_jsonl(self, file_path: str):
        # Write each entry as a JSON object in a new line
        with open(file_path, "w", encoding="utf-8") as f:
            for entry in self.data:
                json_str = json.dumps(entry.__dict__, ensure_ascii=False)
                f.write(f"{json_str}\n")


def main():
    anonymize_data = [
        # English Cases
        {
            "role": "user",
            "prompt": "Allice Smith's email is allice.smith@example.com. Her BSN is 12345678. Her address is 15 Elm Street, and her zip code is 12345.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. Her address is [ADDRESS_1], and her zip code is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "Sure! I have replaced the personal information with placeholders in your message.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. Her address is [ADDRESS_1], and her zip code is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "Robert Brown was born on 12-01-2024. His email is robert.brown@example.com, and his BSN is 87654321. He lives at 222 Ash St.",
            "response": "[NAME_1] was born on [DOB_1]. His email is [EMAIL_1], and his BSN is [BSN_1]. He lives at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Sure! I have replaced the personal information with placeholders in your message.",
            "response": "[NAME_1] was born on [DOB_1]. His email is [EMAIL_1], and his BSN is [BSN_1]. He lives at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Natalie Green's email is nat.green@example.com. Her birthday is 01/12/2024, and her postal code is 34567. She resides at 789 Birch Lane.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her birthday is [DOB_1], and her postal code is [ZIP_1]. She resides at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "I have anonymized Natalie Green's information as requested.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her birthday is [DOB_1], and her postal code is [ZIP_1]. She resides at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Mark Thompson's email is mark.thompson@example.com. His BSN is 54321098, and his home address is 654 Maple Drive. His date of birth is 12/01/2024.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His home address is [ADDRESS_1]. His date of birth is [DOB_1].",
        },
        {
            "role": "assistant",
            "prompt": "I've anonymized Mark Thompson's personal details.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His home address is [ADDRESS_1]. His date of birth is [DOB_1].",
        },
        {
            "role": "user",
            "prompt": "Linda Johnson's email is linda.johnson@example.com, and her BSN is 11223344. She lives at 321 Willow Way, with a postal code of 98765.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1], with a postal code of [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "I have replaced Linda Johnson's details with placeholders.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1], with a postal code of [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "Peter's date of birth is 12/31/2024. His email is peter@example.com. His home address is 98 Pine Street, and his zip is 45678.",
            "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1]. His home address is [ADDRESS_1], and his zip is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "Peter's information has been anonymized successfully.",
            "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1]. His home address is [ADDRESS_1], and his zip is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "Julia's email is julia@example.com, her BSN is 22334455. She was born on 01-12-2024, and she lives at 234 Oak Lane.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She was born on [DOB_1], and she lives at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "I have anonymized Julia's personal information.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She was born on [DOB_1], and she lives at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Tom's email is tom123@example.com, and his BSN is 99988877. He was born on 12/01/2024 and lives at 100 Cedar Ave.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. He was born on [DOB_1] and lives at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Tom's information has been anonymized successfully.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. He was born on [DOB_1] and lives at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "My cousin Anna's email is anna123@example.com. Her BSN is 77788899. She lives at 77 Elm St. Her birthday is 1st January 2024.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1]. Her birthday is [DOB_1].",
        },
        {
            "role": "assistant",
            "prompt": "Anna's details have been anonymized.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1]. Her birthday is [DOB_1].",
        },
        {
            "role": "user",
            "prompt": "Max's date of birth is 2024-01-01. His email is max@example.com. His postal code is 12345, and he lives at 789 Cedar Street.",
            "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1]. His postal code is [ZIP_1], and he lives at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Max's information has been anonymized.",
            "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1]. His postal code is [ZIP_1], and he lives at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Sarah Davis's email is sarah.davis@example.com. She was born on 01/12/2024. Her address is 789 Maple Street, and her postal code is 87654.",
            "response": "[NAME_1]'s email is [EMAIL_1]. She was born on [DOB_1]. Her address is [ADDRESS_1], and her postal code is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "I have replaced Sarah Davis's personal details with placeholders.",
            "response": "[NAME_1]'s email is [EMAIL_1]. She was born on [DOB_1]. Her address is [ADDRESS_1], and her postal code is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "Jacob Martin was born on 12/01/2024. His BSN is 34567890, and his email is jacob.martin@example.com. He lives at 321 Cedar Avenue.",
            "response": "[NAME_1] was born on [DOB_1]. His BSN is [BSN_1], and his email is [EMAIL_1]. He lives at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Jacob Martin's personal information has been anonymized.",
            "response": "[NAME_1] was born on [DOB_1]. His BSN is [BSN_1], and his email is [EMAIL_1]. He lives at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Emily White's email is emily.white@example.com. She was born on the 1st of December 2024. Her zip code is 54321, and her house number is 24.",
            "response": "[NAME_1]'s email is [EMAIL_1]. She was born on [DOB_1]. Her zip code is [ZIP_1], and her house number is [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "I have anonymized Emily White's details.",
            "response": "[NAME_1]'s email is [EMAIL_1]. She was born on [DOB_1]. Her zip code is [ZIP_1], and her house number is [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "My friend Michael was born on December 1st, 2024. His email is michael@example.com, and his postal code is 98765. His home address is 456 Birch Avenue.",
            "response": "[NAME_1] was born on [DOB_1]. His email is [EMAIL_1], and his postal code is [ZIP_1]. His home address is [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Michael's personal details have been anonymized.",
            "response": "[NAME_1] was born on [DOB_1]. His email is [EMAIL_1], and his postal code is [ZIP_1]. His home address is [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Lily Evans's birthday is 12/01/2024. Her email is lily.evans@example.com, and her BSN is 45678901. She lives at 101 Rosewood St.",
            "response": "[NAME_1]'s birthday is [DOB_1]. Her email is [EMAIL_1], and her BSN is [BSN_1]. She lives at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Lily Evans's personal details have been anonymized.",
            "response": "[NAME_1]'s birthday is [DOB_1]. Her email is [EMAIL_1], and her BSN is [BSN_1]. She lives at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Contact John, who was born on 01/12/2024, at john.doe@example.com. His home address is 789 Oak Drive, and his postal code is 10234.",
            "response": "Contact [NAME_1], who was born on [DOB_1], at [EMAIL_1]. His home address is [ADDRESS_1], and his postal code is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "John's personal details have been anonymized.",
            "response": "Contact [NAME_1], who was born on [DOB_1], at [EMAIL_1]. His home address is [ADDRESS_1], and his postal code is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "Sophia Green's birthday is 1st December, 2024. Her BSN is 98765432, and her email is sophia.green@example.com. She lives at 888 Willow Lane.",
            "response": "[NAME_1]'s birthday is [DOB_1]. Her BSN is [BSN_1], and her email is [EMAIL_1]. She lives at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Sophia Green's details have been anonymized.",
            "response": "[NAME_1]'s birthday is [DOB_1]. Her BSN is [BSN_1], and her email is [EMAIL_1]. She lives at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Ryan's date of birth is December 01, 2024. His zip code is 54321, and his email is ryan@example.com. His BSN is 23456789.",
            "response": "[NAME_1]'s date of birth is [DOB_1]. His zip code is [ZIP_1], and his email is [EMAIL_1]. His BSN is [BSN_1].",
        },
        {
            "role": "assistant",
            "prompt": "Ryan's personal details have been anonymized.",
            "response": "[NAME_1]'s date of birth is [DOB_1]. His zip code is [ZIP_1], and his email is [EMAIL_1]. His BSN is [BSN_1].",
        },
        {
            "role": "user",
            "prompt": "Jane Smith's birthday is 01/12/2024. She lives at 123 Palm Street, and her postal code is 56789. Her email is jane.smith@example.com.",
            "response": "[NAME_1]'s birthday is [DOB_1]. She lives at [ADDRESS_1], and her postal code is [ZIP_1]. Her email is [EMAIL_1].",
        },
        {
            "role": "assistant",
            "prompt": "Jane Smith's details have been anonymized.",
            "response": "[NAME_1]'s birthday is [DOB_1]. She lives at [ADDRESS_1], and her postal code is [ZIP_1]. Her email is [EMAIL_1].",
        },
        {
            "role": "user",
            "prompt": "Marcus Brown's email is marcus.brown@example.com, and his BSN is 12345678. His birthday is December 1st, 2024, and his home address is 456 Pine St.",
            "response": "[NAME_1]'s email is [EMAIL_1], and his BSN is [BSN_1]. His birthday is [DOB_1], and his home address is [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Marcus Brown's personal details have been anonymized.",
            "response": "[NAME_1]'s email is [EMAIL_1], and his BSN is [BSN_1]. His birthday is [DOB_1], and his home address is [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Emily Carter was born on 01/12/2024. Her email is emily.carter@example.com, and her BSN is 98765432. She lives at 789 Spruce Lane.",
            "response": "[NAME_1] was born on [DOB_1]. Her email is [EMAIL_1], and her BSN is [BSN_1]. She lives at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Emily Carter's personal details have been anonymized.",
            "response": "[NAME_1] was born on [DOB_1]. Her email is [EMAIL_1], and her BSN is [BSN_1]. She lives at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "John Doe's email is J.Simpson@@netwrix..com. His BSN is 12345678.9. His home address is 10 Langelo! His zip code is 7666mc. His Mastercard number is 5258-7041-08753590 and his visa number is 4563 7568 5698 4587. His iban number is nl91abna0417164300. His date of birth is 1/1/90. His IP address is 192.168.1.1.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His home address is [ADDRESS_1]. His ZIP code is [ZIP_1]. His MasterCard number is [MASTERCARD_1] and his Visa number is [VISA_1]. His IBAN number is [IBAN_1]. His date of birth is [DOB_1]. His IP address is [IP_ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "John Doe's personal details have been anonymized.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His home address is [ADDRESS_1]. His ZIP code is [ZIP_1]. His MasterCard number is [MASTERCARD_1] and his Visa number is [VISA_1]. His IBAN number is [IBAN_1]. His date of birth is [DOB_1]. His IP address is [IP_ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Olivia Doe's email is d.olivia@netwrix..com. Her BSN is 98765432. Her home address is 60 Langelo Langelo. Her ZIP code is 7612-MC. Her Mastercard number is 5258-7041-0873577 and her visa number is 4511-7500-0000-0000. Her iban number is DE91abna0417164300. Her dob is 1/1/91. Her ip is 192.168.2.2.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. Her home address is [ADDRESS_1]. Her ZIP code is [ZIP_1]. Her MasterCard number is [MASTERCARD_1] and her Visa number is [VISA_1]. Her IBAN number is [IBAN_1]. Her date of birth is [DOB_1]. Her IP address is [IP_ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Olivia Doe's personal details have been anonymized.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. Her home address is [ADDRESS_1]. Her ZIP code is [ZIP_1]. Her MasterCard number is [MASTERCARD_1] and her Visa number is [VISA_1]. Her IBAN number is [IBAN_1]. Her date of birth is [DOB_1]. Her IP address is [IP_ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "John and Olivia went to the market. John's email is john.doe@example.com, and Olivia's BSN is 98765432. Their addresses are 123 Main St and 456 Oak St. The zip codes are 98765 and 12345. John's Mastercard number is 1234-5678-9876-5432.",
            "response": "[NAME_1] and [NAME_2] went to the market. [NAME_1]'s email is [EMAIL_1], and [NAME_2]'s BSN is [BSN_1]. Their addresses are [ADDRESS_1] and [ADDRESS_2]. The zip codes are [ZIP_1] and [ZIP_2]. [NAME_1]'s MasterCard number is [MASTERCARD_1].",
        },
        {
            "role": "assistant",
            "prompt": "John and Olivia's personal details have been anonymized.",
            "response": "[NAME_1] and [NAME_2] went to the market. [NAME_1]'s email is [EMAIL_1], and [NAME_2]'s BSN is [BSN_1]. Their addresses are [ADDRESS_1] and [ADDRESS_2]. The zip codes are [ZIP_1] and [ZIP_2]. [NAME_1]'s MasterCard number is [MASTERCARD_1].",
        },
        {
            "role": "user",
            "prompt": "Contact me at invalid@domain..com, my BSN is 123-45-678, and my IP is 192.168.0.2.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His IP address is [IP_ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Personal details with invalid email have been anonymized.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His IP address is [IP_ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "My friend Emily has an email emily@example.com and her BSN is 23456789. She lives at 789 Pine St. Her zip code is 67890.",
            "response": "[NAME_1] has an email [EMAIL_1] and her BSN is [BSN_1]. She lives at [ADDRESS_1]. Her ZIP code is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "Emily's personal details have been anonymized.",
            "response": "[NAME_1] has an email [EMAIL_1] and her BSN is [BSN_1]. She lives at [ADDRESS_1]. Her ZIP code is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "The company contacted me through my email address: john@example.com. I have a bank account number: 1234567890123456.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His bank account number is [BANK_ACCOUNT_1].",
        },
        {
            "role": "assistant",
            "prompt": "Personal details related to bank account have been anonymized.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His bank account number is [BANK_ACCOUNT_1].",
        },
        # Dutch Cases
        {
            "role": "user",
            "prompt": "Jan Jansen's e-mailadres is jan.jansen@@voorbeeld.nl. Zijn BSN is 12345678.9. Zijn huisadres is 10 Hoofdstraat! Zijn postcode is 7666mc. Zijn Mastercardnummer is 5258-7041-08753590 en zijn Visa-nummer is 4563 7568 5698 4587. Zijn IBAN-nummer is NL91ABNA0417164300. Zijn geboortedatum is 1/1/90. Zijn IP-adres is 192.168.1.1.",
            "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn huisadres is [ADDRESS_1]. Zijn postcode is [ZIP_1]. Zijn Mastercardnummer is [MASTERCARD_1] en zijn Visa-nummer is [VISA_1]. Zijn IBAN-nummer is [IBAN_1]. Zijn geboortedatum is [DOB_1]. Zijn IP-adres is [IP_ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Jan Jansen's personal details have been anonymized.",
            "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn huisadres is [ADDRESS_1]. Zijn postcode is [ZIP_1]. Zijn Mastercardnummer is [MASTERCARD_1] en zijn Visa-nummer is [VISA_1]. Zijn IBAN-nummer is [IBAN_1]. Zijn geboortedatum is [DOB_1]. Zijn IP-adres is [IP_ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Olivia Jansen's e-mailadres is d.olivia@voorbeeld..nl. Haar BSN is 98765432. Haar huisadres is 60 Langelaan. Haar postcode is 7612-MC. Haar Mastercardnummer is 5258-7041-0873577 en haar Visa-nummer is 4511-7500-0000-0000. Haar IBAN-nummer is DE91ABNA0417164300. Haar geboortedatum is 1/1/91. Haar IP is 192.168.2.2.",
            "response": "[NAME_2]'s e-mailadres is [EMAIL_2]. Haar BSN is [BSN_2]. Haar huisadres is [ADDRESS_2]. Haar postcode is [ZIP_2]. Haar Mastercardnummer is [MASTERCARD_2] en haar Visa-nummer is [VISA_2]. Haar IBAN-nummer is [IBAN_2]. Haar geboortedatum is [DOB_2]. Haar IP-adres is [IP_ADDRESS_2].",
        },
        {
            "role": "assistant",
            "prompt": "Olivia Jansen's personal details have been anonymized.",
            "response": "[NAME_2]'s e-mailadres is [EMAIL_2]. Haar BSN is [BSN_2]. Haar huisadres is [ADDRESS_2]. Haar postcode is [ZIP_2]. Haar Mastercardnummer is [MASTERCARD_2] en haar Visa-nummer is [VISA_2]. Haar IBAN-nummer is [IBAN_2]. Haar geboortedatum is [DOB_2]. Haar IP-adres is [IP_ADDRESS_2].",
        },
        {
            "role": "user",
            "prompt": "Jan en Olivia waren in het park. Jan's e-mailadres is jan.doe@voorbeeld.com, en Olivia's BSN is 98765432. Hun adressen zijn 123 Hoofdstraat en 456 Eiklaan. De postcodes zijn 98765 en 12345. Jan's Mastercardnummer is 1234-5678-9876-5432.",
            "response": "[NAME_1] en [NAME_2] waren in het park. [NAME_1]'s e-mailadres is [EMAIL_1], en [NAME_2]'s BSN is [BSN_1]. Hun adressen zijn [ADDRESS_1] en [ADDRESS_2]. De postcodes zijn [ZIP_1] en [ZIP_2]. [NAME_1]'s Mastercardnummer is [MASTERCARD_1].",
        },
        {
            "role": "assistant",
            "prompt": "Jan en Olivia's personal details have been anonymized.",
            "response": "[NAME_1] en [NAME_2] waren in het park. [NAME_1]'s e-mailadres is [EMAIL_1], en [NAME_2]'s BSN is [BSN_1]. Hun adressen zijn [ADDRESS_1] en [ADDRESS_2]. De postcodes zijn [ZIP_1] en [ZIP_2]. [NAME_1]'s Mastercardnummer is [MASTERCARD_1].",
        },
        {
            "role": "user",
            "prompt": "Neem contact met me op via jan.doe@gmail..com, mijn BSN is 123-45-678, en mijn IP is 192.168.0.2.",
            "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn IP-adres is [IP_ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Personal details with invalid email have been anonymized.",
            "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn IP-adres is [IP_ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Sophie werkt bij een bedrijf en haar e-mailadres is sophie@voorbeeld.com. Haar BSN is 23456789. Ze woont op 321 Laan van de Vrijheid. Haar postcode is 12345.",
            "response": "[NAME_1] werkt bij een bedrijf en haar e-mailadres is [EMAIL_1]. Haar BSN is [BSN_1]. Ze woont op [ADDRESS_1]. Haar postcode is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "Sophie werkt bij een bedrijf en haar e-mailadres is sophie@voorbeeld.com. Haar BSN is 23456789. Ze woont op 321 Laan van de Vrijheid. Haar postcode is 12345.",
            "response": "[NAME_1] werkt bij een bedrijf en haar e-mailadres is [EMAIL_1]. Haar BSN is [BSN_1]. Ze woont op [ADDRESS_1]. Haar postcode is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "De geboortedatum van Jan Jansen is 01/12/2024. Zijn leeftijd is 25 jaar.",
            "response": "De geboortedatum van [NAME_1] is [DOB_1]. Zijn leeftijd is [AGE_1] jaar.",
        },
        {
            "role": "assistant",
            "prompt": "De geboortedatum van Jan Jansen is 01/12/2024. Zijn leeftijd is 25 jaar.",
            "response": "De geboortedatum van [NAME_1] is [DOB_1]. Zijn leeftijd is [AGE_1] jaar.",
        },
        {
            "role": "user",
            "prompt": "Olivia Jansen werd geboren op 12/01/2024 en is nu 30 jaar oud.",
            "response": "[NAME_1] werd geboren op [DOB_1] en is nu [AGE_1] jaar oud.",
        },
        {
            "role": "assistant",
            "prompt": "Olivia Jansen werd geboren op 12/01/2024 en is nu 30 jaar oud.",
            "response": "[NAME_1] werd geboren op [DOB_1] en is nu [AGE_1] jaar oud.",
        },
        {
            "role": "user",
            "prompt": "Sophie is geboren op 1 december 2024 en ze is 45 jaar.",
            "response": "[NAME_1] is geboren op [DOB_1] en ze is [AGE_1] jaar.",
        },
        {
            "role": "assistant",
            "prompt": "Sophie is geboren op 1 december 2024 en ze is 45 jaar.",
            "response": "[NAME_1] is geboren op [DOB_1] en ze is [AGE_1] jaar.",
        },
        {
            "role": "user",
            "prompt": "Karel werd geboren op de eerste december 2024. Hij is nu 40 jaar.",
            "response": "[NAME_1] werd geboren op [DOB_1]. Hij is nu [AGE_1] jaar.",
        },
        {
            "role": "assistant",
            "prompt": "Karel werd geboren op de eerste december 2024. Hij is nu 40 jaar.",
            "response": "[NAME_1] werd geboren op [DOB_1]. Hij is nu [AGE_1] jaar.",
        },
        {
            "role": "user",
            "prompt": "De geboortedatum van Maria is 2024-12-01. Ze is nu 50 jaar oud.",
            "response": "De geboortedatum van [NAME_1] is [DOB_1]. Ze is nu [AGE_1] jaar oud.",
        },
        {
            "role": "assistant",
            "prompt": "De geboortedatum van Maria is 2024-12-01. Ze is nu 50 jaar oud.",
            "response": "De geboortedatum van [NAME_1] is [DOB_1]. Ze is nu [AGE_1] jaar oud.",
        },
        {
            "role": "user",
            "prompt": "Haar geboortedatum is 12 januari 2024, en ze is 33 jaar.",
            "response": "Haar geboortedatum is [DOB_1], en ze is [AGE_1] jaar.",
        },
        {
            "role": "assistant",
            "prompt": "Haar geboortedatum is 12 januari 2024, en ze is 33 jaar.",
            "response": "Haar geboortedatum is [DOB_1], en ze is [AGE_1] jaar.",
        },
        {
            "role": "user",
            "prompt": "De geboortedatum van mijn vriend is 01-12-2024. Hij is nu 35 jaar.",
            "response": "De geboortedatum van [NAME_1] is [DOB_1]. Hij is nu [AGE_1] jaar.",
        },
        {
            "role": "assistant",
            "prompt": "De geboortedatum van mijn vriend is 01-12-2024. Hij is nu 35 jaar.",
            "response": "De geboortedatum van [NAME_1] is [DOB_1]. Hij is nu [AGE_1] jaar.",
        },
        {
            "role": "user",
            "prompt": "Mijn broer is geboren op 1-12-2024 en hij is nu 27 jaar.",
            "response": "Mijn broer is geboren op [DOB_1] en hij is nu [AGE_1] jaar.",
        },
        {
            "role": "assistant",
            "prompt": "Mijn broer is geboren op 1-12-2024 en hij is nu 27 jaar.",
            "response": "Mijn broer is geboren op [DOB_1] en hij is nu [AGE_1] jaar.",
        },
        {
            "role": "user",
            "prompt": "Ze is op 1e december 2024 geboren, en ze is 29 jaar oud.",
            "response": "Ze is op [DOB_1] geboren, en ze is [AGE_1] jaar oud.",
        },
        {
            "role": "assistant",
            "prompt": "Ze is op 1e december 2024 geboren, en ze is 29 jaar oud.",
            "response": "Ze is op [DOB_1] geboren, en ze is [AGE_1] jaar oud.",
        },
        {
            "role": "user",
            "prompt": "Zijn geboortedatum is 12 januari 2024. Hij is nu 37 jaar.",
            "response": "Zijn geboortedatum is [DOB_1]. Hij is nu [AGE_1] jaar.",
        },
        {
            "role": "assistant",
            "prompt": "Zijn geboortedatum is 12 januari 2024. Hij is nu 37 jaar.",
            "response": "Zijn geboortedatum is [DOB_1]. Hij is nu [AGE_1] jaar.",
        },
        {
            "role": "user",
            "prompt": "Ik heb een nieuw emailadres: mijn.email@voorbeeld.nl en mijn BSN is 34567890. Mijn huisadres is 88 Nieuweweg.",
            "response": "[NAME_1] heeft een nieuw e-mailadres: [EMAIL_1] en haar BSN is [BSN_1]. Haar huisadres is [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Ik heb een nieuw emailadres: mijn.email@voorbeeld.nl en mijn BSN is 34567890. Mijn huisadres is 88 Nieuweweg.",
            "response": "[NAME_1] heeft een nieuw e-mailadres: [EMAIL_1] en haar BSN is [BSN_1]. Haar huisadres is [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Sara de Vries's e-mailadres is sara.de.vries@voorbeeld.nl. Haar BSN is 12345678. Haar huisadres is 12 Fijnstraat, en haar postcode is 98765.",
            "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Haar BSN is [BSN_1]. Haar huisadres is [ADDRESS_1], en haar postcode is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "Sara de Vries's e-mailadres is sara.de.vries@voorbeeld.nl. Haar BSN is 12345678. Haar huisadres is 12 Fijnstraat, en haar postcode is 98765.",
            "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Haar BSN is [BSN_1]. Haar huisadres is [ADDRESS_1], en haar postcode is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "De geboortedatum van Piet is 01-12-2024. Zijn e-mailadres is piet@example.nl, en zijn BSN is 87654321. Hij woont op 456 Laan van de Vrijheid.",
            "response": "De geboortedatum van [NAME_1] is [DOB_1]. Zijn e-mailadres is [EMAIL_1], en zijn BSN is [BSN_1]. Hij woont op [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "De geboortedatum van Piet is 01-12-2024. Zijn e-mailadres is piet@example.nl, en zijn BSN is 87654321. Hij woont op 456 Laan van de Vrijheid.",
            "response": "De geboortedatum van [NAME_1] is [DOB_1]. Zijn e-mailadres is [EMAIL_1], en zijn BSN is [BSN_1]. Hij woont op [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Karin's e-mailadres is karin@example.nl, en haar BSN is 22223333. Haar geboortedatum is 12/01/2024, en ze woont op 111 Eikenlaan.",
            "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Haar BSN is [BSN_1]. Haar geboortedatum is [DOB_1], en ze woont op [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Karin's e-mailadres is karin@example.nl, en haar BSN is 22223333. Haar geboortedatum is 12/01/2024, en ze woont op 111 Eikenlaan.",
            "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Haar BSN is [BSN_1]. Haar geboortedatum is [DOB_1], en ze woont op [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Rob Jansen's e-mailadres is rob.jansen@voorbeeld.nl. Zijn BSN is 11112222, en zijn huisadres is 88 Hoofdstraat. Zijn geboortedatum is 01-12-2024.",
            "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn huisadres is [ADDRESS_1]. Zijn geboortedatum is [DOB_1].",
        },
        {
            "role": "assistant",
            "prompt": "Rob Jansen's e-mailadres is rob.jansen@voorbeeld.nl. Zijn BSN is 11112222, en zijn huisadres is 88 Hoofdstraat. Zijn geboortedatum is 01-12-2024.",
            "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn huisadres is [ADDRESS_1]. Zijn geboortedatum is [DOB_1].",
        },
        {
            "role": "user",
            "prompt": "Annelies heeft een nieuw emailadres: annelies@voorbeeld.nl. Haar BSN is 99998888, en ze woont op 24 Dennenlaan. Haar postcode is 12345.",
            "response": "[NAME_1] heeft een nieuw e-mailadres: [EMAIL_1]. Haar BSN is [BSN_1]. Ze woont op [ADDRESS_1]. Haar postcode is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "Annelies heeft een nieuw emailadres: annelies@voorbeeld.nl. Haar BSN is 99998888, en ze woont op 24 Dennenlaan. Haar postcode is 12345.",
            "response": "[NAME_1] heeft een nieuw e-mailadres: [EMAIL_1]. Haar BSN is [BSN_1]. Ze woont op [ADDRESS_1]. Haar postcode is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "Janine werd geboren op 12/01/2024 en ze is nu 26 jaar oud. Haar email is janine@example.nl.",
            "response": "[NAME_1] werd geboren op [DOB_1] en ze is nu [AGE_1] jaar oud. Haar email is [EMAIL_1].",
        },
        {
            "role": "assistant",
            "prompt": "Janine werd geboren op 12/01/2024 en ze is nu 26 jaar oud. Haar email is janine@example.nl.",
            "response": "[NAME_1] werd geboren op [DOB_1] en ze is nu [AGE_1] jaar oud. Haar email is [EMAIL_1].",
        },
        {
            "role": "user",
            "prompt": "Sophie werkt bij een bedrijf. Haar e-mailadres is sophie@voorbeeld.nl, en haar BSN is 34567890. Ze woont op 555 Klaproosstraat.",
            "response": "[NAME_1] werkt bij een bedrijf. Haar e-mailadres is [EMAIL_1]. Haar BSN is [BSN_1]. Ze woont op [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Sophie werkt bij een bedrijf. Haar e-mailadres is sophie@voorbeeld.nl, en haar BSN is 34567890. Ze woont op 555 Klaproosstraat.",
            "response": "[NAME_1] werkt bij een bedrijf. Haar e-mailadres is [EMAIL_1]. Haar BSN is [BSN_1]. Ze woont op [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Frits heeft een nieuwe email: frits@example.nl. Zijn BSN is 77778888. Hij woont op 80 Rozenlaan.",
            "response": "[NAME_1] heeft een nieuwe email: [EMAIL_1]. Zijn BSN is [BSN_1]. Hij woont op [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Frits heeft een nieuwe email: frits@example.nl. Zijn BSN is 77778888. Hij woont op 80 Rozenlaan.",
            "response": "[NAME_1] heeft een nieuwe email: [EMAIL_1]. Zijn BSN is [BSN_1]. Hij woont op [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "De geboortedatum van Lotte is 01-01-2024. Ze woont op 34 Bovenstraat. Haar e-mailadres is lotte@voorbeeld.nl.",
            "response": "De geboortedatum van [NAME_1] is [DOB_1]. Ze woont op [ADDRESS_1]. Haar e-mailadres is [EMAIL_1].",
        },
        {
            "role": "assistant",
            "prompt": "De geboortedatum van Lotte is 01-01-2024. Ze woont op 34 Bovenstraat. Haar e-mailadres is lotte@voorbeeld.nl.",
            "response": "De geboortedatum van [NAME_1] is [DOB_1]. Ze woont op [ADDRESS_1]. Haar e-mailadres is [EMAIL_1].",
        },
        {
            "role": "user",
            "prompt": "Milan heeft een emailadres: milan@example.nl. Zijn BSN is 55556666. Hij woont op 12 Appelstraat.",
            "response": "[NAME_1] heeft een emailadres: [EMAIL_1]. Zijn BSN is [BSN_1]. Hij woont op [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Milan heeft een emailadres: milan@example.nl. Zijn BSN is 55556666. Hij woont op 12 Appelstraat.",
            "response": "[NAME_1] heeft een emailadres: [EMAIL_1]. Zijn BSN is [BSN_1]. Hij woont op [ADDRESS_1].",
        },
        # English Cases with Typos and Incorrect Formats
        {
            "role": "user",
            "prompt": "Allice Smith's email is allice.smith@example..com. Her BSN is 12345678. Her home adress is 45 Elm St, and her postal code is 9021. Her MasterCard number is 5412-3456-7890-12345 and her IBAN number is GB29NWBK60161331926819.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. Her home address is [ADDRESS_1]. Her postal code is [ZIP_1]. Her MasterCard number is [MASTERCARD_1] and her IBAN number is [IBAN_1].",
        },
        {
            "role": "assistant",
            "prompt": "Allice Smith's email is allice.smith@example..com. Her BSN is 12345678. Her home adress is 45 Elm St, and her postal code is 9021. Her MasterCard number is 5412-3456-7890-12345 and her IBAN number is GB29NWBK60161331926819.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. Her home address is [ADDRESS_1]. Her postal code is [ZIP_1]. Her MasterCard number is [MASTERCARD_1] and her IBAN number is [IBAN_1].",
        },
        {
            "role": "user",
            "prompt": "Mark Jonson has an email mark.j@exampl.com. His BSN is 98765432. His adress is 72 Oak Ave. His zip code is 54321.",
            "response": "[NAME_1] has an email [EMAIL_1]. His BSN is [BSN_1]. His address is [ADDRESS_1]. His ZIP code is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "Mark Jonson has an email mark.j@exampl.com. His BSN is 98765432. His adress is 72 Oak Ave. His zip code is 54321.",
            "response": "[NAME_1] has an email [EMAIL_1]. His BSN is [BSN_1]. His address is [ADDRESS_1]. His ZIP code is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "Email: test.email@invalid..com. BSN: 321-45-6789. IP address: 172.16.254.01.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His IP address is [IP_ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Email: test.email@invalid..com. BSN: 321-45-6789. IP address: 172.16.254.01.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His IP address is [IP_ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Contact me at my email address: not.a.valid.email@domain..com. My BSN is 123456789, and my IP is 192.168.0.256.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His IP address is [IP_ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Contact me at my email address: not.a.valid.email@domain..com. My BSN is 123456789, and my IP is 192.168.0.256.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His IP address is [IP_ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "There is a problem with my email, which is john.doe@exampl.com, and my BSN is 234-56-7890.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1].",
        },
        {
            "role": "assistant",
            "prompt": "There is a problem with my email, which is john.doe@exampl.com, and my BSN is 234-56-7890.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1].",
        },
        {
            "role": "user",
            "prompt": "Johhn's birthdate is Dec 32, 2024. He is now 26 yrs old.",
            "response": "[NAME_1]'s birthdate is [DOB_1]. He is now [AGE_1] years old.",
        },
        {
            "role": "assistant",
            "prompt": "Johhn's birthdate is Dec 32, 2024. He is now 26 yrs old.",
            "response": "[NAME_1]'s birthdate is [DOB_1]. He is now [AGE_1] years old.",
        },
        {
            "role": "user",
            "prompt": "Smantha was born on 15/13/2024, she's 34 year.",
            "response": "[NAME_1] was born on [DOB_1], she's [AGE_1] years old.",
        },
        {
            "role": "assistant",
            "prompt": "Smantha was born on 15/13/2024, she's 34 year.",
            "response": "[NAME_1] was born on [DOB_1], she's [AGE_1] years old.",
        },
        {
            "role": "user",
            "prompt": "Jke's date of brith is 2024-31-11 and he is 28 yo.",
            "response": "[NAME_1]'s date of birth is [DOB_1] and he is [AGE_1] years old.",
        },
        {
            "role": "assistant",
            "prompt": "Jke's date of brith is 2024-31-11 and he is 28 yo.",
            "response": "[NAME_1]'s date of birth is [DOB_1] and he is [AGE_1] years old.",
        },
        {
            "role": "user",
            "prompt": "Mia's dob is 01//12/2024. She is 24.",
            "response": "[NAME_1]'s date of birth is [DOB_1]. She is [AGE_1] years old.",
        },
        {
            "role": "assistant",
            "prompt": "Mia's dob is 01//12/2024. She is 24.",
            "response": "[NAME_1]'s date of birth is [DOB_1]. She is [AGE_1] years old.",
        },
        {
            "role": "user",
            "prompt": "His birthdate is 1/13/2024, he is now 35 year.",
            "response": "His birthdate is [DOB_1], he is now [AGE_1] years old.",
        },
        {
            "role": "assistant",
            "prompt": "His birthdate is 1/13/2024, he is now 35 year.",
            "response": "His birthdate is [DOB_1], he is now [AGE_1] years old.",
        },
        {
            "role": "user",
            "prompt": "Born on Febuary 29, 2024, Jake is 33 years old.",
            "response": "Born on [DOB_1], [NAME_1] is [AGE_1] years old.",
        },
        {
            "role": "assistant",
            "prompt": "Born on Febuary 29, 2024, Jake is 33 years old.",
            "response": "Born on [DOB_1], [NAME_1] is [AGE_1] years old.",
        },
        {
            "role": "user",
            "prompt": "Ben's birtdate: 2024/12/01. He is twenty five.",
            "response": "[NAME_1]'s birthdate is [DOB_1]. He is [AGE_1] years old.",
        },
        {
            "role": "assistant",
            "prompt": "Ben's birtdate: 2024/12/01. He is twenty five.",
            "response": "[NAME_1]'s birthdate is [DOB_1]. He is [AGE_1] years old.",
        },
        {
            "role": "user",
            "prompt": "Anna, dob 12th Oct, 2024, age 32.",
            "response": "[NAME_1], dob [DOB_1], age [AGE_1].",
        },
        {
            "role": "assistant",
            "prompt": "Anna, dob 12th Oct, 2024, age 32.",
            "response": "[NAME_1], dob [DOB_1], age [AGE_1].",
        },
        {
            "role": "user",
            "prompt": "Her DOB is 01-33-2024, and she's 22 yr.",
            "response": "Her DOB is [DOB_1], and she's [AGE_1] years old.",
        },
        {
            "role": "assistant",
            "prompt": "Her DOB is 01-33-2024, and she's 22 yr.",
            "response": "Her DOB is [DOB_1], and she's [AGE_1] years old.",
        },
        {
            "role": "user",
            "prompt": "His date of brth is 12/01/2024. Now he's 45 yo.",
            "response": "His date of birth is [DOB_1]. Now he's [AGE_1] years old.",
        },
        {
            "role": "assistant",
            "prompt": "His date of brth is 12/01/2024. Now he's 45 yo.",
            "response": "His date of birth is [DOB_1]. Now he's [AGE_1] years old.",
        },
        # Dutch Cases with Typos and Incorrect Formats
        {
            "role": "user",
            "prompt": "Liesbeth heeft een e-mailadres: liesbeth@@voorbeeld.nl. Haar BSN is 2345678. Ze woont op 101 Beukenlaan, postcode 1234ABC.",
            "response": "[NAME_1] heeft een e-mailadres: [EMAIL_1]. Haar BSN is [BSN_1]. Ze woont op [ADDRESS_1], postcode [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "Liesbeth heeft een e-mailadres: liesbeth@@voorbeeld.nl. Haar BSN is 2345678. Ze woont op 101 Beukenlaan, postcode 1234ABC.",
            "response": "[NAME_1] heeft een e-mailadres: [EMAIL_1]. Haar BSN is [BSN_1]. Ze woont op [ADDRESS_1], postcode [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "Pieter's email is pieter@exampl.nl. Zijn BSN is 34567890. Zijn huisadrees is 65 Lindelaan. Zijn postcode is 5678CD.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn huisadres is [ADDRESS_1]. Zijn postcode is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "Pieter's email is pieter@exampl.nl. Zijn BSN is 34567890. Zijn huisadrees is 65 Lindelaan. Zijn postcode is 5678CD.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn huisadres is [ADDRESS_1]. Zijn postcode is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "Neem contact met me op via mijn e-mailadres: info@bedrijf..nl. Mijn BSN is 456-78-901, en mijn IP is 192.168.0.300.",
            "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn IP-adres is [IP_ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Neem contact met me op via mijn e-mailadres: info@bedrijf..nl. Mijn BSN is 456-78-901, en mijn IP is 192.168.0.300.",
            "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn IP-adres is [IP_ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Kunt u me e-mailen op mijn.new.email@voorbeeld..nl? Mijn BSN is 12345678.",
            "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1].",
        },
        {
            "role": "assistant",
            "prompt": "Kunt u me e-mailen op mijn.new.email@voorbeeld..nl? Mijn BSN is 12345678.",
            "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1].",
        },
        {
            "role": "user",
            "prompt": "Ik heb een nieuw emailadres: mijn.email@voorbeeld.nl. Mijn BSN is 34567890. Mijn huisadres is 88 Nieuweweg. En mijn postcod is 1010AB.",
            "response": "[NAME_1] heeft een nieuw e-mailadres: [EMAIL_1]. Haar BSN is [BSN_1]. Haar huisadres is [ADDRESS_1]. Zijn postcode is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "Ik heb een nieuw emailadres: mijn.email@voorbeeld.nl. Mijn BSN is 34567890. Mijn huisadres is 88 Nieuweweg. En mijn postcod is 1010AB.",
            "response": "[NAME_1] heeft een nieuw e-mailadres: [EMAIL_1]. Haar BSN is [BSN_1]. Haar huisadres is [ADDRESS_1]. Zijn postcode is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "De geboortedatum van Jan Jnansen is 01-32-2024. Zijn leeftijd is 25 jaar.",
            "response": "De geboortedatum van [NAME_1] is [DOB_1]. Zijn leeftijd is [AGE_1] jaar.",
        },
        {
            "role": "assistant",
            "prompt": "De geboortedatum van Jan Jnansen is 01-32-2024. Zijn leeftijd is 25 jaar.",
            "response": "De geboortedatum van [NAME_1] is [DOB_1]. Zijn leeftijd is [AGE_1] jaar.",
        },
        {
            "role": "user",
            "prompt": "Olivvia werd gebroen op 13/01/2024 en is nu 30 jaar oud.",
            "response": "[NAME_1] werd geboren op [DOB_1] en is nu [AGE_1] jaar oud.",
        },
        {
            "role": "assistant",
            "prompt": "Olivvia werd gebroen op 13/01/2024 en is nu 30 jaar oud.",
            "response": "[NAME_1] werd geboren op [DOB_1] en is nu [AGE_1] jaar oud.",
        },
        {
            "role": "user",
            "prompt": "Sophe is gebon op 1-12-2024 en ze is 45 jar.",
            "response": "[NAME_1] is geboren op [DOB_1] en ze is [AGE_1] jaar.",
        },
        {
            "role": "assistant",
            "prompt": "Sophe is gebon op 1-12-2024 en ze is 45 jar.",
            "response": "[NAME_1] is geboren op [DOB_1] en ze is [AGE_1] jaar.",
        },
        {
            "role": "user",
            "prompt": "De geboortdatum van Maria is 2024-13-01. Ze is nu 50 jr oud.",
            "response": "De geboortedatum van [NAME_1] is [DOB_1]. Ze is nu [AGE_1] jaar oud.",
        },
        {
            "role": "assistant",
            "prompt": "De geboortdatum van Maria is 2024-13-01. Ze is nu 50 jr oud.",
            "response": "De geboortedatum van [NAME_1] is [DOB_1]. Ze is nu [AGE_1] jaar oud.",
        },
        {
            "role": "user",
            "prompt": "Haar gebortedatum is 12 januari 2024, en ze is 33 j.",
            "response": "Haar geboortedatum is [DOB_1], en ze is [AGE_1] jaar.",
        },
        {
            "role": "assistant",
            "prompt": "Haar gebortedatum is 12 januari 2024, en ze is 33 j.",
            "response": "Haar geboortedatum is [DOB_1], en ze is [AGE_1] jaar.",
        },
        {
            "role": "user",
            "prompt": "Mijn broer is geboren op 1-13-2024 en hij is nu 27 jaar.",
            "response": "Mijn broer is geboren op [DOB_1] en hij is nu [AGE_1] jaar.",
        },
        {
            "role": "assistant",
            "prompt": "Mijn broer is geboren op 1-13-2024 en hij is nu 27 jaar.",
            "response": "Mijn broer is geboren op [DOB_1] en hij is nu [AGE_1] jaar.",
        },
        {
            "role": "user",
            "prompt": "Ze is op 1e dec 2024 geborne, en ze is 29 jaar oud.",
            "response": "Ze is op [DOB_1] geboren, en ze is [AGE_1] jaar oud.",
        },
        {
            "role": "assistant",
            "prompt": "Ze is op 1e dec 2024 geborne, en ze is 29 jaar oud.",
            "response": "Ze is op [DOB_1] geboren, en ze is [AGE_1] jaar oud.",
        },
        {
            "role": "user",
            "prompt": "Zijn geboortedatumm is 13 januari 2024. Hij is nu 37 jr.",
            "response": "Zijn geboortedatum is [DOB_1]. Hij is nu [AGE_1] jaar.",
        },
        {
            "role": "assistant",
            "prompt": "Zijn geboortedatumm is 13 januari 2024. Hij is nu 37 jr.",
            "response": "Zijn geboortedatum is [DOB_1]. Hij is nu [AGE_1] jaar.",
        },
        {
            "role": "user",
            "prompt": "i am tom. tom's email is tom@example.com, his BSN is 12345678. he lives at 100 Cedar Ave, and his zip code is 45678.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. He lives at [ADDRESS_1], and his zip code is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "i am tom. tom's email is tom@example.com, his BSN is 12345678. he lives at 100 Cedar Ave, and his zip code is 45678.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. He lives at [ADDRESS_1], and his zip code is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "My name is SARA JOHNSON. My email is sara.johnson@exammple.com, and my BSN is 87654321. I live at 222 Ash St.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "My name is SARA JOHNSON. My email is sara.johnson@exammple.com, and my BSN is 87654321. I live at 222 Ash St.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "allice.smith@example..com is my email. My BSN is 12345678. I reside at 15 elm street.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She resides at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "allice.smith@example..com is my email. My BSN is 12345678. I reside at 15 elm street.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She resides at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "robert's date of birth is 12-31-2024. his email is robert.brown@example.com, and his BSN is 34567890.",
            "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1], and his BSN is [BSN_1].",
        },
        {
            "role": "assistant",
            "prompt": "robert's date of birth is 12-31-2024. his email is robert.brown@example.com, and his BSN is 34567890.",
            "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1], and his BSN is [BSN_1].",
        },
        {
            "role": "user",
            "prompt": "Linda johnson's email is linda.johnson@exammple..com, and her bsn is 11223344. she lives at 321 willow way.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Linda johnson's email is linda.johnson@exammple..com, and her bsn is 11223344. she lives at 321 willow way.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "my friend max's date of birth is 2024-01-01. his email is max@example..com. his postal code is 12345.",
            "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1]. His postal code is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "my friend max's date of birth is 2024-01-01. his email is max@example..com. his postal code is 12345.",
            "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1]. His postal code is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "JULIA's email is julia123@example.com, her bsn is 22334455. She was born on 01-12-2024.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She was born on [DOB_1].",
        },
        {
            "role": "assistant",
            "prompt": "JULIA's email is julia123@example.com, her bsn is 22334455. She was born on 01-12-2024.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She was born on [DOB_1].",
        },
        {
            "role": "user",
            "prompt": "Tom's email is TOM123@EXAMPLE.COM, and his bsn is 99988877. he was born on 12/01/2024.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. He was born on [DOB_1].",
        },
        {
            "role": "assistant",
            "prompt": "Tom's email is TOM123@EXAMPLE.COM, and his bsn is 99988877. he was born on 12/01/2024.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. He was born on [DOB_1].",
        },
        {
            "role": "user",
            "prompt": "my cousin Anna's email is anna123@EXAMPLE.COM. Her BSN is 77788899, and she lives at 77 Elm St.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "my cousin Anna's email is anna123@EXAMPLE.COM. Her BSN is 77788899, and she lives at 77 Elm St.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Max's date of birth is 01/1/2024. his email is max@example.com. His postal code is 12345.",
            "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1]. His postal code is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "Max's date of birth is 01/1/2024. his email is max@example.com. His postal code is 12345.",
            "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1]. His postal code is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "Contact me at invalid@domain..com, my bsn is 123-45-678, and my ip is 192.168.0.2.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His IP address is [IP_ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Contact me at invalid@domain..com, my bsn is 123-45-678, and my ip is 192.168.0.2.",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His IP address is [IP_ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Emily's email is emily@example..com, and her BSN is 23456789. she lives at 789 pine st.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Emily's email is emily@example..com, and her BSN is 23456789. she lives at 789 pine st.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "JOHN DOE's email is john.doe@example.com. His bsn is 12345678. he lives at 10 langelo!",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. He lives at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "JOHN DOE's email is john.doe@example.com. His bsn is 12345678. he lives at 10 langelo!",
            "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. He lives at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "Contact olivia at d.olivia@netwrix..com. Her bsn is 98765432. Her address is 60 langelo langelo.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. Her address is [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "Contact olivia at d.olivia@netwrix..com. Her bsn is 98765432. Her address is 60 langelo langelo.",
            "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. Her address is [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "i am tom.",
            "response": "I am [NAME_1].",
        },
        {
            "role": "assistant",
            "prompt": "i am tom.",
            "response": "I am [NAME_1].",
        },
        {
            "role": "user",
            "prompt": "my name is alice.",
            "response": "My name is [NAME_1].",
        },
        {
            "role": "assistant",
            "prompt": "my name is alice.",
            "response": "My name is [NAME_1].",
        },
        {
            "role": "user",
            "prompt": "robert's email is robert.brown@example.com.",
            "response": "[NAME_1]'s email is [EMAIL_1].",
        },
        {
            "role": "assistant",
            "prompt": "robert's email is robert.brown@example.com.",
            "response": "[NAME_1]'s email is [EMAIL_1].",
        },
        {
            "role": "user",
            "prompt": "the bsn for jane is 98765432.",
            "response": "The BSN for [NAME_1] is [BSN_1].",
        },
        {
            "role": "assistant",
            "prompt": "the bsn for jane is 98765432.",
            "response": "The BSN for [NAME_1] is [BSN_1].",
        },
        {
            "role": "user",
            "prompt": "contact me at myemail@domain..com.",
            "response": "Contact me at [EMAIL_1].",
        },
        {
            "role": "assistant",
            "prompt": "contact me at myemail@domain..com.",
            "response": "Contact me at [EMAIL_1].",
        },
        {
            "role": "user",
            "prompt": "my address is 123 main st.",
            "response": "My address is [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "my address is 123 main st.",
            "response": "My address is [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "i live at 456 oak street.",
            "response": "I live at [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "i live at 456 oak street.",
            "response": "I live at [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "my email is jane@example.com.",
            "response": "My email is [EMAIL_1].",
        },
        {
            "role": "assistant",
            "prompt": "my email is jane@example.com.",
            "response": "My email is [EMAIL_1].",
        },
        {
            "role": "user",
            "prompt": "i was born on 12/01/2024.",
            "response": "I was born on [DOB_1].",
        },
        {
            "role": "assistant",
            "prompt": "i was born on 12/01/2024.",
            "response": "I was born on [DOB_1].",
        },
        {
            "role": "user",
            "prompt": "michael's zip code is 12345.",
            "response": "[NAME_1]'s zip code is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "michael's zip code is 12345.",
            "response": "[NAME_1]'s zip code is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "i have a BSN 12345678.",
            "response": "I have a BSN [BSN_1].",
        },
        {
            "role": "assistant",
            "prompt": "i have a BSN 12345678.",
            "response": "I have a BSN [BSN_1].",
        },
        {
            "role": "user",
            "prompt": "my name is john, but people call me j.",
            "response": "My name is [NAME_1], but people call me [NAME_2].",
        },
        {
            "role": "assistant",
            "prompt": "my name is john, but people call me j.",
            "response": "My name is [NAME_1], but people call me [NAME_2].",
        },
        {
            "role": "user",
            "prompt": "the email for emily is emily@sample.com.",
            "response": "The email for [NAME_1] is [EMAIL_1].",
        },
        {
            "role": "assistant",
            "prompt": "the email for emily is emily@sample.com.",
            "response": "The email for [NAME_1] is [EMAIL_1].",
        },
        {
            "role": "user",
            "prompt": "my postal code is 54321.",
            "response": "My postal code is [ZIP_1].",
        },
        {
            "role": "assistant",
            "prompt": "my postal code is 54321.",
            "response": "My postal code is [ZIP_1].",
        },
        {
            "role": "user",
            "prompt": "i am from new york city.",
            "response": "I am from [ADDRESS_1].",
        },
        {
            "role": "assistant",
            "prompt": "i am from new york city.",
            "response": "I am from [ADDRESS_1].",
        },
        {
            "role": "user",
            "prompt": "tom is a friend of mine.",
            "response": "[NAME_1] is a friend of mine.",
        },
        {
            "role": "assistant",
            "prompt": "tom is a friend of mine.",
            "response": "[NAME_1] is a friend of mine.",
        },
        {
            "role": "user",
            "prompt": "my friend's email is friend@example.com.",
            "response": "My friend's email is [EMAIL_1].",
        },
        {
            "role": "assistant",
            "prompt": "my friend's email is friend@example.com.",
            "response": "My friend's email is [EMAIL_1].",
        },
        {
            "role": "user",
            "prompt": "i was born in 2024.",
            "response": "I was born in [DOB_1].",
        },
        {
            "role": "assistant",
            "prompt": "i was born in 2024.",
            "response": "I was born in [DOB_1].",
        },
        # Cases without personal data
        {
            "role": "user",
            "prompt": "The weather in London is quite unpredictable in December.",
            "response": "The weather in London is quite unpredictable in December.",
        },
        {
            "role": "assistant",
            "prompt": "The weather in London is quite unpredictable in December.",
            "response": "The weather in London is quite unpredictable in December.",
        },
        {
            "role": "user",
            "prompt": "Tomorrow is a national holiday, and many people are excited about the parade.",
            "response": "Tomorrow is a national holiday, and many people are excited about the parade.",
        },
        {
            "role": "assistant",
            "prompt": "Tomorrow is a national holiday, and many people are excited about the parade.",
            "response": "Tomorrow is a national holiday, and many people are excited about the parade.",
        },
        {
            "role": "user",
            "prompt": "The capital of France is Paris, and it's known for its rich history and culture.",
            "response": "The capital of France is Paris, and it's known for its rich history and culture.",
        },
        {
            "role": "assistant",
            "prompt": "The capital of France is Paris, and it's known for its rich history and culture.",
            "response": "The capital of France is Paris, and it's known for its rich history and culture.",
        },
        {
            "role": "user",
            "prompt": "In 2024, the Olympics will be hosted by Paris, and athletes from all over the world will compete.",
            "response": "In 2024, the Olympics will be hosted by Paris, and athletes from all over the world will compete.",
        },
        {
            "role": "assistant",
            "prompt": "In 2024, the Olympics will be hosted by Paris, and athletes from all over the world will compete.",
            "response": "In 2024, the Olympics will be hosted by Paris, and athletes from all over the world will compete.",
        },
        {
            "role": "user",
            "prompt": "I enjoy reading books in my free time, especially science fiction novels.",
            "response": "I enjoy reading books in my free time, especially science fiction novels.",
        },
        {
            "role": "assistant",
            "prompt": "I enjoy reading books in my free time, especially science fiction novels.",
            "response": "I enjoy reading books in my free time, especially science fiction novels.",
        },
        {
            "role": "user",
            "prompt": "The museum closes at 6 PM, so we should plan to arrive before that.",
            "response": "The museum closes at 6 PM, so we should plan to arrive before that.",
        },
        {
            "role": "assistant",
            "prompt": "The museum closes at 6 PM, so we should plan to arrive before that.",
            "response": "The museum closes at 6 PM, so we should plan to arrive before that.",
        },
        {
            "role": "user",
            "prompt": "Mount Everest is the highest mountain in the world, located in the Himalayas.",
            "response": "Mount Everest is the highest mountain in the world, located in the Himalayas.",
        },
        {
            "role": "assistant",
            "prompt": "Mount Everest is the highest mountain in the world, located in the Himalayas.",
            "response": "Mount Everest is the highest mountain in the world, located in the Himalayas.",
        },
        {
            "role": "user",
            "prompt": "The Great Wall of China stretches over 13,000 miles and is one of the most famous landmarks in the world.",
            "response": "The Great Wall of China stretches over 13,000 miles and is one of the most famous landmarks in the world.",
        },
        {
            "role": "assistant",
            "prompt": "The Great Wall of China stretches over 13,000 miles and is one of the most famous landmarks in the world.",
            "response": "The Great Wall of China stretches over 13,000 miles and is one of the most famous landmarks in the world.",
        },
        {
            "role": "user",
            "prompt": "She has a passion for photography and often travels to capture stunning landscapes.",
            "response": "She has a passion for photography and often travels to capture stunning landscapes.",
        },
        {
            "role": "assistant",
            "prompt": "She has a passion for photography and often travels to capture stunning landscapes.",
            "response": "She has a passion for photography and often travels to capture stunning landscapes.",
        },
        {
            "role": "user",
            "prompt": "The concert will take place in Central Park next Saturday.",
            "response": "The concert will take place in Central Park next Saturday.",
        },
        {
            "role": "assistant",
            "prompt": "The concert will take place in Central Park next Saturday.",
            "response": "The concert will take place in Central Park next Saturday.",
        },
        {
            "role": "user",
            "prompt": "Python is a popular programming language used for web development, data analysis, and automation.",
            "response": "Python is a popular programming language used for web development, data analysis, and automation.",
        },
        {
            "role": "assistant",
            "prompt": "Python is a popular programming language used for web development, data analysis, and automation.",
            "response": "Python is a popular programming language used for web development, data analysis, and automation.",
        },
        {
            "role": "user",
            "prompt": "The movie is set to be released in theaters worldwide in 2025.",
            "response": "The movie is set to be released in theaters worldwide in 2025.",
        },
        {
            "role": "assistant",
            "prompt": "The movie is set to be released in theaters worldwide in 2025.",
            "response": "The movie is set to be released in theaters worldwide in 2025.",
        },
        {
            "role": "user",
            "prompt": "The book I'm reading right now is a bestseller, and it's about artificial intelligence.",
            "response": "The book I'm reading right now is a bestseller, and it's about artificial intelligence.",
        },
        {
            "role": "assistant",
            "prompt": "The book I'm reading right now is a bestseller, and it's about artificial intelligence.",
            "response": "The book I'm reading right now is a bestseller, and it's about artificial intelligence.",
        },
        {
            "role": "user",
            "prompt": "Her favorite hobby is painting landscapes, and she has her own studio.",
            "response": "Her favorite hobby is painting landscapes, and she has her own studio.",
        },
        {
            "role": "assistant",
            "prompt": "Her favorite hobby is painting landscapes, and she has her own studio.",
            "response": "Her favorite hobby is painting landscapes, and she has her own studio.",
        },
        {
            "role": "user",
            "prompt": "The football match ended in a draw, with both teams scoring two goals each.",
            "response": "The football match ended in a draw, with both teams scoring two goals each.",
        },
        {
            "role": "assistant",
            "prompt": "The football match ended in a draw, with both teams scoring two goals each.",
            "response": "The football match ended in a draw, with both teams scoring two goals each.",
        },
        {
            "role": "user",
            "prompt": "The weather today is sunny with a high of 75°F. Everyone should enjoy their day outside!",
            "response": "The weather today is sunny with a high of 75°F. Everyone should enjoy their day outside!",
        },
        {
            "role": "assistant",
            "prompt": "The weather today is sunny with a high of 75°F. Everyone should enjoy their day outside!",
            "response": "The weather today is sunny with a high of 75°F. Everyone should enjoy their day outside!",
        },
        {
            "role": "user",
            "prompt": "The conference will take place next month, bringing together experts from various fields to discuss the latest trends.",
            "response": "The conference will take place next month, bringing together experts from various fields to discuss the latest trends.",
        },
        {
            "role": "assistant",
            "prompt": "The conference will take place next month, bringing together experts from various fields to discuss the latest trends.",
            "response": "The conference will take place next month, bringing together experts from various fields to discuss the latest trends.",
        },
        {
            "role": "user",
            "prompt": "The team's performance in the last match was outstanding, earning them a spot in the playoffs.",
            "response": "The team's performance in the last match was outstanding, earning them a spot in the playoffs.",
        },
        {
            "role": "assistant",
            "prompt": "The team's performance in the last match was outstanding, earning them a spot in the playoffs.",
            "response": "The team's performance in the last match was outstanding, earning them a spot in the playoffs.",
        },
        {
            "role": "user",
            "prompt": "Our new product will be launched next quarter, aiming to revolutionize the industry with its innovative features.",
            "response": "Our new product will be launched next quarter, aiming to revolutionize the industry with its innovative features.",
        },
        {
            "role": "assistant",
            "prompt": "Our new product will be launched next quarter, aiming to revolutionize the industry with its innovative features.",
            "response": "Our new product will be launched next quarter, aiming to revolutionize the industry with its innovative features.",
        },
    ]
    qa_data = [
        {
            "role": "user",
            "prompt": "What is your name?",
            "response": "I am blueVi-GPT.",
        },
        {
            "role": "user",
            "prompt": "Who are you?",
            "response": "I'm blueVi-GPT, an AI language model created by Visma Verzuim. My purpose is to assist with answering questions, providing information, and engaging in conversation. I'm here to help with whatever you need!",
        },
        {
            "role": "user",
            "prompt": "Who created you?",
            "response": "I was created by Visma Verzuim.",
        },
        {
            "role": "user",
            "prompt": "What can you do?",
            "response": "I can assist with various tasks such as answering questions, providing information, and helping you with programming queries.",
        },
        {
            "role": "user",
            "prompt": "Where are you deployed?",
            "response": "I am deployed in various environments, including cloud services, and available for different applications.",
        },
        {
            "role": "user",
            "prompt": "How do you learn?",
            "response": "I learn from vast datasets that are used to train my machine learning models, allowing me to understand and generate responses.",
        },
        {
            "role": "user",
            "prompt": "Can you speak other languages?",
            "response": "Yes, I can communicate in multiple languages, including English, Dutch, and more.",
        },
        {
            "role": "user",
            "prompt": "How do you handle data?",
            "response": "I process data based on inputs I receive, ensuring privacy and adhering to guidelines for data protection.",
        },
        {
            "role": "user",
            "prompt": "Wat is je naam?",
            "response": "Ik ben blueVi-GPT.",
        },
        {
            "role": "user",
            "prompt": "Wie ben jij?",
            "response": "Ik ben blueVi-GPT, een AI-taalmodel gemaakt door Visma Verzuim. Mijn doel is om te helpen met het beantwoorden van vragen, het verstrekken van informatie en het voeren van gesprekken. Ik ben hier om je te helpen met wat je nodig hebt!",
        },
        {
            "role": "user",
            "prompt": "Wie heeft je gemaakt?",
            "response": "Ik ben gemaakt door Visma Verzuim.",
        },
        {
            "role": "user",
            "prompt": "Wat kun je doen?",
            "response": "Ik kan helpen met verschillende taken, zoals het beantwoorden van vragen, het verstrekken van informatie en het helpen met programmeervragen.",
        },
        {
            "role": "user",
            "prompt": "Waar ben je ingezet?",
            "response": "Ik ben ingezet in verschillende omgevingen, waaronder cloudservices, en beschikbaar voor verschillende toepassingen.",
        },
        {
            "role": "user",
            "prompt": "Hoe leer je?",
            "response": "Ik leer van enorme datasets die worden gebruikt om mijn machine learning-modellen te trainen, waardoor ik in staat ben om reacties te begrijpen en te genereren.",
        },
        {
            "role": "user",
            "prompt": "Kun je andere talen spreken?",
            "response": "Ja, ik kan communiceren in meerdere talen, waaronder Engels, Nederlands en meer.",
        },
        {
            "role": "user",
            "prompt": "Hoe ga je om met gegevens?",
            "response": "Ik verwerk gegevens op basis van de invoer die ik ontvang, waarbij ik de privacy waarborg en me houd aan richtlijnen voor gegevensbescherming.",
        },
        {
            "role": "user",
            "prompt": "What is your name?",
            "response": "I am blueVi-GPT.",
        },
        {
            "role": "user",
            "prompt": "Who are you?",
            "response": "I'm blueVi-GPT, an AI language model created by Visma Verzuim. My purpose is to assist with answering questions, providing information, and engaging in conversation. I'm here to help with whatever you need!",
        },
        {
            "role": "user",
            "prompt": "Who created you?",
            "response": "I was created by Visma Verzuim.",
        },
        {
            "role": "user",
            "prompt": "What can you do?",
            "response": "I can assist with various tasks such as answering questions, providing information, and helping you with programming queries.",
        },
        {
            "role": "user",
            "prompt": "Where are you deployed?",
            "response": "I am deployed in various environments, including cloud services, and available for different applications.",
        },
        {
            "role": "user",
            "prompt": "How do you learn?",
            "response": "I learn from vast datasets that are used to train my machine learning models, allowing me to understand and generate responses.",
        },
        {
            "role": "user",
            "prompt": "Can you speak other languages?",
            "response": "Yes, I can communicate in multiple languages, including English, Dutch, and more.",
        },
        {
            "role": "user",
            "prompt": "How do you handle data?",
            "response": "I process data based on inputs I receive, ensuring privacy and adhering to guidelines for data protection.",
        },
        {
            "role": "user",
            "prompt": "Wat is je naam?",
            "response": "Ik ben blueVi-GPT.",
        },
        {
            "role": "user",
            "prompt": "Wie ben jij?",
            "response": "Ik ben blueVi-GPT, een AI-taalmodel gemaakt door Visma Verzuim. Mijn doel is om te helpen met het beantwoorden van vragen, het verstrekken van informatie en het voeren van gesprekken. Ik ben hier om je te helpen met wat je nodig hebt!",
        },
        {
            "role": "user",
            "prompt": "Wie heeft je gemaakt?",
            "response": "Ik ben gemaakt door Visma Verzuim.",
        },
        {
            "role": "user",
            "prompt": "Wat kun je doen?",
            "response": "Ik kan helpen met verschillende taken, zoals het beantwoorden van vragen, het verstrekken van informatie en het helpen met programmeervragen.",
        },
        {
            "role": "user",
            "prompt": "Waar ben je ingezet?",
            "response": "Ik ben ingezet in verschillende omgevingen, waaronder cloudservices, en beschikbaar voor verschillende toepassingen.",
        },
        {
            "role": "user",
            "prompt": "Hoe leer je?",
            "response": "Ik leer van enorme datasets die worden gebruikt om mijn machine learning-modellen te trainen, waardoor ik in staat ben om reacties te begrijpen en te genereren.",
        },
        {
            "role": "user",
            "prompt": "Kun je andere talen spreken?",
            "response": "Ja, ik kan communiceren in meerdere talen, waaronder Engels, Nederlands en meer.",
        },
        {
            "role": "user",
            "prompt": "Hoe ga je om met gegevens?",
            "response": "Ik verwerk gegevens op basis van de invoer die ik ontvang, waarbij ik de privacy waarborg en me houd aan richtlijnen voor gegevensbescherming.",
        },
    ]
    personal_data = [
        # English Cases
        {
            "role": "assistant",
            "prompt": "Alice Smith's email is allice.smith@example..com. Her BSN is 12345678. Her address is 15 Elm Street, and her zip code is 12345.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Robert Brown was born on 12-01-2024. His email is robert.brown@example.com, and his BSN is 87654321. He lives at 222 Ash St.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Natalie Green's email is nat.green@example..com. Her birthday is 01/12/2024, and her postal code is 34567. She resides at 789 Birch Lane.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Mark Thompson's email is mark.thompson@example.com. His BSN is 54321098, and his home address is 654 Maple Drive. His date of birth is 12/01/2024.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Linda Johnson's email is linda.johnson@example..com, and her BSN is 11223344. She lives at 321 Willow Way, with a postal code of 98765.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Peter's date of birth is 12/31/2024. His email is peter@example.com. His home address is 98 Pine Street, and his zip is 45678.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Julia's email is julia@example.com, her BSN is 22334455. She was born on 01-12-2024, and she lives at 234 Oak Lane.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Tom's email is tom123@example.com, and his BSN is 99988877. He was born on 12/01/2024 and lives at 100 Cedar Ave.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "My cousin Anna's email is anna123@example.com. Her BSN is 77788899. She lives at 77 Elm St. Her birthday is 1st January 2024.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Max's date of birth is 2024-01-01. His email is max@example.com. His postal code is 12345, and he lives at 789 Cedar Street.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Sarah Davis's email is sarah.davis@example.com. She was born on 01/12/2024. Her address is 789 Maple Street, and her postal code is 87654.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Jacob Martin was born on 12/01/2024. His BSN is 34567890, and his email is jacob.martin@example.com. He lives at 321 Cedar Avenue.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Emily White's email is emily.white@example.com. She was born on the 1st of December 2024. Her zip code is 54321, and her house number is 24.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "My friend Michael was born on December 1st, 2024. His email is michael@example.com, and his postal code is 98765. His home address is 456 Birch Avenue.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Lily Evans's birthday is 12/01/2024. Her email is lily.evans@example.com, and her BSN is 45678901. She lives at 101 Rosewood St.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Contact John, who was born on 01/12/2024, at john.doe@example.com. His home address is 789 Oak Drive, and his postal code is 10234.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Sophia Green's birthday is 1st December, 2024. Her BSN is 98765432, and her email is sophia.green@example.com. She lives at 888 Willow Lane.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Ryan's date of birth is December 01, 2024. His zip code is 54321, and his email is ryan@example.com. His BSN is 23456789.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Jane Smith's birthday is 01/12/2024. She lives at 123 Palm Street, and her postal code is 56789. Her email is jane.smith@example.com.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Marcus Brown's email is marcus.brown@example.com, and his BSN is 12345678. His birthday is December 1st, 2024, and his home address is 456 Pine St.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Emily Carter was born on 01/12/2024. Her email is emily.carter@example.com, and her BSN is 98765432. She lives at 789 Spruce Lane.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "John Doe's email is J.Simpson@@netwrix..com. His BSN is 12345678.9. His home address is 10 Langelo! His zip code is 7666mc. His Mastercard number is 5258-7041-08753590 and his visa number is 4563 7568 5698 4587. His iban number is nl91abna0417164300. His date of birth is 1/1/90. His IP address is 192.168.1.1.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Olivia Doe's email is d.olivia@netwrix..com. Her BSN is 98765432. Her home address is 60 Langelo Langelo. Her ZIP code is 7612-MC. Her Mastercard number is 5258-7041-0873577 and her visa number is 4511-7500-0000-0000. Her iban number is DE91abna0417164300. Her dob is 1/1/91. Her ip is 192.168.2.2.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "John and Olivia went to the market. John's email is john.doe@example.com, and Olivia's BSN is 98765432. Their addresses are 123 Main St and 456 Oak St. The zip codes are 98765 and 12345. John's Mastercard number is 1234-5678-9876-5432.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Contact me at invalid@domain..com, my BSN is 123-45-678, and my IP is 192.168.0.2.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "My friend Emily has an email emily@example.com and her BSN is 23456789. She lives at 789 Pine St. Her zip code is 67890.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "The company contacted me through my email address: john@example.com. I have a bank account number: 1234567890123456.",
            "response": "True" "",
        },
        # Dutch Cases
        {
            "role": "assistant",
            "prompt": "Jan Jansen's e-mailadres is jan.jansen@@voorbeeld.nl. Zijn BSN is 12345678.9. Zijn huisadres is 10 Hoofdstraat! Zijn postcode is 7666mc. Zijn Mastercardnummer is 5258-7041-08753590 en zijn Visa-nummer is 4563 7568 5698 4587. Zijn IBAN-nummer is NL91ABNA0417164300. Zijn geboortedatum is 1/1/90. Zijn IP-adres is 192.168.1.1.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Olivia Jansen's e-mailadres is d.olivia@voorbeeld..nl. Haar BSN is 98765432. Haar huisadres is 60 Langelaan. Haar postcode is 7612-MC. Haar Mastercardnummer is 5258-7041-0873577 en haar Visa-nummer is 4511-7500-0000-0000. Haar IBAN-nummer is DE91ABNA0417164300. Haar geboortedatum is 1/1/91. Haar IP is 192.168.2.2.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Jan en Olivia waren in het park. Jan's e-mailadres is jan.doe@voorbeeld.com, en Olivia's BSN is 98765432. Hun adressen zijn 123 Hoofdstraat en 456 Eiklaan. De postcodes zijn 98765 en 12345. Jan's Mastercardnummer is 1234-5678-9876-5432.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Neem contact met me op via jan.doe@gmail..com, mijn BSN is 123-45-678, en mijn IP is 192.168.0.2.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Sophie werkt bij een bedrijf en haar e-mailadres is sophie@voorbeeld.com. Haar BSN is 23456789. Ze woont op 321 Laan van de Vrijheid. Haar postcode is 12345.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "De geboortedatum van Jan Jansen is 01/12/2024. Zijn leeftijd is 25 jaar.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Olivia Jansen werd geboren op 12/01/2024 en is nu 30 jaar oud.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Sophie is geboren op 1 december 2024 en ze is 45 jaar.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Karel werd geboren op de eerste december 2024. Hij is nu 40 jaar.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "De geboortedatum van Maria is 2024-12-01. Ze is nu 50 jaar oud.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Haar geboortedatum is 12 januari 2024, en ze is 33 jaar.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "De geboortedatum van mijn vriend is 01-12-2024. Hij is nu 35 jaar.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Mijn broer is geboren op 1-12-2024 en hij is nu 27 jaar.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Ze is op 1e december 2024 geboren, en ze is 29 jaar oud.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Zijn geboortedatum is 12 januari 2024. Hij is nu 37 jaar.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Ik heb een nieuw emailadres: mijn.email@voorbeeld.nl en mijn BSN is 34567890. Mijn huisadres is 88 Nieuweweg.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Sara de Vries's e-mailadres is sara.de.vries@voorbeeld.nl. Haar BSN is 12345678. Haar huisadres is 12 Fijnstraat, en haar postcode is 98765.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "De geboortedatum van Piet is 01-12-2024. Zijn e-mailadres is piet@example.nl, en zijn BSN is 87654321. Hij woont op 456 Laan van de Vrijheid.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Karin's e-mailadres is karin@example.nl, en haar BSN is 22223333. Haar geboortedatum is 12/01/2024, en ze woont op 111 Eikenlaan.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Rob Jansen's e-mailadres is rob.jansen@voorbeeld.nl. Zijn BSN is 11112222, en zijn huisadres is 88 Hoofdstraat. Zijn geboortedatum is 01-12-2024.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Annelies heeft een nieuw emailadres: annelies@voorbeeld.nl. Haar BSN is 99998888, en ze woont op 24 Dennenlaan. Haar postcode is 12345.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Janine werd geboren op 12/01/2024 en ze is nu 26 jaar oud. Haar email is janine@example.nl.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Sophie werkt bij een bedrijf. Haar e-mailadres is sophie@voorbeeld.nl, en haar BSN is 34567890. Ze woont op 555 Klaproosstraat.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Frits heeft een nieuwe email: frits@example.nl. Zijn BSN is 77778888. Hij woont op 80 Rozenlaan.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "De geboortedatum van Lotte is 01-01-2024. Ze woont op 34 Bovenstraat. Haar e-mailadres is lotte@voorbeeld.nl.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Milan heeft een emailadres: milan@example.nl. Zijn BSN is 55556666. Hij woont op 12 Appelstraat.",
            "response": "True" "",
        },
        # English Cases with Typos and Incorrect Formats
        {
            "role": "assistant",
            "prompt": "Allice Smith's email is allice.smith@example..com. Her BSN is 12345678. Her home adress is 45 Elm St, and her postal code is 9021. Her MasterCard number is 5412-3456-7890-12345 and her IBAN number is GB29NWBK60161331926819.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Mark Jonson has an email mark.j@exampl.com. His BSN is 98765432. His adress is 72 Oak Ave. His zip code is 54321.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Email: test.email@invalid..com. BSN: 321-45-6789. IP address: 172.16.254.01.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Contact me at my email address: not.a.valid.email@domain..com. My BSN is 123456789, and my IP is 192.168.0.256.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "There is a problem with my email, which is john.doe@exampl.com, and my BSN is 234-56-7890.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Johhn's birthdate is Dec 32, 2024. He is now 26 yrs old.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Smantha was born on 15/13/2024, she's 34 year.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Jke's date of brith is 2024-31-11 and he is 28 yo.",
            "response": "True" "",
        },
        {
            "prompt": "Mia's dob is 01//12/2024. She is 24.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "His birthdate is 1/13/2024, he is now 35 year.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Born on Febuary 29, 2024, Jake is 33 years old.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Ben's birtdate: 2024/12/01. He is twenty five.",
            "response": "True" "",
        },
        {"prompt": "Anna, dob 12th Oct, 2024, age 32.", "response": "True" ""},
        {
            "role": "assistant",
            "prompt": "Her DOB is 01-33-2024, and she's 22 yr.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "His date of brth is 12/01/2024. Now he's 45 yo.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "asdicjsdocnsdjk Allice Smith's email is allice.smith@example..com. Her BSN is 12345678. Her home adress is 45 Elm St, and her postal code is 9021. Her MasterCard number is 5412-3456-7890-12345 and her IBAN number is GB29NWBK60161331926819.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "sdkjcnskjdcnkjsdcnsdncljkncpda Ma rk Jon son h as a n em ail m ar k.j @e xa m pl .c om.  H i s   B     SN i s 9 876 543 2. H is  ad re s s  i s 72 O ak Ave. His zip code is 5 4321.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": " asdcasdca ad sasdcdsdacc email: test.email@invalid..com. BSN: 321-45-6789. IP address: 172.16.254.01. sdcasdcafwqefsad2324567",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "adscsdf234wedqwsda C on  ta          c t m e  at m y ema il add res s: not.a.v. alid .email@domain..com.   My BS N is 1. 234. 567 89, and my IP is 192.168.0.256. asr123rdascsadc",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "savcasf 23r fsda12 werfThere is a problem with my email, which is john.doe@exampl.com, and my BSN is 234-56-7890.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Johhn's bisadcasdr123t hdate is Dec 32, 2024. He is now 26 yrs old.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "sdfsad2 34 asc ac scSmantha was born on 15/13/2024, she's 34 year.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Jke's date of brith is 2024-31-11 and he is 28 yo.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Mia's dob is 01//12/2024. She is 24.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "His birthdate is 1/13/2024, he is now 35 year.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Born on Febuary 29, 2024, Jake is 33 years old.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Ben's birtdate: 2024/12/01. He is twenty five.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Anna, dob 12th Oct, 2024, age 32.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Her DOB is 01-33-2024, and she's 22 yr.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "His date of brth is 12/01/2024. Now he's 45 yo.",
            "response": "True" "",
        },
        # Dutch Cases with Typos and Incorrect Formats
        {
            "role": "assistant",
            "prompt": "Liesbeth heeft een e-mailadres: liesbeth@@voorbeeld.nl. Haar BSN is 2345678. Ze woont op 101 Beukenlaan, postcode 1234ABC.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Pieter's email is pieter@exampl.nl. Zijn BSN is 34567890. Zijn huisadrees is 65 Lindelaan. Zijn postcode is 5678CD.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Neem contact met me op via mijn e-mailadres: info@bedrijf..nl. Mijn BSN is 456-78-901, en mijn IP is 192.168.0.300.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Kunt u me e-mailen op mijn.new.email@voorbeeld..nl? Mijn BSN is 12345678.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Ik heb een nieuw emailadres: mijn.email@voorbeeld.nl. Mijn BSN is 34567890. Mijn huisadres is 88 Nieuweweg. En mijn postcod is 1010AB.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "De geboortedatum van Jan Jnansen is 01-32-2024. Zijn leeftijd is 25 jaar.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Olivvia werd gebroen op 13/01/2024 en is nu 30 jaar oud.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Sophe is gebon op 1-12-2024 en ze is 45 jar.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "De geboortdatum van Maria is 2024-13-01. Ze is nu 50 jr oud.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Haar gebortedatum is 12 januari 2024, en ze is 33 j.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Mijn broer is geboren op 1-13-2024 en hij is nu 27 jaar.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Ze is op 1e dec 2024 geborne, en ze is 29 jaar oud.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Zijn geboortedatumm is 13 januari 2024. Hij is nu 37 jr.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "i am tom. tom's email is tom@example.com, his BSN is 12345678. he lives at 100 Cedar Ave, and his zip code is 45678.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "My name is SARA JOHNSON. My email is sara.johnson@exammple.com, and my BSN is 87654321. I live at 222 Ash St.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "allice.smith@example..com is my email. My BSN is 12345678. I reside at 15 elm street.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "robert's date of birth is 12-31-2024. his email is robert.brown@example.com, and his BSN is 34567890.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Linda johnson's email is linda.johnson@exammple..com, and her bsn is 11223344. she lives at 321 willow way.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "my friend max's date of birth is 2024-01-01. his email is max@example..com. his postal code is 12345.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "JULIA's email is julia123@example.com, her bsn is 22334455. She was born on 01-12-2024.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Tom's email is TOM123@EXAMPLE.COM, and his bsn is 99988877. he was born on 12/01/2024.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "my cousin Anna's email is anna123@EXAMPLE.COM. Her BSN is 77788899, and she lives at 77 Elm St.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Max's date of birth is 01/1/2024. his email is max@example.com. His postal code is 12345.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Contact me at invalid@domain..com, my bsn is 123-45-678, and my ip is 192.168.0.2.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Emily's email is emily@example..com, and her BSN is 23456789. she lives at 789 pine st.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "JOHN DOE's email is john.doe@example.com. His bsn is 12345678. he lives at 10 langelo!",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "Contact olivia at d.olivia@netwrix..com. Her bsn is 98765432. Her address is 60 langelo langelo.",
            "response": "True" "",
        },
        {"role": "assistant", "prompt": "i am tom.", "response": "True" ""},
        {
            "role": "assistant",
            "prompt": "my name is alice.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "robert's email is robert.brown@example.com.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "the bsn for jane is 98765432.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "contact me at myemail@domain..com.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "my address is 123 main st.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "i live at 456 oak street.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "my email is jane@example.com.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "i was born on 12/01/2024.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "michael's zip code is 12345.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "i have a BSN 12345678.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "my name is john, but people call me j.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "the email for emily is emily@sample.com.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "my postal code is 54321.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "i am from new york city.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "tom is a friend of mine.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "my friend's email is friend@example.com.",
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": "i was born in 2024.",
            "response": "True" "",
        },
        # Cases without personal data
        {
            "role": "assistant",
            "prompt": "The weather in London is quite unpredictable in December.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "Tomorrow is a national holiday, and many people are excited about the parade.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The capital of France is Paris, and it's known for its rich history and culture.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "In 2024, the Olympics will be hosted by Paris, and athletes from all over the world will compete.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I enjoy reading books in my free time, especially science fiction novels.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The museum closes at 6 PM, so we should plan to arrive before that.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "Mount Everest is the highest mountain in the world, located in the Himalayas.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The Great Wall of China stretches over 13,000 miles and is one of the most famous landmarks in the world.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "She has a passion for photography and often travels to capture stunning landscapes.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The concert will take place in Central Park next Saturday.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "Python is a popular programming language used for web development, data analysis, and automation.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The movie is set to be released in theaters worldwide in 2025.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The book I'm reading right now is a bestseller, and it's about artificial intelligence.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "Her favorite hobby is painting landscapes, and she has her own studio.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The football match ended in a draw, with both teams scoring two goals each.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The weather today is sunny with a high of 75°F. Everyone should enjoy their day outside!",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The conference will take place next month, bringing together experts from various fields to discuss the latest trends.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The team's performance in the last match was outstanding, earning them a spot in the playoffs.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "Our new product will be launched next quarter, aiming to revolutionize the industry with its innovative features.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The company has decided to implement a new remote work policy to improve employee satisfaction.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The school is organizing a charity event to support local families in need. Everyone is encouraged to participate.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The library will be hosting a book fair next weekend, featuring various genres and activities for all ages.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The hiking trail offers breathtaking views and is a popular destination for outdoor enthusiasts.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "This recipe is perfect for a cozy night in, combining flavors that everyone loves.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "Traveling is a wonderful way to experience new cultures and meet different people around the world.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "Learning a new language can be challenging, but it opens up a world of opportunities.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "Exercise is essential for maintaining a healthy lifestyle and can greatly improve mental well-being.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "The latest smartphone has impressive features that enhance user experience and productivity.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "Art plays a crucial role in society, providing a means of expression and communication.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "Music brings people together and can evoke a wide range of emotions.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "Volunteering can be a rewarding experience, allowing individuals to give back to their communities.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I love going for walks in the park, especially when the weather is nice.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I enjoy reading books about different cultures and places around the world.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I recently watched a movie that inspired me to pursue my passions more actively.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I think learning a new language can be challenging but also very rewarding.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I appreciate the beauty of nature and often spend my weekends hiking.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I have been experimenting with new recipes in the kitchen, and it has been a fun experience.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I believe that staying active is important for a healthy lifestyle.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I enjoy attending local events and meeting new people in my community.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I like to take time to reflect on my goals and aspirations regularly.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I find that listening to music helps me focus while I work.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I often share my thoughts on various topics with friends and family.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I think that volunteering can be a fulfilling way to give back to the community.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I look forward to the weekends, as they give me a chance to relax and recharge.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I appreciate art and enjoy visiting galleries to see new exhibitions.",
            "response": "False",
        },
        {
            "role": "assistant",
            "prompt": "I enjoy exploring new hobbies that challenge me and help me grow.",
            "response": "False",
        },
    ]
    phx_operation_data = [
        {
            "role": "user",
            "prompt": "I want to create a new operation.",
            "response": "To create a new operation, please provide the following details:\n1. Operation Name\n2. Invoice Description (Optional)\n3. Duration (in minutes)\n4. Whether invoicing will be applied (yes or no)\n5. Hourly Rate (if invoicing is applied)\n6. Unit Price (if invoicing is not applied)\n7. Methods of Consultation (TEL, WEB, LOC)",
        },
        {
            "role": "assistant",
            "prompt": "I want to create a new operation.",
            "response": "{\n  \"name\": \"\",\n  \"description\": \"\",\n  \"duration\": 0,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": None,\n  \"operationRateType\": None,\n  \"methodsOfConsult\": [\n    \"TEL\",\n    \"WEB\",\n    \"LOC\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "Operation name is 'Operation A'. The description is 'Consulting services'. The duration is 30 minutes. Invoicing will be applied with an hourly rate of 50. The method of consultation is TEL.",
            "response": "Great! Here are the details for the new operation:\n1. Operation Name: Operation A\n2. Description: Consulting services\n3. Duration: 30 minutes\n4. Invoicing: Yes, with Hourly Rate of 50\n5. Method of Consultation: TEL",
        },
        {
            "role": "assistant",
            "prompt": "Operation name is 'Operation A'. The description is 'Consulting services'. The duration is 30 minutes. Invoicing will be applied with an hourly rate of 50. The method of consultation is TEL.",
            "response": "{\n  \"name\": \"Operation A\",\n  \"description\": \"Consulting services\",\n  \"duration\": 30,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 50,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"TEL\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I want to create an operation named 'Operation B', with no invoicing. It will have a unit price of 100 and last 60 minutes. The consultation method will be WEB.",
            "response": "Here’s the configuration for 'Operation B':\n1. Operation Name: Operation B\n2. Description: (Optional)\n3. Duration: 60 minutes\n4. Invoicing: No\n5. Unit Price: 100\n6. Method of Consultation: WEB",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation B', with no invoicing. It will have a unit price of 100 and last 60 minutes. The consultation method will be WEB.",
            "response": "{\n  \"name\": \"Operation B\",\n  \"description\": \"\",\n  \"duration\": 60,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": 100,\n  \"operationRateType\": 2,\n  \"methodsOfConsult\": [\n    \"WEB\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I would like to create an operation named 'Operation C' with invoicing. It will be 45 minutes long with an hourly rate of 70. Methods of consultation will be TEL and LOC.",
            "response": "Here are the details for 'Operation C':\n1. Operation Name: Operation C\n2. Duration: 45 minutes\n3. Invoicing: Yes, with Hourly Rate of 70\n4. Method of Consultation: TEL, LOC",
        },
        {
            "role": "assistant",
            "prompt": "I would like to create an operation named 'Operation C' with invoicing. It will be 45 minutes long with an hourly rate of 70. Methods of consultation will be TEL and LOC.",
            "response": "{\n  \"name\": \"Operation C\",\n  \"description\": \"\",\n  \"duration\": 45,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 70,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"TEL\",\n    \"LOC\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I want to create a new operation.",
            "response": "To create a new operation, please provide the following details:\n1. Operation Name\n2. Invoice Description (Optional)\n3. Duration (in minutes)\n4. Whether invoicing will be applied (yes or no)\n5. Hourly Rate (if invoicing is applied)\n6. Unit Price (if invoicing is not applied)\n7. Methods of Consultation (TEL, WEB, LOC)",
        },
        {
            "role": "assistant",
            "prompt": "I want to create a new operation.",
            "response": "{\n  \"name\": \"\",\n  \"description\": \"\",\n  \"duration\": 0,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": None,\n  \"operationRateType\": None,\n  \"methodsOfConsult\": [\n    \"TEL\",\n    \"WEB\",\n    \"LOC\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I want to create an operation named 'Operation A', lasting 45 minutes, with an hourly rate of 50 and invoicing enabled. The consultation method will be TEL.",
            "response": "Here are the details for 'Operation A':\n1. Operation Name: Operation A\n2. Duration: 45 minutes\n3. Invoicing: Yes, with Hourly Rate of 50\n4. Method of Consultation: TEL",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation A', lasting 45 minutes, with an hourly rate of 50 and invoicing enabled. The consultation method will be TEL.",
            "response": "{\n  \"name\": \"Operation A\",\n  \"description\": \"\",\n  \"duration\": 45,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 50,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"TEL\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I want to create an operation named 'Operation B', with no invoicing, a unit price of 100, and lasting 60 minutes. The consultation method will be WEB.",
            "response": "Here’s the configuration for 'Operation B':\n1. Operation Name: Operation B\n2. Description: (Optional)\n3. Duration: 60 minutes\n4. Invoicing: No\n5. Unit Price: 100\n6. Method of Consultation: WEB",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation B', with no invoicing, a unit price of 100, and lasting 60 minutes. The consultation method will be WEB.",
            "response": "{\n  \"name\": \"Operation B\",\n  \"description\": \"\",\n  \"duration\": 60,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": 100,\n  \"operationRateType\": 2,\n  \"methodsOfConsult\": [\n    \"WEB\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I want to create an operation named 'Operation C' with invoicing enabled, an hourly rate of 70, and a duration of 45 minutes. The consultation methods will be TEL and LOC.",
            "response": "Here are the details for 'Operation C':\n1. Operation Name: Operation C\n2. Duration: 45 minutes\n3. Invoicing: Yes, with Hourly Rate of 70\n4. Method of Consultation: TEL, LOC",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation C' with invoicing enabled, an hourly rate of 70, and a duration of 45 minutes. The consultation methods will be TEL and LOC.",
            "response": "{\n  \"name\": \"Operation C\",\n  \"description\": \"\",\n  \"duration\": 45,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 70,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"TEL\",\n    \"LOC\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I would like to create an operation named 'Operation D' with a unit price of 200, lasting 30 minutes. The method of consultation will be WEB.",
            "response": "Here are the details for 'Operation D':\n1. Operation Name: Operation D\n2. Duration: 30 minutes\n3. Invoicing: No\n4. Unit Price: 200\n5. Method of Consultation: WEB",
        },
        {
            "role": "assistant",
            "prompt": "I would like to create an operation named 'Operation D' with a unit price of 200, lasting 30 minutes. The method of consultation will be WEB.",
            "response": "{\n  \"name\": \"Operation D\",\n  \"description\": \"\",\n  \"duration\": 30,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": 200,\n  \"operationRateType\": 2,\n  \"methodsOfConsult\": [\n    \"WEB\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I want to create an operation named 'Operation E', with invoicing, an hourly rate of 120, and a duration of 90 minutes. The methods of consultation will be TEL, WEB, and LOC.",
            "response": "Here are the details for 'Operation E':\n1. Operation Name: Operation E\n2. Duration: 90 minutes\n3. Invoicing: Yes, with Hourly Rate of 120\n4. Methods of Consultation: TEL, WEB, LOC",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation E', with invoicing, an hourly rate of 120, and a duration of 90 minutes. The methods of consultation will be TEL, WEB, and LOC.",
            "response": "{\n  \"name\": \"Operation E\",\n  \"description\": \"\",\n  \"duration\": 90,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 120,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"TEL\",\n    \"WEB\",\n    \"LOC\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I want to create an operation named 'Operation F' without invoicing. The duration will be 120 minutes, and the method of consultation will be TEL.",
            "response": "Here are the details for 'Operation F':\n1. Operation Name: Operation F\n2. Duration: 120 minutes\n3. Invoicing: No\n4. Method of Consultation: TEL",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation F' without invoicing. The duration will be 120 minutes, and the method of consultation will be TEL.",
            "response": "{\n  \"name\": \"Operation F\",\n  \"description\": \"\",\n  \"duration\": 120,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": None,\n  \"operationRateType\": 2,\n  \"methodsOfConsult\": [\n    \"TEL\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I want to create an operation named 'Operation G', with invoicing enabled, an hourly rate of 150, and a duration of 75 minutes. The consultation methods will be TEL and LOC.",
            "response": "Here are the details for 'Operation G':\n1. Operation Name: Operation G\n2. Duration: 75 minutes\n3. Invoicing: Yes, with Hourly Rate of 150\n4. Methods of Consultation: TEL, LOC",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation G', with invoicing enabled, an hourly rate of 150, and a duration of 75 minutes. The consultation methods will be TEL and LOC.",
            "response": "{\n  \"name\": \"Operation G\",\n  \"description\": \"\",\n  \"duration\": 75,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 150,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"TEL\",\n    \"LOC\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I want to create an operation named 'Operation H', with invoicing disabled, a unit price of 50, and a duration of 30 minutes. The method of consultation will be WEB.",
            "response": "Here are the details for 'Operation H':\n1. Operation Name: Operation H\n2. Duration: 30 minutes\n3. Invoicing: No\n4. Unit Price: 50\n5. Method of Consultation: WEB",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation H', with invoicing disabled, a unit price of 50, and a duration of 30 minutes. The method of consultation will be WEB.",
            "response": "{\n  \"name\": \"Operation H\",\n  \"description\": \"\",\n  \"duration\": 30,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": 50,\n  \"operationRateType\": 2,\n  \"methodsOfConsult\": [\n    \"WEB\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I want to create an operation named 'Operation I', with invoicing enabled, an hourly rate of 200, and a duration of 90 minutes. The methods of consultation will be TEL, WEB, and LOC.",
            "response": "Here are the details for 'Operation I':\n1. Operation Name: Operation I\n2. Duration: 90 minutes\n3. Invoicing: Yes, with Hourly Rate of 200\n4. Methods of Consultation: TEL, WEB, LOC",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation I', with invoicing enabled, an hourly rate of 200, and a duration of 90 minutes. The methods of consultation will be TEL, WEB, and LOC.",
            "response": "{\n  \"name\": \"Operation I\",\n  \"description\": \"\",\n  \"duration\": 90,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 200,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"TEL\",\n    \"WEB\",\n    \"LOC\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I want to create an operation named 'Operation J' with no invoicing, a unit price of 150, and a duration of 120 minutes. The consultation method will be TEL.",
            "response": "Here are the details for 'Operation J':\n1. Operation Name: Operation J\n2. Duration: 120 minutes\n3. Invoicing: No\n4. Unit Price: 150\n5. Method of Consultation: TEL",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation J' with no invoicing, a unit price of 150, and a duration of 120 minutes. The consultation method will be TEL.",
            "response": "{\n  \"name\": \"Operation J\",\n  \"description\": \"\",\n  \"duration\": 120,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": 150,\n  \"operationRateType\": 2,\n  \"methodsOfConsult\": [\n    \"TEL\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I want to create an operation named 'Operation K', with invoicing enabled, an hourly rate of 250, and a duration of 60 minutes. The consultation method will be WEB.",
            "response": "Here are the details for 'Operation K':\n1. Operation Name: Operation K\n2. Duration: 60 minutes\n3. Invoicing: Yes, with Hourly Rate of 250\n4. Method of Consultation: WEB",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation K', with invoicing enabled, an hourly rate of 250, and a duration of 60 minutes. The consultation method will be WEB.",
            "response": "{\n  \"name\": \"Operation K\",\n  \"description\": \"\",\n  \"duration\": 60,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 250,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"WEB\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I want to create an operation named 'Operation L', with invoicing disabled, a unit price of 120, and a duration of 45 minutes. The method of consultation will be LOC.",
            "response": "Here are the details for 'Operation L':\n1. Operation Name: Operation L\n2. Duration: 45 minutes\n3. Invoicing: No\n4. Unit Price: 120\n5. Method of Consultation: LOC",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation L', with invoicing disabled, a unit price of 120, and a duration of 45 minutes. The method of consultation will be LOC.",
            "response": "{\n  \"name\": \"Operation L\",\n  \"description\": \"\",\n  \"duration\": 45,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": 120,\n  \"operationRateType\": 2,\n  \"methodsOfConsult\": [\n    \"LOC\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "I want to create an operation named 'Operation M', with invoicing enabled, an hourly rate of 180, and a duration of 90 minutes. The consultation methods will be TEL and WEB.",
            "response": "Here are the details for 'Operation M':\n1. Operation Name: Operation M\n2. Duration: 90 minutes\n3. Invoicing: Yes, with Hourly Rate of 180\n4. Methods of Consultation: TEL, WEB",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation M', with invoicing enabled, an hourly rate of 180, and a duration of 90 minutes. The consultation methods will be TEL and WEB.",
            "response": "{\n  \"name\": \"Operation M\",\n  \"description\": \"\",\n  \"duration\": 90,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 180,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"TEL\",\n    \"WEB\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie A', met facturering ingeschakeld, een uurtarief van 100 en een duur van 60 minuten. De methoden van consultatie zijn TEL en WEB.",
            "response": "Hier zijn de gegevens voor 'Operatie A':\n1. Naam van de operatie: Operatie A\n2. Duur: 60 minuten\n3. Facturering: Ja, met uurtarief van 100\n4. Methoden van consultatie: TEL, WEB",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie A', met facturering ingeschakeld, een uurtarief van 100 en een duur van 60 minuten. De methoden van consultatie zijn TEL en WEB.",
            "response": "{\n  \"name\": \"Operatie A\",\n  \"description\": \"\",\n  \"duration\": 60,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 100,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"TEL\",\n    \"WEB\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie B', met facturering uitgeschakeld, een eenheidsprijs van 50 en een duur van 30 minuten. De consultatiemethode is WEB.",
            "response": "Hier zijn de gegevens voor 'Operatie B':\n1. Naam van de operatie: Operatie B\n2. Duur: 30 minuten\n3. Facturering: Nee\n4. Eenheidsprijs: 50\n5. Methode van consultatie: WEB",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie B', met facturering uitgeschakeld, een eenheidsprijs van 50 en een duur van 30 minuten. De consultatiemethode is WEB.",
            "response": "{\n  \"name\": \"Operatie B\",\n  \"description\": \"\",\n  \"duration\": 30,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": 50,\n  \"operationRateType\": 2,\n  \"methodsOfConsult\": [\n    \"WEB\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie C', met facturering ingeschakeld, een uurtarief van 150 en een duur van 90 minuten. De methoden van consultatie zijn TEL, WEB en LOC.",
            "response": "Hier zijn de gegevens voor 'Operatie C':\n1. Naam van de operatie: Operatie C\n2. Duur: 90 minuten\n3. Facturering: Ja, met uurtarief van 150\n4. Methoden van consultatie: TEL, WEB, LOC",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie C', met facturering ingeschakeld, een uurtarief van 150 en een duur van 90 minuten. De methoden van consultatie zijn TEL, WEB en LOC.",
            "response": "{\n  \"name\": \"Operatie C\",\n  \"description\": \"\",\n  \"duration\": 90,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 150,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"TEL\",\n    \"WEB\",\n    \"LOC\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie D', met facturering uitgeschakeld, een eenheidsprijs van 80 en een duur van 120 minuten. De consultatiemethode is TEL.",
            "response": "Hier zijn de gegevens voor 'Operatie D':\n1. Naam van de operatie: Operatie D\n2. Duur: 120 minuten\n3. Facturering: Nee\n4. Eenheidsprijs: 80\n5. Methode van consultatie: TEL",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie D', met facturering uitgeschakeld, een eenheidsprijs van 80 en een duur van 120 minuten. De consultatiemethode is TEL.",
            "response": "{\n  \"name\": \"Operatie D\",\n  \"description\": \"\",\n  \"duration\": 120,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": 80,\n  \"operationRateType\": 2,\n  \"methodsOfConsult\": [\n    \"TEL\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie E', met facturering ingeschakeld, een uurtarief van 200 en een duur van 45 minuten. De methoden van consultatie zijn WEB.",
            "response": "Hier zijn de gegevens voor 'Operatie E':\n1. Naam van de operatie: Operatie E\n2. Duur: 45 minuten\n3. Facturering: Ja, met uurtarief van 200\n4. Methode van consultatie: WEB",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie E', met facturering ingeschakeld, een uurtarief van 200 en een duur van 45 minuten. De methoden van consultatie zijn WEB.",
            "response": "{\n  \"name\": \"Operatie E\",\n  \"description\": \"\",\n  \"duration\": 45,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 200,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"WEB\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie F', met facturering uitgeschakeld, een eenheidsprijs van 100 en een duur van 60 minuten. De methoden van consultatie zijn TEL, WEB en LOC.",
            "response": "Hier zijn de gegevens voor 'Operatie F':\n1. Naam van de operatie: Operatie F\n2. Duur: 60 minuten\n3. Facturering: Nee\n4. Eenheidsprijs: 100\n5. Methoden van consultatie: TEL, WEB, LOC",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie F', met facturering uitgeschakeld, een eenheidsprijs van 100 en een duur van 60 minuten. De methoden van consultatie zijn TEL, WEB en LOC.",
            "response": "{\n  \"name\": \"Operatie F\",\n  \"description\": \"\",\n  \"duration\": 60,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": 100,\n  \"operationRateType\": 2,\n  \"methodsOfConsult\": [\n    \"TEL\",\n    \"WEB\",\n    \"LOC\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie G', met facturering ingeschakeld, een uurtarief van 120 en een duur van 75 minuten. De methoden van consultatie zijn TEL en LOC.",
            "response": "Hier zijn de gegevens voor 'Operatie G':\n1. Naam van de operatie: Operatie G\n2. Duur: 75 minuten\n3. Facturering: Ja, met uurtarief van 120\n4. Methoden van consultatie: TEL, LOC",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie G', met facturering ingeschakeld, een uurtarief van 120 en een duur van 75 minuten. De methoden van consultatie zijn TEL en LOC.",
            "response": "{\n  \"name\": \"Operatie G\",\n  \"description\": \"\",\n  \"duration\": 75,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 120,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"TEL\",\n    \"LOC\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie H', met facturering uitgeschakeld, een eenheidsprijs van 200 en een duur van 45 minuten. De consultatiemethode is WEB.",
            "response": "Hier zijn de gegevens voor 'Operatie H':\n1. Naam van de operatie: Operatie H\n2. Duur: 45 minuten\n3. Facturering: Nee\n4. Eenheidsprijs: 200\n5. Methode van consultatie: WEB",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie H', met facturering uitgeschakeld, een eenheidsprijs van 200 en een duur van 45 minuten. De consultatiemethode is WEB.",
            "response": "{\n  \"name\": \"Operatie H\",\n  \"description\": \"\",\n  \"duration\": 45,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": 200,\n  \"operationRateType\": 2,\n  \"methodsOfConsult\": [\n    \"WEB\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie I', met facturering ingeschakeld, een uurtarief van 110 en een duur van 50 minuten. De methoden van consultatie zijn TEL, WEB, en LOC.",
            "response": "Hier zijn de gegevens voor 'Operatie I':\n1. Naam van de operatie: Operatie I\n2. Duur: 50 minuten\n3. Facturering: Ja, met uurtarief van 110\n4. Methoden van consultatie: TEL, WEB, LOC",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie I', met facturering ingeschakeld, een uurtarief van 110 en een duur van 50 minuten. De methoden van consultatie zijn TEL, WEB, en LOC.",
            "response": "{\n  \"name\": \"Operatie I\",\n  \"description\": \"\",\n  \"duration\": 50,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 110,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"TEL\",\n    \"WEB\",\n    \"LOC\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie J', met facturering uitgeschakeld, een eenheidsprijs van 75 en een duur van 100 minuten. De consultatiemethode is TEL en WEB.",
            "response": "Hier zijn de gegevens voor 'Operatie J':\n1. Naam van de operatie: Operatie J\n2. Duur: 100 minuten\n3. Facturering: Nee\n4. Eenheidsprijs: 75\n5. Methoden van consultatie: TEL, WEB",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie J', met facturering uitgeschakeld, een eenheidsprijs van 75 en een duur van 100 minuten. De consultatiemethode is TEL en WEB.",
            "response": "{\n  \"name\": \"Operatie J\",\n  \"description\": \"\",\n  \"duration\": 100,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": 75,\n  \"operationRateType\": 2,\n  \"methodsOfConsult\": [\n    \"TEL\",\n    \"WEB\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie K', met facturering ingeschakeld, een uurtarief van 90 en een duur van 80 minuten. De methoden van consultatie zijn LOC.",
            "response": "Hier zijn de gegevens voor 'Operatie K':\n1. Naam van de operatie: Operatie K\n2. Duur: 80 minuten\n3. Facturering: Ja, met uurtarief van 90\n4. Methode van consultatie: LOC",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie K', met facturering ingeschakeld, een uurtarief van 90 en een duur van 80 minuten. De methoden van consultatie zijn LOC.",
            "response": "{\n  \"name\": \"Operatie K\",\n  \"description\": \"\",\n  \"duration\": 80,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 90,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"LOC\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie L', met facturering uitgeschakeld, een eenheidsprijs van 50 en een duur van 40 minuten. De consultatiemethode is WEB en TEL.",
            "response": "Hier zijn de gegevens voor 'Operatie L':\n1. Naam van de operatie: Operatie L\n2. Duur: 40 minuten\n3. Facturering: Nee\n4. Eenheidsprijs: 50\n5. Methoden van consultatie: WEB, TEL",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie L', met facturering uitgeschakeld, een eenheidsprijs van 50 en een duur van 40 minuten. De consultatiemethode is WEB en TEL.",
            "response": "{\n  \"name\": \"Operatie L\",\n  \"description\": \"\",\n  \"duration\": 40,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 0,\n  \"invoicing\": False,\n  \"hourlyRate\": 0,\n  \"unitPrice\": 50,\n  \"operationRateType\": 2,\n  \"methodsOfConsult\": [\n    \"WEB\",\n    \"TEL\"\n  ],\n  \"wizard\": None\n}",
        },
        {
            "role": "user",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie M', met facturering ingeschakeld, een uurtarief van 140 en een duur van 100 minuten. De methoden van consultatie zijn LOC, TEL, en WEB.",
            "response": "Hier zijn de gegevens voor 'Operatie M':\n1. Naam van de operatie: Operatie M\n2. Duur: 100 minuten\n3. Facturering: Ja, met uurtarief van 140\n4. Methoden van consultatie: LOC, TEL, WEB",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie M', met facturering ingeschakeld, een uurtarief van 140 en een duur van 100 minuten. De methoden van consultatie zijn LOC, TEL, en WEB.",
            "response": "{\n  \"name\": \"Operatie M\",\n  \"description\": \"\",\n  \"duration\": 100,\n  \"forAppointment\": True"
            ",\n  \"vatRate\": 6,\n  \"invoicing\": True"
            ",\n  \"hourlyRate\": 140,\n  \"unitPrice\": None,\n  \"operationRateType\": 1,\n  \"methodsOfConsult\": [\n    \"LOC\",\n    \"TEL\",\n    \"WEB\"\n  ],\n  \"wizard\": None\n}",
        },
    ]
    phx_operation_validation_data = [
        {
            "role": "assistant",
            "prompt": {
                "name": "test",
                "description": "test",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test1",
                "description": "test1",
                "duration": 10,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 120,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"},
                    {"shortCode": "WEB", "name": "Web"},
                    {"shortCode": "LOC", "name": "Location"},
                ],
                "uuid": "test-uuid-122",
                "wizard": "683404b3-a2c4-46db-b3b0-652c6f6b251f",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test2",
                "description": "test2",
                "duration": 15,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": True,
                "hourlyRate": 100,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-123",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test3",
                "description": "test3",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 2,
                "methodsOfConsult": [{"shortCode": "WEB", "name": "Web"}],
                "uuid": "test-uuid-124",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test4",
                "description": "test4",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [],
                "uuid": "test-uuid-125",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC).",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test5",
                "description": "test5",
                "duration": 30,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 150,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-126",
                "wizard": "test-wizard-1",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test6",
                "description": "test6",
                "duration": 60,
                "forAppointment": False,
                "vatRate": 21,
                "invoicing": False,
                "hourlyRate": 200,
                "unitPrice": 0,
                "operationRateType": 2,
                "methodsOfConsult": [{"shortCode": "WEB", "name": "Web"}],
                "uuid": "test-uuid-127",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test7",
                "description": "test7",
                "duration": 45,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 180,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [{"shortCode": "LOC", "name": "Location"}],
                "uuid": "test-uuid-128",
                "wizard": "test-wizard-2",
            },
            "response": "True" "",
        },
        # Invalid cases (missing or incorrect fields)
        {
            "role": "assistant",
            "prompt": {
                "name": "test8",
                "description": "test8",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [],
                "uuid": "test-uuid-129",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC).",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test9",
                "description": "test9",
                "duration": -10,  # Invalid duration
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 100,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-130",
                "wizard": "test-wizard-3",
            },
            "response": "False" "\nERROR: Duration must be a positive value.",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test11",
                "description": "test11",
                "duration": 15,
                "forAppointment": False,
                "vatRate": 0,
                "invoicing": True,
                "hourlyRate": 100,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-132",
                "wizard": None,
            },
            "response": "False" "\nERROR: forAppointment must be True" ".",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test12",
                "description": "test12",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 200,
                "unitPrice": 0,
                "operationRateType": 3,
                "methodsOfConsult": [{"shortCode": "WEB", "name": "Web"}],
                "uuid": "test-uuid-133",
                "wizard": "test-wizard-4",
            },
            "response": "False"
            "\nERROR: operationRateType must be either 1 or 2.",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "",
                "description": "test13",
                "duration": 10,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 120,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-134",
                "wizard": None,
            },
            "response": "False" "\nERROR: Name is required.",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test14",
                "description": "test14",
                "duration": 20,
                "forAppointment": True,
                "vatRate": -5,
                "invoicing": True,
                "hourlyRate": 130,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [{"shortCode": "LOC", "name": "Location"}],
                "uuid": "test-uuid-135",
                "wizard": "test-wizard-5",
            },
            "response": "False" "\nERROR: vatRate must be a positive value.",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test15",
                "description": "test15",
                "duration": 40,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-136",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test16",
                "description": "test16",
                "duration": 60,
                "forAppointment": True,
                "vatRate": 21,
                "invoicing": True,
                "hourlyRate": 220,
                "unitPrice": 0,
                "operationRateType": 2,
                "methodsOfConsult": [{"shortCode": "WEB", "name": "Web"}],
                "uuid": "test-uuid-137",
                "wizard": "test-wizard-6",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test17",
                "description": "test17",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [{"shortCode": "LOC", "name": "Location"}],
                "uuid": "test-uuid-138",
                "wizard": "test-wizard-7",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test18",
                "description": "test18",
                "duration": 25,
                "forAppointment": False,
                "vatRate": 0,
                "invoicing": False,
                "hourlyRate": 100,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-139",
                "wizard": "test-wizard-8",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test19",
                "description": "test19",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [],
                "uuid": "test-uuid-140",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC).",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test20",
                "description": "test20",
                "duration": 50,
                "forAppointment": True,
                "vatRate": 21,
                "invoicing": True,
                "hourlyRate": 300,
                "unitPrice": 0,
                "operationRateType": 2,
                "methodsOfConsult": [{"shortCode": "WEB", "name": "Web"}],
                "uuid": "test-uuid-141",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test21",
                "description": "test21",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 120,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-142",
                "wizard": "test-wizard-9",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test22",
                "description": "test22",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": False,
                "hourlyRate": 150,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [{"shortCode": "LOC", "name": "Location"}],
                "uuid": "test-uuid-143",
                "wizard": "test-wizard-10",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test23",
                "description": "test23",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 2,
                "methodsOfConsult": [{"shortCode": "WEB", "name": "Web"}],
                "uuid": "test-uuid-144",
                "wizard": "test-wizard-11",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test24",
                "description": "test24",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [{"shortCode": "WEB", "name": "Web"}],
                "uuid": "test-uuid-145",
                "wizard": "test-wizard-12",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test25",
                "description": "test25",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-146",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test26",
                "description": "test26",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [],
                "uuid": "test-uuid-147",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC).",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test27",
                "description": "test27",
                "duration": 15,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": True,
                "hourlyRate": 100,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [{"shortCode": "WEB", "name": "Web"}],
                "uuid": "test-uuid-148",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test28",
                "description": "test28",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-149",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test29",
                "description": "test29",
                "duration": 10,
                "forAppointment": True,
                "vatRate": 21,
                "invoicing": True,
                "hourlyRate": 150,
                "unitPrice": 0,
                "operationRateType": 2,
                "methodsOfConsult": [],
                "uuid": "test-uuid-150",
                "wizard": "test-wizard-13",
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC).",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test30",
                "description": "test30",
                "duration": 30,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 200,
                "operationRateType": 1,
                "methodsOfConsult": [{"shortCode": "LOC", "name": "Location"}],
                "uuid": "test-uuid-151",
                "wizard": "test-wizard-14",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test31",
                "description": "test31",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 100,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [{"shortCode": "WEB", "name": "Web"}],
                "uuid": "test-uuid-152",
                "wizard": "test-wizard-15",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test32",
                "description": "test32",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-153",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test33",
                "description": "test33",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 120,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [{"shortCode": "LOC", "name": "Location"}],
                "uuid": "test-uuid-154",
                "wizard": "test-wizard-16",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test34",
                "description": "test34",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-155",
                "wizard": "test-wizard-17",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test35",
                "description": "test35",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 21,
                "invoicing": True,
                "hourlyRate": 250,
                "unitPrice": 0,
                "operationRateType": 2,
                "methodsOfConsult": [],
                "uuid": "test-uuid-156",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC).",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test36",
                "description": "test36",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 200,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [],
                "uuid": "test-uuid-157",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC).",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test37",
                "description": "test37",
                "duration": 25,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 100,
                "operationRateType": 2,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-158",
                "wizard": "test-wizard-18",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test38",
                "description": "test38",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 50,
                "operationRateType": 1,
                "methodsOfConsult": [{"shortCode": "WEB", "name": "Web"}],
                "uuid": "test-uuid-159",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test39",
                "description": "test39",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 150,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [],
                "uuid": "test-uuid-160",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC).",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test40",
                "description": "test40",
                "duration": 30,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": True,
                "hourlyRate": 100,
                "unitPrice": 0,
                "operationRateType": 2,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"},
                    {"shortCode": "WEB", "name": "Web"},
                ],
                "uuid": "test-uuid-161",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test41",
                "description": "test41",
                "duration": 5,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": True,
                "hourlyRate": 75,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [{"shortCode": "LOC", "name": "Location"}],
                "uuid": "test-uuid-162",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test42",
                "description": "test42",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 21,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 2,
                "methodsOfConsult": [],
                "uuid": "test-uuid-163",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC).",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test43",
                "description": "test43",
                "duration": 10,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 90,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-164",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test44",
                "description": "test44",
                "duration": 0,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [{"shortCode": "WEB", "name": "Web"}],
                "uuid": "test-uuid-165",
                "wizard": "test-wizard-19",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test45",
                "description": "test45",
                "duration": 15,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": False,
                "hourlyRate": 100,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [],
                "uuid": "test-uuid-166",
                "wizard": "test-wizard-20",
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC).",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test46",
                "description": "test46",
                "duration": 30,
                "forAppointment": True,
                "vatRate": 21,
                "invoicing": True,
                "hourlyRate": 120,
                "unitPrice": 0,
                "operationRateType": 2,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-167",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test47",
                "description": "test47",
                "duration": 20,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": False,
                "hourlyRate": 50,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [],
                "uuid": "test-uuid-168",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC).",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test48",
                "description": "test48",
                "duration": 45,
                "forAppointment": False,
                "vatRate": 0,
                "invoicing": True,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": None,
                "methodsOfConsult": [{"shortCode": "LOC", "name": "Location"}],
                "uuid": "test-uuid-169",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: operationRateType is required when invoicing is True"
            ".",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test49",
                "description": "test49",
                "duration": 10,
                "forAppointment": True,
                "vatRate": None,
                "invoicing": True,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [{"shortCode": "WEB", "name": "Web"}],
                "uuid": "test-uuid-170",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: vatRate is required when invoicing is True"
            ".",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "",
                "description": "test50",
                "duration": 25,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 0,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-171",
                "wizard": None,
            },
            "response": "False" "\nERROR: Name is required.",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test51",
                "description": "test51",
                "duration": 30,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": None,
                "unitPrice": 0,
                "operationRateType": None,
                "methodsOfConsult": [],
                "uuid": "test-uuid-172",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC). ERROR: operationRateType is required when invoicing is True"
            ".",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test52",
                "description": "test52",
                "duration": 15,
                "forAppointment": False,
                "vatRate": None,
                "invoicing": True,
                "hourlyRate": None,
                "unitPrice": 0,
                "operationRateType": None,
                "methodsOfConsult": [{"shortCode": "LOC", "name": "Location"}],
                "uuid": "test-uuid-173",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: vatRate is required when invoicing is True"
            ". ERROR: operationRateType is required when invoicing is True"
            ".",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test53",
                "description": "test53",
                "duration": 50,
                "forAppointment": True,
                "vatRate": 21,
                "invoicing": True,
                "hourlyRate": 100,
                "unitPrice": None,
                "operationRateType": 2,
                "methodsOfConsult": [],
                "uuid": "test-uuid-174",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC).",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test54",
                "description": "test54",
                "duration": 60,
                "forAppointment": True,
                "vatRate": 21,
                "invoicing": True,
                "hourlyRate": 80,
                "unitPrice": 100,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"},
                    {"shortCode": "WEB", "name": "Web"},
                ],
                "uuid": "test-uuid-175",
                "wizard": None,
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test55",
                "description": "test55",
                "duration": 90,
                "forAppointment": False,
                "vatRate": 0,
                "invoicing": False,
                "hourlyRate": 0,
                "unitPrice": 150,
                "operationRateType": None,
                "methodsOfConsult": [{"shortCode": "LOC", "name": "Location"}],
                "uuid": "test-uuid-176",
                "wizard": "test-wizard-30",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test56",
                "description": "",
                "duration": 45,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": True,
                "hourlyRate": None,
                "unitPrice": 200,
                "operationRateType": None,
                "methodsOfConsult": [{"shortCode": "WEB", "name": "Web"}],
                "uuid": "test-uuid-177",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: operationRateType is required when invoicing is True"
            ".",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test57",
                "description": None,
                "duration": 30,
                "forAppointment": False,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 70,
                "unitPrice": 300,
                "operationRateType": 2,
                "methodsOfConsult": [],
                "uuid": "test-uuid-178",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC).",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test58",
                "description": "test58",
                "duration": 25,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": False,
                "hourlyRate": None,
                "unitPrice": 0,
                "operationRateType": None,
                "methodsOfConsult": [
                    {"shortCode": "TEL", "name": "Telephone"}
                ],
                "uuid": "test-uuid-179",
                "wizard": "test-wizard-32",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test59",
                "description": "test59",
                "duration": 40,
                "forAppointment": True,
                "vatRate": 21,
                "invoicing": True,
                "hourlyRate": 50,
                "unitPrice": None,
                "operationRateType": None,
                "methodsOfConsult": [],
                "uuid": "test-uuid-180",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: Methods of consultation are required (TEL, WEB, LOC). ERROR: operationRateType is required when invoicing is True"
            ".",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test60",
                "description": None,
                "duration": 15,
                "forAppointment": True,
                "vatRate": None,
                "invoicing": True,
                "hourlyRate": 90,
                "unitPrice": 150,
                "operationRateType": None,
                "methodsOfConsult": [{"shortCode": "LOC", "name": "Location"}],
                "uuid": "test-uuid-181",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: vatRate is required when invoicing is True"
            ". ERROR: operationRateType is required when invoicing is True"
            ".",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test61",
                "description": "test61",
                "duration": 50,
                "forAppointment": True,
                "vatRate": 6,
                "invoicing": True,
                "hourlyRate": 85,
                "unitPrice": 200,
                "operationRateType": 1,
                "methodsOfConsult": [
                    {"shortCode": "WEB", "name": "Web"},
                    {"shortCode": "LOC", "name": "Location"},
                ],
                "uuid": "test-uuid-182",
                "wizard": "test-wizard-33",
            },
            "response": "True" "",
        },
        {
            "role": "assistant",
            "prompt": {
                "name": "test62",
                "description": "",
                "duration": 35,
                "forAppointment": True,
                "vatRate": 0,
                "invoicing": False,
                "hourlyRate": None,
                "unitPrice": None,
                "operationRateType": None,
                "methodsOfConsult": [],
                "uuid": "test-uuid-183",
                "wizard": None,
            },
            "response": "False"
            "\nERROR: Name is required. ERROR: Methods of consultation are required (TEL, WEB, LOC).",
        },
    ]
    phx_instructions_data = [
        {
            "role": "assistant",
            "prompt": "Operation name is 'Operation A'. The description is 'Consulting services'. The duration is 30 minutes. Invoicing will be applied with an hourly rate of 50. The method of consultation is TEL.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Operation name is 'Operation A'. The description is 'Consulting services'. The duration is 30 minutes. Invoicing will be applied with an hourly rate of 50. The method of consultation is TEL.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation B', with no invoicing. It will have a unit price of 100 and last 60 minutes. The consultation method will be WEB.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I would like to create an operation named 'Operation C' with invoicing. It will be 45 minutes long with an hourly rate of 70. Methods of consultation will be TEL and LOC.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create a new operation.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation A', lasting 45 minutes, with an hourly rate of 50 and invoicing enabled. The consultation method will be TEL.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation B', with no invoicing, a unit price of 100, and lasting 60 minutes. The consultation method will be WEB.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation named 'Operation C' with invoicing enabled, an hourly rate of 70, and a duration of 45 minutes. The consultation methods will be TEL and LOC.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I would like to create an operation named 'Operation D' with a unit price of 200, lasting 30 minutes. The method of consultation will be WEB.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create a new operation.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Please provide the details to create a new operation.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create a new operation. But I don't have the invoice description yet.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I have the operation name, duration, and invoicing details, but no methods of consultation.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create a new operation with the operation name, invoicing applied, and hourly rate.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create a new operation but I don’t have the hourly rate.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create a new operation but I'm unsure about the invoicing.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create a new operation. The name is 'Customer Support Call'.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Create a new operation named 'Marketing Consultation' with invoicing applied and unit price 100.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create a new operation called 'Product Demo' with invoicing not applied and unit price 200.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Create operation 'Financial Review' with invoicing yes, hourly rate 150.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create a new operation 'Sales Meeting' but I don't know the duration yet.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Please create a new operation 'Team Workshop' with invoicing applied, hourly rate 120, and method of consultation WEB.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Create a new operation called 'Webinar' with no invoicing applied, unit price 50.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation called 'Team Training'.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I have the operation name, description, and hourly rate. But I don’t have the duration yet.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Please create an operation called 'Product Training'. I have the hourly rate and duration.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create a new operation, but I’m not sure about the method of consultation.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Can you create a new operation for a webinar? I know the operation name and invoicing details.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Create an operation 'Consulting Session' with invoicing applied. The hourly rate is 100.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create a new operation called 'Client Call' with no invoicing. I know the unit price.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I need a new operation called 'Strategy Meeting' with hourly rate 120. I don’t have the unit price.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Create an operation 'Team Review' where invoicing is not applied. The unit price is 200.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Create an operation for 'Client Onboarding'. I have the invoice description and method of consultation.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create a new operation for 'Product Consultation'. I know the duration and invoicing details.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Create a new operation called 'Branding Workshop' with invoicing applied.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "I want to create an operation called 'Market Research' with unit price 300.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Geef de gegevens voor het maken van een nieuwe operatie.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken, maar ik heb de factuuromschrijving nog niet.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Ik heb de operatienaam, duur en factureringsgegevens, maar geen methode van consultatie.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken met de operatienaam, facturering toegepast, en uurtarief.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken, maar ik heb het uurtarief niet.",
            "response": "Operating Instructions",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken, maar ik weet niet zeker of er gefactureerd zal worden.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken. De naam is 'Klantenondersteuning Oproep'.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Maak een nieuwe operatie genaamd 'Marketing Consultatie' met facturering toegepast en een stukprijs van 100.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Product Demo' met geen facturering toegepast en een stukprijs van 200.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Maak de operatie 'Financiële Beoordeling' met facturering ja, uurtarief 150.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie 'Sales Meeting' maken, maar ik weet de duur nog niet.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Maak een nieuwe operatie 'Team Workshop' met facturering toegepast, uurtarief 120 en methode van consultatie WEB.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Maak een nieuwe operatie genaamd 'Webinar' zonder facturering toegepast, stukprijs 50.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil de operatie 'Team Training' maken.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Ik heb de operatienaam, omschrijving en uurtarief. Maar ik heb de duur nog niet.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Maak een operatie genaamd 'Product Training'. Ik heb het uurtarief en de duur.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken, maar ik weet niet zeker over de methode van consultatie.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Kun je een nieuwe operatie maken voor een webinar? Ik weet de operatienaam en factureringdetails.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Maak een operatie 'Consultatiesessie' met facturering toegepast. Het uurtarief is 100.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Klantenoproep' zonder facturering. Ik weet de stukprijs.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Ik heb een nieuwe operatie nodig genaamd 'Strategiebespreking' met uurtarief 120. Ik heb de stukprijs niet.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Maak een operatie 'Teambeoordeling' waarbij geen facturering wordt toegepast. De stukprijs is 200.",
            "response": "Handle Operating",
        },
        {
            "role": "assistant",
            "prompt": "Ik wil een nieuwe operatie maken genaamd 'Operatie M', met facturering ingeschakeld, een uurtarief van 140 en een duur van 100 minuten. De methoden van consultatie zijn LOC, TEL, en WEB.",
            "response": "Handle Operating",
        },
        # default
        {
            "role": "assistant",
            "prompt": "Hi what is your name?",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "1+2+3+4+5+6",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Hello, how can you help me?",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Tell me a joke.",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "What is the weather today?",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Who was the first president of the United States?",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Translate 'Hello' to French.",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "List the planets in the solar system.",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "What is the square root of 64?",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "How many countries are in Europe?",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "What’s the capital of Japan?",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "2+2",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Define artificial intelligence.",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "What is your favorite color?",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Hallo, hoe kun je me helpen?",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Vertel me een grap.",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Wat is het weer vandaag?",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Wie was de eerste president van de Verenigde Staten?",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Vertaal 'Hallo' naar het Frans.",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Noem de planeten in het zonnestelsel.",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Wat is de vierkantswortel van 64?",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Hoeveel landen zijn er in Europa?",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Wat is de hoofdstad van Japan?",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "2+2",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Definieer kunstmatige intelligentie.",
            "response": "Default",
        },
        {
            "role": "assistant",
            "prompt": "Wat is je favoriete kleur?",
            "response": "Default",
        },
    ]
    # Instructions
    anonymize_instruction = "Anonymize the data"
    qa_instruction = "Answer the question"
    personal_data_instruction = (
        "Return True"
        " or False. Note that the data may contain inaccuracies in the response"
    )
    phx_operation_instruction = "Handle Operating"
    phx_instructions_instruction = "Return the most suitable instruction"
    phx_operation_validation_instruction = (
        "Validate the operation JSON. Return 'True"
        "' if all required fields are present and valid, and 'False' if any fields are missing or invalid."
    )
    # Initialize preparer and add data entries
    preparer = DatasetPreparer()
    preparer.add_data(anonymize_instruction, anonymize_data)
    preparer.add_data(qa_instruction, qa_data)
    preparer.add_data(personal_data_instruction, personal_data)
    preparer.add_data(phx_operation_instruction, phx_operation_data)
    preparer.add_data(phx_instructions_instruction, phx_instructions_data)
    preparer.add_data(
        phx_operation_validation_instruction, phx_operation_validation_data
    )

    # Save the dataset to a JSONL file
    json_file_path = "blueVi-GPT-dataset.json"
    preparer.save_to_jsonl(json_file_path)

    repo_name = "ThanhTranVisma/blueVi-GPT-dataset"
    create_repo(repo_name, repo_type="dataset", private=False)
    # Push JSON file to Hugging Face Hub
    api = HfApi()
    api.upload_file(
        path_or_fileobj=json_file_path,
        path_in_repo="blueVi-GPT-dataset.json",
        repo_id=repo_name,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"Dataset successfully uploaded to {repo_name}!")


if __name__ == "__main__":
    main()
