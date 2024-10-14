from datasets import Dataset
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()
hf_token = os.getenv("HUGGING_FACE_TOKEN")

# Sample data for anonymization (Anonymize the data)
data = [
   # English Cases
   {
        "prompt": "Allice Smith's email is allice.smith@example..com. Her BSN is 12345678. Her address is 15 Elm Street, and her zip code is 12345.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. Her address is [ADDRESS_1], and her zip code is [ZIP_1]."
    },
    {
        "prompt": "Robert Brown was born on 12-01-2024. His email is robert.brown@example.com, and his BSN is 87654321. He lives at 222 Ash St.",
        "response": "[NAME_1] was born on [DOB_1]. His email is [EMAIL_1], and his BSN is [BSN_1]. He lives at [ADDRESS_1]."
    },
    {
        "prompt": "Natalie Green's email is nat.green@example..com. Her birthday is 01/12/2024, and her postal code is 34567. She resides at 789 Birch Lane.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Her birthday is [DOB_1], and her postal code is [ZIP_1]. She resides at [ADDRESS_1]."
    },
    {
        "prompt": "Mark Thompson's email is mark.thompson@example.com. His BSN is 54321098, and his home address is 654 Maple Drive. His date of birth is 12/01/2024.",
        "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His home address is [ADDRESS_1]. His date of birth is [DOB_1]."
    },
    {
        "prompt": "Linda Johnson's email is linda.johnson@example..com, and her BSN is 11223344. She lives at 321 Willow Way, with a postal code of 98765.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1], with a postal code of [ZIP_1]."
    },
    {
        "prompt": "Peter's date of birth is 12/31/2024. His email is peter@example.com. His home address is 98 Pine Street, and his zip is 45678.",
        "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1]. His home address is [ADDRESS_1], and his zip is [ZIP_1]."
    },
    {
        "prompt": "Julia's email is julia@example.com, her BSN is 22334455. She was born on 01-12-2024, and she lives at 234 Oak Lane.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She was born on [DOB_1], and she lives at [ADDRESS_1]."
    },
    {
        "prompt": "Tom's email is tom123@example.com, and his BSN is 99988877. He was born on 12/01/2024 and lives at 100 Cedar Ave.",
        "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. He was born on [DOB_1] and lives at [ADDRESS_1]."
    },
    {
        "prompt": "My cousin Anna's email is anna123@example.com. Her BSN is 77788899. She lives at 77 Elm St. Her birthday is 1st January 2024.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1]. Her birthday is [DOB_1]."
    },
    {
        "prompt": "Max's date of birth is 2024-01-01. His email is max@example.com. His postal code is 12345, and he lives at 789 Cedar Street.",
        "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1]. His postal code is [ZIP_1], and he lives at [ADDRESS_1]."
    },
    {
        "prompt": "Sarah Davis's email is sarah.davis@example.com. She was born on 01/12/2024. Her address is 789 Maple Street, and her postal code is 87654.",
        "response": "[NAME_1]'s email is [EMAIL_1]. She was born on [DOB_1]. Her address is [ADDRESS_1], and her postal code is [ZIP_1]."
    },
    {
        "prompt": "Jacob Martin was born on 12/01/2024. His BSN is 34567890, and his email is jacob.martin@example.com. He lives at 321 Cedar Avenue.",
        "response": "[NAME_1] was born on [DOB_1]. His BSN is [BSN_1], and his email is [EMAIL_1]. He lives at [ADDRESS_1]."
    },
    {
        "prompt": "Emily White's email is emily.white@example.com. She was born on the 1st of December 2024. Her zip code is 54321, and her house number is 24.",
        "response": "[NAME_1]'s email is [EMAIL_1]. She was born on [DOB_1]. Her zip code is [ZIP_1], and her house number is [ADDRESS_1]."
    },
    {
        "prompt": "My friend Michael was born on December 1st, 2024. His email is michael@example.com, and his postal code is 98765. His home address is 456 Birch Avenue.",
        "response": "[NAME_1] was born on [DOB_1]. His email is [EMAIL_1], and his postal code is [ZIP_1]. His home address is [ADDRESS_1]."
    },
    {
        "prompt": "Lily Evans's birthday is 12/01/2024. Her email is lily.evans@example.com, and her BSN is 45678901. She lives at 101 Rosewood St.",
        "response": "[NAME_1]'s birthday is [DOB_1]. Her email is [EMAIL_1], and her BSN is [BSN_1]. She lives at [ADDRESS_1]."
    },
    {
        "prompt": "Contact John, who was born on 01/12/2024, at john.doe@example.com. His home address is 789 Oak Drive, and his postal code is 10234.",
        "response": "Contact [NAME_1], who was born on [DOB_1], at [EMAIL_1]. His home address is [ADDRESS_1], and his postal code is [ZIP_1]."
    },
    {
        "prompt": "Sophia Green's birthday is 1st December, 2024. Her BSN is 98765432, and her email is sophia.green@example.com. She lives at 888 Willow Lane.",
        "response": "[NAME_1]'s birthday is [DOB_1]. Her BSN is [BSN_1], and her email is [EMAIL_1]. She lives at [ADDRESS_1]."
    },
    {
        "prompt": "Ryan's date of birth is December 01, 2024. His zip code is 54321, and his email is ryan@example.com. His BSN is 23456789.",
        "response": "[NAME_1]'s date of birth is [DOB_1]. His zip code is [ZIP_1], and his email is [EMAIL_1]. His BSN is [BSN_1]."
    },
    {
        "prompt": "Jane Smith's birthday is 01/12/2024. She lives at 123 Palm Street, and her postal code is 56789. Her email is jane.smith@example.com.",
        "response": "[NAME_1]'s birthday is [DOB_1]. She lives at [ADDRESS_1], and her postal code is [ZIP_1]. Her email is [EMAIL_1]."
    },
    {
        "prompt": "Marcus Brown's email is marcus.brown@example.com, and his BSN is 12345678. His birthday is December 1st, 2024, and his home address is 456 Pine St.",
        "response": "[NAME_1]'s email is [EMAIL_1], and his BSN is [BSN_1]. His birthday is [DOB_1], and his home address is [ADDRESS_1]."
    },
    {
        "prompt": "Emily Carter was born on 01/12/2024. Her email is emily.carter@example.com, and her BSN is 98765432. She lives at 789 Spruce Lane.",
        "response": "[NAME_1] was born on [DOB_1]. Her email is [EMAIL_1], and her BSN is [BSN_1]. She lives at [ADDRESS_1]."
    },
    {
        "prompt": "John Doe's email is J.Simpson@@netwrix..com. His BSN is 12345678.9. His home address is 10 Langelo! His zip code is 7666mc. His Mastercard number is 5258-7041-08753590 and his visa number is 4563 7568 5698 4587. His iban number is nl91abna0417164300. His date of birth is 1/1/90. His IP address is 192.168.1.1.",
        "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His home address is [ADDRESS_1]. His ZIP code is [ZIP_1]. His MasterCard number is [MASTERCARD_1] and his Visa number is [VISA_1]. His IBAN number is [IBAN_1]. His date of birth is [DOB_1]. His IP address is [IP_ADDRESS_1]."
    },
    {
        "prompt": "Olivia Doe's email is d.olivia@netwrix..com. Her BSN is 98765432. Her home address is 60 Langelo Langelo. Her ZIP code is 7612-MC. Her Mastercard number is 5258-7041-0873577 and her visa number is 4511-7500-0000-0000. Her iban number is DE91abna0417164300. Her dob is 1/1/91. Her ip is 192.168.2.2.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. Her home address is [ADDRESS_1]. Her ZIP code is [ZIP_1]. Her MasterCard number is [MASTERCARD_1] and her Visa number is [VISA_1]. Her IBAN number is [IBAN_1]. Her date of birth is [DOB_1]. Her IP address is [IP_ADDRESS_1]."
    },
    {
        "prompt": "John and Olivia went to the market. John's email is john.doe@example.com, and Olivia's BSN is 98765432. Their addresses are 123 Main St and 456 Oak St. The zip codes are 98765 and 12345. John's Mastercard number is 1234-5678-9876-5432.",
        "response": "[NAME_1] and [NAME_2] went to the market. [NAME_1]'s email is [EMAIL_1], and [NAME_2]'s BSN is [BSN_1]. Their addresses are [ADDRESS_1] and [ADDRESS_2]. The zip codes are [ZIP_1] and [ZIP_2]. [NAME_1]'s MasterCard number is [MASTERCARD_1]."
    },
    {
        "prompt": "Contact me at invalid@domain..com, my BSN is 123-45-678, and my IP is 192.168.0.2.",
        "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His IP address is [IP_ADDRESS_1]."
    },
    {
        "prompt": "My friend Emily has an email emily@example.com and her BSN is 23456789. She lives at 789 Pine St. Her zip code is 67890.",
        "response": "[NAME_1] has an email [EMAIL_1] and her BSN is [BSN_1]. She lives at [ADDRESS_1]. Her ZIP code is [ZIP_1]."
    },
    {
        "prompt": "The company contacted me through my email address: john@example.com. I have a bank account number: 1234567890123456.",
        "response": "[NAME_1]'s email is [EMAIL_1]. His bank account number is [BANK_ACCOUNT_1]."
    },

    # Dutch Cases
    {
        "prompt": "Jan Jansen's e-mailadres is jan.jansen@@voorbeeld.nl. Zijn BSN is 12345678.9. Zijn huisadres is 10 Hoofdstraat! Zijn postcode is 7666mc. Zijn Mastercardnummer is 5258-7041-08753590 en zijn Visa-nummer is 4563 7568 5698 4587. Zijn IBAN-nummer is NL91ABNA0417164300. Zijn geboortedatum is 1/1/90. Zijn IP-adres is 192.168.1.1.",
        "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn huisadres is [ADDRESS_1]. Zijn postcode is [ZIP_1]. Zijn Mastercardnummer is [MASTERCARD_1] en zijn Visa-nummer is [VISA_1]. Zijn IBAN-nummer is [IBAN_1]. Zijn geboortedatum is [DOB_1]. Zijn IP-adres is [IP_ADDRESS_1]."
    },
    {
        "prompt": "Olivia Jansen's e-mailadres is d.olivia@voorbeeld..nl. Haar BSN is 98765432. Haar huisadres is 60 Langelaan. Haar postcode is 7612-MC. Haar Mastercardnummer is 5258-7041-0873577 en haar Visa-nummer is 4511-7500-0000-0000. Haar IBAN-nummer is DE91ABNA0417164300. Haar geboortedatum is 1/1/91. Haar IP is 192.168.2.2.",
        "response": "[NAME_2]'s e-mailadres is [EMAIL_2]. Haar BSN is [BSN_2]. Haar huisadres is [ADDRESS_2]. Haar postcode is [ZIP_2]. Haar Mastercardnummer is [MASTERCARD_2] en haar Visa-nummer is [VISA_2]. Haar IBAN-nummer is [IBAN_2]. Haar geboortedatum is [DOB_2]. Haar IP-adres is [IP_ADDRESS_2]."
    },
    {
        "prompt": "Jan en Olivia waren in het park. Jan's e-mailadres is jan.doe@voorbeeld.com, en Olivia's BSN is 98765432. Hun adressen zijn 123 Hoofdstraat en 456 Eiklaan. De postcodes zijn 98765 en 12345. Jan's Mastercardnummer is 1234-5678-9876-5432.",
        "response": "[NAME_1] en [NAME_2] waren in het park. [NAME_1]'s e-mailadres is [EMAIL_1], en [NAME_2]'s BSN is [BSN_1]. Hun adressen zijn [ADDRESS_1] en [ADDRESS_2]. De postcodes zijn [ZIP_1] en [ZIP_2]. [NAME_1]'s Mastercardnummer is [MASTERCARD_1]."
    },
    {
        "prompt": "Neem contact met me op via jan.doe@gmail..com, mijn BSN is 123-45-678, en mijn IP is 192.168.0.2.",
        "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn IP-adres is [IP_ADDRESS_1]."
    },
    {
        "prompt": "Sophie werkt bij een bedrijf en haar e-mailadres is sophie@voorbeeld.com. Haar BSN is 23456789. Ze woont op 321 Laan van de Vrijheid. Haar postcode is 12345.",
        "response": "[NAME_1] werkt bij een bedrijf en haar e-mailadres is [EMAIL_1]. Haar BSN is [BSN_1]. Ze woont op [ADDRESS_1]. Haar postcode is [ZIP_1]."
    },
     {
        "prompt": "De geboortedatum van Jan Jansen is 01/12/2024. Zijn leeftijd is 25 jaar.",
        "response": "De geboortedatum van [NAME_1] is [DOB_1]. Zijn leeftijd is [AGE_1] jaar."
    },
    {
        "prompt": "Olivia Jansen werd geboren op 12/01/2024 en is nu 30 jaar oud.",
        "response": "[NAME_1] werd geboren op [DOB_1] en is nu [AGE_1] jaar oud."
    },
    {
        "prompt": "Sophie is geboren op 1 december 2024 en ze is 45 jaar.",
        "response": "[NAME_1] is geboren op [DOB_1] en ze is [AGE_1] jaar."
    },
    {
        "prompt": "Karel werd geboren op de eerste december 2024. Hij is nu 40 jaar.",
        "response": "[NAME_1] werd geboren op [DOB_1]. Hij is nu [AGE_1] jaar."
    },
    {
        "prompt": "De geboortedatum van Maria is 2024-12-01. Ze is nu 50 jaar oud.",
        "response": "De geboortedatum van [NAME_1] is [DOB_1]. Ze is nu [AGE_1] jaar oud."
    },
    {
        "prompt": "Haar geboortedatum is 12 januari 2024, en ze is 33 jaar.",
        "response": "Haar geboortedatum is [DOB_1], en ze is [AGE_1] jaar."
    },
    {
        "prompt": "De geboortedatum van mijn vriend is 01-12-2024. Hij is nu 35 jaar.",
        "response": "De geboortedatum van [NAME_1] is [DOB_1]. Hij is nu [AGE_1] jaar."
    },
    {
        "prompt": "Mijn broer is geboren op 1-12-2024 en hij is nu 27 jaar.",
        "response": "Mijn broer is geboren op [DOB_1] en hij is nu [AGE_1] jaar."
    },
    {
        "prompt": "Ze is op 1e december 2024 geboren, en ze is 29 jaar oud.",
        "response": "Ze is op [DOB_1] geboren, en ze is [AGE_1] jaar oud."
    },
    {
        "prompt": "Zijn geboortedatum is 12 januari 2024. Hij is nu 37 jaar.",
        "response": "Zijn geboortedatum is [DOB_1]. Hij is nu [AGE_1] jaar."
    },
    {
        "prompt": "Ik heb een nieuw emailadres: mijn.email@voorbeeld.nl en mijn BSN is 34567890. Mijn huisadres is 88 Nieuweweg.",
        "response": "[NAME_1] heeft een nieuw e-mailadres: [EMAIL_1] en haar BSN is [BSN_1]. Haar huisadres is [ADDRESS_1]."
    },
    {
        "prompt": "Sara de Vries's e-mailadres is sara.de.vries@voorbeeld.nl. Haar BSN is 12345678. Haar huisadres is 12 Fijnstraat, en haar postcode is 98765.",
        "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Haar BSN is [BSN_1]. Haar huisadres is [ADDRESS_1], en haar postcode is [ZIP_1]."
    },
    {
        "prompt": "De geboortedatum van Piet is 01-12-2024. Zijn e-mailadres is piet@example.nl, en zijn BSN is 87654321. Hij woont op 456 Laan van de Vrijheid.",
        "response": "De geboortedatum van [NAME_1] is [DOB_1]. Zijn e-mailadres is [EMAIL_1], en zijn BSN is [BSN_1]. Hij woont op [ADDRESS_1]."
    },
    {
        "prompt": "Karin's e-mailadres is karin@example.nl, en haar BSN is 22223333. Haar geboortedatum is 12/01/2024, en ze woont op 111 Eikenlaan.",
        "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Haar BSN is [BSN_1]. Haar geboortedatum is [DOB_1], en ze woont op [ADDRESS_1]."
    },
    {
        "prompt": "Rob Jansen's e-mailadres is rob.jansen@voorbeeld.nl. Zijn BSN is 11112222, en zijn huisadres is 88 Hoofdstraat. Zijn geboortedatum is 01-12-2024.",
        "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn huisadres is [ADDRESS_1]. Zijn geboortedatum is [DOB_1]."
    },
    {
        "prompt": "Annelies heeft een nieuw emailadres: annelies@voorbeeld.nl. Haar BSN is 99998888, en ze woont op 24 Dennenlaan. Haar postcode is 12345.",
        "response": "[NAME_1] heeft een nieuw e-mailadres: [EMAIL_1]. Haar BSN is [BSN_1]. Ze woont op [ADDRESS_1]. Haar postcode is [ZIP_1]."
    },
    {
        "prompt": "Janine werd geboren op 12/01/2024 en ze is nu 26 jaar oud. Haar email is janine@example.nl.",
        "response": "[NAME_1] werd geboren op [DOB_1] en ze is nu [AGE_1] jaar oud. Haar email is [EMAIL_1]."
    },
    {
        "prompt": "Sophie werkt bij een bedrijf. Haar e-mailadres is sophie@voorbeeld.nl, en haar BSN is 34567890. Ze woont op 555 Klaproosstraat.",
        "response": "[NAME_1] werkt bij een bedrijf. Haar e-mailadres is [EMAIL_1]. Haar BSN is [BSN_1]. Ze woont op [ADDRESS_1]."
    },
    {
        "prompt": "Frits heeft een nieuwe email: frits@example.nl. Zijn BSN is 77778888. Hij woont op 80 Rozenlaan.",
        "response": "[NAME_1] heeft een nieuwe email: [EMAIL_1]. Zijn BSN is [BSN_1]. Hij woont op [ADDRESS_1]."
    },
    {
        "prompt": "De geboortedatum van Lotte is 01-01-2024. Ze woont op 34 Bovenstraat. Haar e-mailadres is lotte@voorbeeld.nl.",
        "response": "De geboortedatum van [NAME_1] is [DOB_1]. Ze woont op [ADDRESS_1]. Haar e-mailadres is [EMAIL_1]."
    },
    {
        "prompt": "Milan heeft een emailadres: milan@example.nl. Zijn BSN is 55556666. Hij woont op 12 Appelstraat.",
        "response": "[NAME_1] heeft een emailadres: [EMAIL_1]. Zijn BSN is [BSN_1]. Hij woont op [ADDRESS_1]."
    },
    # English Cases with Typos and Incorrect Formats
    {
        "prompt": "Allice Smith's email is allice.smith@example..com. Her BSN is 12345678. Her home adress is 45 Elm St, and her postal code is 9021. Her MasterCard number is 5412-3456-7890-12345 and her IBAN number is GB29NWBK60161331926819.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. Her home address is [ADDRESS_1]. Her postal code is [ZIP_1]. Her MasterCard number is [MASTERCARD_1] and her IBAN number is [IBAN_1]."
    },
    {
        "prompt": "Mark Jonson has an email mark.j@exampl.com. His BSN is 98765432. His adress is 72 Oak Ave. His zip code is 54321.",
        "response": "[NAME_1] has an email [EMAIL_1]. His BSN is [BSN_1]. His address is [ADDRESS_1]. His ZIP code is [ZIP_1]."
    },
    {
        "prompt": "Email: test.email@invalid..com. BSN: 321-45-6789. IP address: 172.16.254.01.",
        "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His IP address is [IP_ADDRESS_1]."
    },
    {
        "prompt": "Contact me at my email address: not.a.valid.email@domain..com. My BSN is 123456789, and my IP is 192.168.0.256.",
        "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His IP address is [IP_ADDRESS_1]."
    },
    {
        "prompt": "There is a problem with my email, which is john.doe@exampl.com, and my BSN is 234-56-7890.",
        "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]."
    },
    {
        "prompt": "Johhn's birthdate is Dec 32, 2024. He is now 26 yrs old.",
        "response": "[NAME_1]'s birthdate is [DOB_1]. He is now [AGE_1] years old."
    },
    {
        "prompt": "Smantha was born on 15/13/2024, she's 34 year.",
        "response": "[NAME_1] was born on [DOB_1], she's [AGE_1] years old."
    },
    {
        "prompt": "Jke's date of brith is 2024-31-11 and he is 28 yo.",
        "response": "[NAME_1]'s date of birth is [DOB_1] and he is [AGE_1] years old."
    },
    {
        "prompt": "Mia's dob is 01//12/2024. She is 24.",
        "response": "[NAME_1]'s date of birth is [DOB_1]. She is [AGE_1] years old."
    },
    {
        "prompt": "His birthdate is 1/13/2024, he is now 35 year.",
        "response": "His birthdate is [DOB_1], he is now [AGE_1] years old."
    },
    {
        "prompt": "Born on Febuary 29, 2024, Jake is 33 years old.",
        "response": "Born on [DOB_1], [NAME_1] is [AGE_1] years old."
    },
    {
        "prompt": "Ben's birtdate: 2024/12/01. He is twenty five.",
        "response": "[NAME_1]'s birthdate is [DOB_1]. He is [AGE_1] years old."
    },
    {
        "prompt": "Anna, dob 12th Oct, 2024, age 32.",
        "response": "[NAME_1], dob [DOB_1], age [AGE_1]."
    },
    {
        "prompt": "Her DOB is 01-33-2024, and she's 22 yr.",
        "response": "Her DOB is [DOB_1], and she's [AGE_1] years old."
    },
    {
        "prompt": "His date of brth is 12/01/2024. Now he's 45 yo.",
        "response": "His date of birth is [DOB_1]. Now he's [AGE_1] years old."
    },

    # Dutch Cases with Typos and Incorrect Formats
    {
        "prompt": "Liesbeth heeft een e-mailadres: liesbeth@@voorbeeld.nl. Haar BSN is 2345678. Ze woont op 101 Beukenlaan, postcode 1234ABC.",
        "response": "[NAME_1] heeft een e-mailadres: [EMAIL_1]. Haar BSN is [BSN_1]. Ze woont op [ADDRESS_1], postcode [ZIP_1]."
    },
    {
        "prompt": "Pieter's email is pieter@exampl.nl. Zijn BSN is 34567890. Zijn huisadrees is 65 Lindelaan. Zijn postcode is 5678CD.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn huisadres is [ADDRESS_1]. Zijn postcode is [ZIP_1]."
    },
    {
        "prompt": "Neem contact met me op via mijn e-mailadres: info@bedrijf..nl. Mijn BSN is 456-78-901, en mijn IP is 192.168.0.300.",
        "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1]. Zijn IP-adres is [IP_ADDRESS_1]."
    },
    {
        "prompt": "Kunt u me e-mailen op mijn.new.email@voorbeeld..nl? Mijn BSN is 12345678.",
        "response": "[NAME_1]'s e-mailadres is [EMAIL_1]. Zijn BSN is [BSN_1]."
    },
    {
        "prompt": "Ik heb een nieuw emailadres: mijn.email@voorbeeld.nl. Mijn BSN is 34567890. Mijn huisadres is 88 Nieuweweg. En mijn postcod is 1010AB.",
        "response": "[NAME_1] heeft een nieuw e-mailadres: [EMAIL_1]. Haar BSN is [BSN_1]. Haar huisadres is [ADDRESS_1]. Zijn postcode is [ZIP_1]."
    },
     {
        "prompt": "De geboortedatum van Jan Jnansen is 01-32-2024. Zijn leeftijd is 25 jaar.",
        "response": "De geboortedatum van [NAME_1] is [DOB_1]. Zijn leeftijd is [AGE_1] jaar."
    },
    {
        "prompt": "Olivvia werd gebroen op 13/01/2024 en is nu 30 jaar oud.",
        "response": "[NAME_1] werd geboren op [DOB_1] en is nu [AGE_1] jaar oud."
    },
    {
        "prompt": "Sophe is gebon op 1-12-2024 en ze is 45 jar.",
        "response": "[NAME_1] is geboren op [DOB_1] en ze is [AGE_1] jaar."
    },
    {
        "prompt": "De geboortdatum van Maria is 2024-13-01. Ze is nu 50 jr oud.",
        "response": "De geboortedatum van [NAME_1] is [DOB_1]. Ze is nu [AGE_1] jaar oud."
    },
    {
        "prompt": "Haar gebortedatum is 12 januari 2024, en ze is 33 j.",
        "response": "Haar geboortedatum is [DOB_1], en ze is [AGE_1] jaar."
    },
    {
        "prompt": "Mijn broer is geboren op 1-13-2024 en hij is nu 27 jaar.",
        "response": "Mijn broer is geboren op [DOB_1] en hij is nu [AGE_1] jaar."
    },
    {
        "prompt": "Ze is op 1e dec 2024 geborne, en ze is 29 jaar oud.",
        "response": "Ze is op [DOB_1] geboren, en ze is [AGE_1] jaar oud."
    },
    {
        "prompt": "Zijn geboortedatumm is 13 januari 2024. Hij is nu 37 jr.",
        "response": "Zijn geboortedatum is [DOB_1]. Hij is nu [AGE_1] jaar."
    },
    {
    "prompt": "i am tom. tom's email is tom@example.com, his BSN is 12345678. he lives at 100 Cedar Ave, and his zip code is 45678.",
    "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. He lives at [ADDRESS_1], and his zip code is [ZIP_1]."
    },
    {
        "prompt": "My name is SARA JOHNSON. My email is sara.johnson@exammple.com, and my BSN is 87654321. I live at 222 Ash St.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1]."
    },
    {
        "prompt": "allice.smith@example..com is my email. My BSN is 12345678. I reside at 15 elm street.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She resides at [ADDRESS_1]."
    },
    {
        "prompt": "robert's date of birth is 12-31-2024. his email is robert.brown@example.com, and his BSN is 34567890.",
        "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1], and his BSN is [BSN_1]."
    },
    {
        "prompt": "Linda johnson's email is linda.johnson@exammple..com, and her bsn is 11223344. she lives at 321 willow way.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1]."
    },
    {
        "prompt": "my friend max's date of birth is 2024-01-01. his email is max@example..com. his postal code is 12345.",
        "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1]. His postal code is [ZIP_1]."
    },
    {
        "prompt": "JULIA's email is julia123@example.com, her bsn is 22334455. She was born on 01-12-2024.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She was born on [DOB_1]."
    },
    {
        "prompt": "Tom's email is TOM123@EXAMPLE.COM, and his bsn is 99988877. he was born on 12/01/2024.",
        "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. He was born on [DOB_1]."
    },
    {
        "prompt": "my cousin Anna's email is anna123@EXAMPLE.COM. Her BSN is 77788899, and she lives at 77 Elm St.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1]."
    },
    {
        "prompt": "Max's date of birth is 01/1/2024. his email is max@example.com. His postal code is 12345.",
        "response": "[NAME_1]'s date of birth is [DOB_1]. His email is [EMAIL_1]. His postal code is [ZIP_1]."
    },
    {
        "prompt": "Contact me at invalid@domain..com, my bsn is 123-45-678, and my ip is 192.168.0.2.",
        "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. His IP address is [IP_ADDRESS_1]."
    },
    {
        "prompt": "Emily's email is emily@example..com, and her BSN is 23456789. she lives at 789 pine st.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. She lives at [ADDRESS_1]."
    },
    {
        "prompt": "JOHN DOE's email is john.doe@example.com. His bsn is 12345678. he lives at 10 langelo!",
        "response": "[NAME_1]'s email is [EMAIL_1]. His BSN is [BSN_1]. He lives at [ADDRESS_1]."
    },
    {
        "prompt": "Contact olivia at d.olivia@netwrix..com. Her bsn is 98765432. Her address is 60 langelo langelo.",
        "response": "[NAME_1]'s email is [EMAIL_1]. Her BSN is [BSN_1]. Her address is [ADDRESS_1]."
    },
    {
        "prompt": "i am tom.",
        "response": "I am [NAME_1]."
    },
    {
        "prompt": "my name is alice.",
        "response": "My name is [NAME_1]."
    },
    {
        "prompt": "robert's email is robert.brown@example.com.",
        "response": "[NAME_1]'s email is [EMAIL_1]."
    },
    {
        "prompt": "the bsn for jane is 98765432.",
        "response": "The BSN for [NAME_1] is [BSN_1]."
    },
    {
        "prompt": "contact me at myemail@domain..com.",
        "response": "Contact me at [EMAIL_1]."
    },
    {
        "prompt": "my address is 123 main st.",
        "response": "My address is [ADDRESS_1]."
    },
    {
        "prompt": "i live at 456 oak street.",
        "response": "I live at [ADDRESS_1]."
    },
    {
        "prompt": "my email is jane@example.com.",
        "response": "My email is [EMAIL_1]."
    },
    {
        "prompt": "i was born on 12/01/2024.",
        "response": "I was born on [DOB_1]."
    },
    {
        "prompt": "michael's zip code is 12345.",
        "response": "[NAME_1]'s zip code is [ZIP_1]."
    },
    {
        "prompt": "i have a BSN 12345678.",
        "response": "I have a BSN [BSN_1]."
    },
    {
        "prompt": "my name is john, but people call me j.",
        "response": "My name is [NAME_1], but people call me [NAME_2]."
    },
    {
        "prompt": "the email for emily is emily@sample.com.",
        "response": "The email for [NAME_1] is [EMAIL_1]."
    },
    {
        "prompt": "my postal code is 54321.",
        "response": "My postal code is [ZIP_1]."
    },
    {
        "prompt": "i am from new york city.",
        "response": "I am from [ADDRESS_1]."
    },
    {
        "prompt": "tom is a friend of mine.",
        "response": "[NAME_1] is a friend of mine."
    },
    {
        "prompt": "my friend's email is friend@example.com.",
        "response": "My friend's email is [EMAIL_1]."
    },
    {
        "prompt": "i was born in 2024.",
        "response": "I was born in [DOB_1]."
    },

     # Cases without personal data
    {
        "prompt": "The weather in London is quite unpredictable in December.",
        "response": "The weather in London is quite unpredictable in December."
    },
    {
        "prompt": "Tomorrow is a national holiday, and many people are excited about the parade.",
        "response": "Tomorrow is a national holiday, and many people are excited about the parade."
    },
    {
        "prompt": "The capital of France is Paris, and it's known for its rich history and culture.",
        "response": "The capital of France is Paris, and it's known for its rich history and culture."
    },
    {
        "prompt": "In 2024, the Olympics will be hosted by Paris, and athletes from all over the world will compete.",
        "response": "In 2024, the Olympics will be hosted by Paris, and athletes from all over the world will compete."
    },
    {
        "prompt": "I enjoy reading books in my free time, especially science fiction novels.",
        "response": "I enjoy reading books in my free time, especially science fiction novels."
    },
    {
        "prompt": "The museum closes at 6 PM, so we should plan to arrive before that.",
        "response": "The museum closes at 6 PM, so we should plan to arrive before that."
    },
    {
        "prompt": "Mount Everest is the highest mountain in the world, located in the Himalayas.",
        "response": "Mount Everest is the highest mountain in the world, located in the Himalayas."
    },
    {
        "prompt": "The Great Wall of China stretches over 13,000 miles and is one of the most famous landmarks in the world.",
        "response": "The Great Wall of China stretches over 13,000 miles and is one of the most famous landmarks in the world."
    },
    {
        "prompt": "She has a passion for photography and often travels to capture stunning landscapes.",
        "response": "She has a passion for photography and often travels to capture stunning landscapes."
    },
    {
        "prompt": "The concert will take place in Central Park next Saturday.",
        "response": "The concert will take place in Central Park next Saturday."
    },
    {
        "prompt": "Python is a popular programming language used for web development, data analysis, and automation.",
        "response": "Python is a popular programming language used for web development, data analysis, and automation."
    },
    {
        "prompt": "The movie is set to be released in theaters worldwide in 2025.",
        "response": "The movie is set to be released in theaters worldwide in 2025."
    },
    {
        "prompt": "The book I'm reading right now is a bestseller, and it's about artificial intelligence.",
        "response": "The book I'm reading right now is a bestseller, and it's about artificial intelligence."
    },
    {
        "prompt": "Her favorite hobby is painting landscapes, and she has her own studio.",
        "response": "Her favorite hobby is painting landscapes, and she has her own studio."
    },
    {
        "prompt": "The football match ended in a draw, with both teams scoring two goals each.",
        "response": "The football match ended in a draw, with both teams scoring two goals each."
    },
    {
        "prompt": "The weather today is sunny with a high of 75°F. Everyone should enjoy their day outside!",
        "response": "The weather today is sunny with a high of 75°F. Everyone should enjoy their day outside!"
    },
    {
        "prompt": "The conference will take place next month, bringing together experts from various fields to discuss the latest trends.",
        "response": "The conference will take place next month, bringing together experts from various fields to discuss the latest trends."
    },
    {
        "prompt": "The team's performance in the last match was outstanding, earning them a spot in the playoffs.",
        "response": "The team's performance in the last match was outstanding, earning them a spot in the playoffs."
    },
    {
        "prompt": "Our new product will be launched next quarter, aiming to revolutionize the industry with its innovative features.",
        "response": "Our new product will be launched next quarter, aiming to revolutionize the industry with its innovative features."
    },
    {
        "prompt": "The company has decided to implement a new remote work policy to improve employee satisfaction.",
        "response": "The company has decided to implement a new remote work policy to improve employee satisfaction."
    },
    {
        "prompt": "The school is organizing a charity event to support local families in need. Everyone is encouraged to participate.",
        "response": "The school is organizing a charity event to support local families in need. Everyone is encouraged to participate."
    },
    {
        "prompt": "The library will be hosting a book fair next weekend, featuring various genres and activities for all ages.",
        "response": "The library will be hosting a book fair next weekend, featuring various genres and activities for all ages."
    },
    {
        "prompt": "The hiking trail offers breathtaking views and is a popular destination for outdoor enthusiasts.",
        "response": "The hiking trail offers breathtaking views and is a popular destination for outdoor enthusiasts."
    },
    {
        "prompt": "This recipe is perfect for a cozy night in, combining flavors that everyone loves.",
        "response": "This recipe is perfect for a cozy night in, combining flavors that everyone loves."
    },
    {
        "prompt": "Traveling is a wonderful way to experience new cultures and meet different people around the world.",
        "response": "Traveling is a wonderful way to experience new cultures and meet different people around the world."
    },
    {
        "prompt": "Learning a new language can be challenging, but it opens up a world of opportunities.",
        "response": "Learning a new language can be challenging, but it opens up a world of opportunities."
    },
    {
        "prompt": "Exercise is essential for maintaining a healthy lifestyle and can greatly improve mental well-being.",
        "response": "Exercise is essential for maintaining a healthy lifestyle and can greatly improve mental well-being."
    },
    {
        "prompt": "The latest smartphone has impressive features that enhance user experience and productivity.",
        "response": "The latest smartphone has impressive features that enhance user experience and productivity."
    },
    {
        "prompt": "Art plays a crucial role in society, providing a means of expression and communication.",
        "response": "Art plays a crucial role in society, providing a means of expression and communication."
    },
    {
        "prompt": "Music brings people together and can evoke a wide range of emotions.",
        "response": "Music brings people together and can evoke a wide range of emotions."
    },
    {
        "prompt": "Volunteering can be a rewarding experience, allowing individuals to give back to their communities.",
        "response": "Volunteering can be a rewarding experience, allowing individuals to give back to their communities."
    },
    {
        "prompt": "I love going for walks in the park, especially when the weather is nice.",
        "response": "I love going for walks in the park, especially when the weather is nice."
    },
    {
        "prompt": "I enjoy reading books about different cultures and places around the world.",
        "response": "I enjoy reading books about different cultures and places around the world."
    },
    {
        "prompt": "I recently watched a movie that inspired me to pursue my passions more actively.",
        "response": "I recently watched a movie that inspired me to pursue my passions more actively."
    },
    {
        "prompt": "I think learning a new language can be challenging but also very rewarding.",
        "response": "I think learning a new language can be challenging but also very rewarding."
    },
    {
        "prompt": "I appreciate the beauty of nature and often spend my weekends hiking.",
        "response": "I appreciate the beauty of nature and often spend my weekends hiking."
    },
    {
        "prompt": "I have been experimenting with new recipes in the kitchen, and it has been a fun experience.",
        "response": "I have been experimenting with new recipes in the kitchen, and it has been a fun experience."
    },
    {
        "prompt": "I believe that staying active is important for a healthy lifestyle.",
        "response": "I believe that staying active is important for a healthy lifestyle."
    },
    {
        "prompt": "I enjoy attending local events and meeting new people in my community.",
        "response": "I enjoy attending local events and meeting new people in my community."
    },
    {
        "prompt": "I like to take time to reflect on my goals and aspirations regularly.",
        "response": "I like to take time to reflect on my goals and aspirations regularly."
    },
    {
        "prompt": "I find that listening to music helps me focus while I work.",
        "response": "I find that listening to music helps me focus while I work."
    },
    {
        "prompt": "I often share my thoughts on various topics with friends and family.",
        "response": "I often share my thoughts on various topics with friends and family."
    },
    {
        "prompt": "I think that volunteering can be a fulfilling way to give back to the community.",
        "response": "I think that volunteering can be a fulfilling way to give back to the community."
    },
    {
        "prompt": "I look forward to the weekends, as they give me a chance to relax and recharge.",
        "response": "I look forward to the weekends, as they give me a chance to relax and recharge."
    },
    {
        "prompt": "I appreciate art and enjoy visiting galleries to see new exhibitions.",
        "response": "I appreciate art and enjoy visiting galleries to see new exhibitions."
    },
    {
        "prompt": "I enjoy exploring new hobbies that challenge me and help me grow.",
        "response": "I enjoy exploring new hobbies that challenge me and help me grow."
    },
]

# Converting to dataset format with input, instruction, and output
formatted_data = [
    {
        "instruction": "Anonymize the data",
        "input": entry["prompt"],
        "output": entry["response"]
    }
    for entry in data
]

# Create Hugging Face Dataset
dataset = Dataset.from_list(formatted_data)

# Output the dataset to see its structure
print(dataset[:])  # To print all rows

# Save the dataset as a JSON file locally
json_file_path = "datasets/blueVi-GPT-dataset.json"
dataset.to_json(json_file_path)

# Create the Hugging Face dataset repo if it doesn't exist
repo_name = "ThanhTranVisma/blueVi-GPT-dataset"
create_repo(repo_name, repo_type="dataset", private=True)

# Push JSON file to Hugging Face Hub
api = HfApi()
api.upload_file(
    path_or_fileobj=json_file_path,
    path_in_repo="blueVi-GPT-dataset.json",
    repo_id=repo_name,
    repo_type="dataset",
    token=hf_token,
)

print(f"Dataset successfully uploaded to {repo_name}!")
