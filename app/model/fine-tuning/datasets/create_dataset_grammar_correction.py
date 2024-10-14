from datasets import Dataset
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()
hf_token = os.getenv("HUGGING_FACE_TOKEN")

data = [
    {"prompt": "My name are TOm and my sister name is hella.", "response": "My name is Tom, and my sister's name is Hella."},
    {"prompt": "I wants to eats pizza everyday.", "response": "I want to eat pizza every day."},
    {"prompt": "She go to the market to buys fruit.", "response": "She goes to the market to buy fruit."},
    {"prompt": "He don't know where is the bus station.", "response": "He doesn't know where the bus station is."},
    {"prompt": "Their going to the park after school.", "response": "They're going to the park after school."},
    {"prompt": "I am studying in the libary for my exame.", "response": "I am studying in the library for my exam."},
    {"prompt": "The cat is playng with it's toy.", "response": "The cat is playing with its toy."},
    {"prompt": "We was waiting for the bus for hours.", "response": "We were waiting for the bus for hours."},
    {"prompt": "She have a beautiful garden infront of her house.", "response": "She has a beautiful garden in front of her house."},
    {"prompt": "They is planning to go for vacction next month.", "response": "They are planning to go for vacation next month."},
    {"prompt": "The meeting will occured tomorrow.", "response": "The meeting will occur tomorrow."},
    {"prompt": "I seen him at the mall yesterday.", "response": "I saw him at the mall yesterday."},
    {"prompt": "My phone doesn't works properly.", "response": "My phone doesn't work properly."},
    {"prompt": "He forget to take his umbrella.", "response": "He forgot to take his umbrella."},
    {"prompt": "She is taller then her brother.", "response": "She is taller than her brother."},
    {"prompt": "We have went to that restaurent before.", "response": "We have been to that restaurant before."},
    {"prompt": "It's a beautifull day outside.", "response": "It's a beautiful day outside."},
    {"prompt": "The children is playing in the garden.", "response": "The children are playing in the garden."},
    {"prompt": "I haved never been to Paris.", "response": "I have never been to Paris."},
    {"prompt": "Tom and Jerry was best friends.", "response": "Tom and Jerry were best friends."},
     # Name and general information
    {"prompt": "My name are TOm and my sister name is hella.", "response": "My name is Tom, and my sister's name is Hella."},
    {"prompt": "His email is jhon.doe@email.con", "response": "His email is john.doe@email.com."},
    
    # BSN (Dutch citizen service number)
    {"prompt": "His BSN number are 12345 67 890", "response": "His BSN number is 1234567890."},
    {"prompt": "My BSN are 098-765-432.", "response": "My BSN is 098765432."},
    
    # Credit card information
    {"prompt": "Her Mastercard number is 1234-5678-9012-345", "response": "Her Mastercard number is 1234-5678-9012-3456."},
    {"prompt": "My credit card VISA is expiring 12/2020.", "response": "My VISA credit card is expiring in 12/2023."},
    
    # IBAN
    {"prompt": "His IBAN is DE44-500-10500-0530200", "response": "His IBAN is DE445001050000530200."},
    {"prompt": "My bank account IBAN: NL91ABNA041716430", "response": "My bank account IBAN is NL91ABNA0417164300."},
    
    # Address and ZIP code
    {"prompt": "My adress is 123 Fake St., Cityvil, 1234 AB", "response": "My address is 123 Fake St., Cityville, 1234 AB."},
    {"prompt": "His zip code are 567890.", "response": "His zip code is 56789."},
    
    # Date of birth (DOB)
    {"prompt": "Her DOB is 32nd of May, 1990.", "response": "Her DOB is 31st of May, 1990."},
    {"prompt": "I was born on 1995-13-01.", "response": "I was born on 1995-12-01."},
    
    # IP address
    {"prompt": "My IP adress is 192.168.1.300.", "response": "My IP address is 192.168.1.30."},
    {"prompt": "Her IP address are 256.256.256.256.", "response": "Her IP address is invalid."},

    # Phone numbers and additional information
    {"prompt": "His phone number are +1 (555) 123-456", "response": "His phone number is +1 (555) 123-4567."},
    {"prompt": "My phone are 123-456-7890.", "response": "My phone number is 123-456-7890."},
    
    # Grammar and Typo Corrections
    {"prompt": "She go to the market to buys fruit.", "response": "She goes to the market to buy fruit."},
    {"prompt": "We was waiting for the bus for hours.", "response": "We were waiting for the bus for hours."},
    {"prompt": "They is planning to go for vacction next month.", "response": "They are planning to go for vacation next month."},
    {"prompt": "I seen him at the mall yesterday.", "response": "I saw him at the mall yesterday."},
    {"prompt": "The cat is playng with it's toy.", "response": "The cat is playing with its toy."},
    {"prompt": "She is taller then her brother.", "response": "She is taller than her brother."},
    {"prompt": "We have went to that restaurent before.", "response": "We have been to that restaurant before."},
    
    # Name and general information
    {"prompt": "My name is Jonn and my sisster name is Helda.", "response": "My name is John, and my sister's name is Hilda."},
    {"prompt": "Hes email is jennifer.doe@email.co", "response": "Her email is jennifer.doe@email.com."},

    # BSN (Dutch citizen service number)
    {"prompt": "My BSN number is 12345 67 89", "response": "My BSN number is 123456789."},
    {"prompt": "His BSN number are 123-45-6789", "response": "His BSN number is 123456789."},

    # Credit card information
    {"prompt": "Her Mastecard number is 1234-5678-9012-34", "response": "Her Mastercard number is 1234-5678-9012-3456."},
    {"prompt": "My visa credit card number is 4242-4242-4242-424", "response": "My VISA credit card number is 4242-4242-4242-4242."},

    # IBAN
    {"prompt": "His IBAN is DE4450010500005302000", "response": "His IBAN is DE445001050000530200."},
    {"prompt": "My bank account IBAN is NL91ABNA04171643", "response": "My bank account IBAN is NL91ABNA041716430."},

    # Address and ZIP code
    {"prompt": "I live on 123 Fake Ave, Cityville, 12345A", "response": "I live at 123 Fake Ave, Cityville, 12345."},
    {"prompt": "My adress is 456 Elm St., Townsville, 2345", "response": "My address is 456 Elm St., Townsville, 23456."},

    # Date of birth (DOB)
    {"prompt": "My DOB is 31st February, 1990.", "response": "My DOB is 28th February, 1990."},
    {"prompt": "She was born on 2000-14-07.", "response": "She was born on 2000-07-14."},

    # IP address
    {"prompt": "My IP adrees is 192.168.1.256.", "response": "My IP address is 192.168.1.255."},
    {"prompt": "Her IP address are 10.0.0.999.", "response": "Her IP address is invalid."},

    # Phone numbers and additional information
    {"prompt": "My phone number is (555) 123-456.", "response": "My phone number is (555) 123-4567."},
    {"prompt": "His contact number is 123.456.7890.", "response": "His contact number is 123-456-7890."},

    # More grammar and typo corrections
    {"prompt": "They is going to the mall this weekend.", "response": "They are going to the mall this weekend."},
    {"prompt": "I ain't got no money left.", "response": "I don't have any money left."},
    {"prompt": "She don't like the movie, it was boring.", "response": "She doesn't like the movie; it was boring."},
    {"prompt": "The quick brown fox jump over the lazy dog.", "response": "The quick brown fox jumps over the lazy dog."},
    {"prompt": "Its raining cats and dog outside.", "response": "It's raining cats and dogs outside."},
    {"prompt": "We should of gone to the concert yesterday.", "response": "We should have gone to the concert yesterday."},
    {"prompt": "He should of called me earlier.", "response": "He should have called me earlier."},
    {"prompt": "Her favorite fruit is banana's.", "response": "Her favorite fruit is bananas."},
    {"prompt": "I have seen that movie three times already, it's great!", "response": "I have seen that movie three times already; it's great!"},
    
      # Name and general information
    {"prompt": "My name is Alice and my broher name is Bob.", "response": "My name is Alice, and my brother's name is Bob."},
    {"prompt": "Hes email is alice.smithgmail.com", "response": "Her email is alice.smith@gmail.com."},

    # BSN (Dutch citizen service number)
    {"prompt": "My BSN number is 1234567890", "response": "My BSN number is 123-45-6789."},
    {"prompt": "His BSN is 098765 432", "response": "His BSN is 098-76-5432."},

    # Credit card information
    {"prompt": "Her MasterCard number is 5123-4567-8910-1234.", "response": "Her MasterCard number is 5123-4567-8910-1234."},
    {"prompt": "My Visa card number is 4111-1111-1111-1111", "response": "My Visa card number is 4111-1111-1111-1111."},

    # IBAN
    {"prompt": "My IBAN is GB29NWBK60161331926819", "response": "My IBAN is GB29 NWBK 6016 1331 9268 19."},
    {"prompt": "His bank account IBAN is FR76 3000 6000 0112 3456 7890 189", "response": "His bank account IBAN is FR76 3000 6000 0112 3456 7890 189."},

    # Address and ZIP code
    {"prompt": "I live at 789 Maple St, Hometown, 12345-678", "response": "I live at 789 Maple St, Hometown, 12345."},
    {"prompt": "My adress is 321 Oak Dr., Villagetown, 67890", "response": "My address is 321 Oak Dr., Villagetown, 67890."},

    # Date of birth (DOB)
    {"prompt": "I was born on 15th March 1985.", "response": "I was born on March 15, 1985."},
    {"prompt": "Her DOB is 1990/07/21.", "response": "Her DOB is 1990-07-21."},

    # IP address
    {"prompt": "My IP addres is 192.300.1.1.", "response": "My IP address is 192.168.1.1."},
    {"prompt": "Her IP adress are 256.256.256.256.", "response": "Her IP address is invalid."},

    # Phone numbers and additional information
    {"prompt": "My phone number is (123) 456-789", "response": "My phone number is (123) 456-7890."},
    {"prompt": "His cell number is 123.456.7890", "response": "His cell number is 123-456-7890."},

    # More grammar and typo corrections
    {"prompt": "The team are working on the project.", "response": "The team is working on the project."},
    {"prompt": "I have went to the store earlier.", "response": "I went to the store earlier."},
    {"prompt": "He don't have the necessary skills.", "response": "He doesn't have the necessary skills."},
    {"prompt": "Their going to the concert tonight.", "response": "They're going to the concert tonight."},
    {"prompt": "Its important to take care of your health.", "response": "It's important to take care of your health."},
    {"prompt": "She seen the movie already.", "response": "She has seen the movie already."},
    {"prompt": "I could of finished the work by now.", "response": "I could have finished the work by now."},
    {"prompt": "They both are good friends since childhood.", "response": "They have been good friends since childhood."},
    {"prompt": "Everyone should mind their own business.", "response": "Everyone should mind their own business."},
    {"prompt": "I had a great time at the partty!", "response": "I had a great time at the party!"},

    # More varied examples
    {"prompt": "The book is on the table, next too the lamp.", "response": "The book is on the table, next to the lamp."},
    {"prompt": "I am going to the grosery store later.", "response": "I am going to the grocery store later."},
    {"prompt": "My favrit color is blue.", "response": "My favorite color is blue."},
    {"prompt": "He have a lot of work to do.", "response": "He has a lot of work to do."},
    {"prompt": "Can you lend me your pen, please.", "response": "Can you lend me your pen, please?"},

    # Additional personal information examples
    {"prompt": "My email is johndoe@emal.com", "response": "My email is johndoe@email.com."},
    {"prompt": "My bro has his mastercard with him", "response": "My brother has his Mastercard with him."},
    {"prompt": "I live on 321 Spruce Road, 65432 1A", "response": "I live at 321 Spruce Road, 65432."},
    {"prompt": "My dob is 29-02-1996", "response": "My DOB is 29-02-1996."},
    {"prompt": "My ip is 192.256.0.1", "response": "My IP is invalid."},
    {"prompt": "Her bsn number is 1234 5678 910", "response": "Her BSN number is 123-45-6789."},
    
    # Name and general information
    {"prompt": "My name is Clara and my brothe's name is Felix. We live together in a small apartment.", "response": "My name is Clara, and my brother's name is Felix. We live together in a small apartment."},
    {"prompt": "You can reach me at my email which is clara_smith@email.com, that’s the best way to contact me.", "response": "You can reach me at my email, clara_smith@email.com; that’s the best way to contact me."},

    # BSN (Dutch citizen service number)
    {"prompt": "I recently received my BSN number, which is 9876543210, and I was so relieved to have it.", "response": "I recently received my BSN number, which is 987-65-4321, and I was so relieved to have it."},
    {"prompt": "His BSN is 123456 789 and he uses it for all his official documents.", "response": "His BSN is 123-45-6789, and he uses it for all his official documents."},

    # Credit card information
    {"prompt": "I have a MasterCard with the number 5432-1098-7654-3210, which I use for online shopping.", "response": "I have a MasterCard with the number 5432-1098-7654-3210, which I use for online shopping."},
    {"prompt": "My Visa card number is 4222-2222-2222-2222 and I try to keep it safe and secure.", "response": "My Visa card number is 4222-2222-2222-2222, and I try to keep it safe and secure."},

    # IBAN
    {"prompt": "I opened a new bank account and my IBAN is DE89370400440532013000, which is pretty long.", "response": "I opened a new bank account, and my IBAN is DE89 3704 0044 0532 0130 0000, which is pretty long."},
    {"prompt": "His bank account IBAN is NL91ABNA0417164300, and it is essential for international transfers.", "response": "His bank account IBAN is NL91 ABNA 0417 1643 00, and it is essential for international transfers."},

    # Address and ZIP code
    {"prompt": "I recently moved to 1234 Elm Street, Springfield, with the ZIP code 54321, and I love it here.", "response": "I recently moved to 1234 Elm Street, Springfield, with the ZIP code 54321, and I love it here."},
    {"prompt": "My address is 9876 Pine Avenue, Uptown, ZIP 67890, and it’s a quiet neighborhood.", "response": "My address is 9876 Pine Avenue, Uptown, ZIP code 67890, and it’s a quiet neighborhood."},

    # Date of birth (DOB)
    {"prompt": "I was born on the 1st of January, 1990, and every year I throw a big birthday party.", "response": "I was born on January 1, 1990, and every year, I throw a big birthday party."},
    {"prompt": "Her DOB is 12/05/1988 and she always celebrates it with her family.", "response": "Her DOB is December 5, 1988, and she always celebrates it with her family."},

    # IP address
    {"prompt": "I checked my IP address yesterday and it was 10.0.0.1, which is quite common for private networks.", "response": "I checked my IP address yesterday, and it was 10.0.0.1, which is quite common for private networks."},
    {"prompt": "His IP address is 192.168.1.254, but he isn’t sure if it’s secure enough.", "response": "His IP address is 192.168.1.254, but he isn’t sure if it’s secure enough."},

    # Phone numbers and additional information
    {"prompt": "My phone number is (555) 123-4567, and you can call me anytime during business hours.", "response": "My phone number is (555) 123-4567, and you can call me anytime during business hours."},
    {"prompt": "Her cell number is 987-654-3210, but she prefers text messages over phone calls.", "response": "Her cell number is 987-654-3210, but she prefers text messages over phone calls."},

    # More grammar and typo corrections
    {"prompt": "The project have been very challenging, but we are making progress.", "response": "The project has been very challenging, but we are making progress."},
    {"prompt": "I should of studied more for that test, I really didn’t expect it to be that hard.", "response": "I should have studied more for that test; I really didn’t expect it to be that hard."},
    {"prompt": "Their going to the park after school to play some basketball, which sounds like fun.", "response": "They’re going to the park after school to play some basketball, which sounds like fun."},
    {"prompt": "Its crucial to remember to backup your data regularly, so you don’t lose anything important.", "response": "It's crucial to remember to back up your data regularly, so you don’t lose anything important."},
    {"prompt": "She told me that she seen the new movie last night and loved it so much.", "response": "She told me that she saw the new movie last night and loved it so much."},

    # Additional personal information examples
    {"prompt": "My email address is johndoe@example.com, and I use it for work-related correspondence.", "response": "My email address is johndoe@example.com, and I use it for work-related correspondence."},
    {"prompt": "I carry my Mastercard with me everywhere, it's very convenient for payments.", "response": "I carry my Mastercard with me everywhere; it's very convenient for payments."},
    {"prompt": "I just filled out my new address form, which is 555 Birch Lane, 12345 Citytown, and I hope it gets processed quickly.", "response": "I just filled out my new address form, which is 555 Birch Lane, 12345 Citytown, and I hope it gets processed quickly."},
    {"prompt": "I have an upcoming birthday on 02-29-1996, and I'm really looking forward to the celebration.", "response": "I have an upcoming birthday on February 29, 1996, and I'm really looking forward to the celebration."},
    {"prompt": "The IP I’m using at home is 192.168.0.101, but I’m not sure if it’s the same at work.", "response": "The IP I’m using at home is 192.168.0.101, but I’m not sure if it’s the same at work."},
    {"prompt": "Her BSN number, which is 1234-5678-910, is required for her application to be processed.", "response": "Her BSN number, which is 123-45-6789, is required for her application to be processed."},
    
    # Name and general information
    {"prompt": "My name are Alex and my sisters name is Mia, we both enjoy hiking during weekends.", "response": "My name is Alex, and my sister's name is Mia. We both enjoy hiking during weekends."},
    {"prompt": "You can contact me through my email: alex_jones123@domain.com, which I check regularly.", "response": "You can contact me through my email: alex_jones123@domain.com, which I check regularly."},

    # BSN (Dutch citizen service number)
    {"prompt": "I got my BSN last week, its number is 0123456789 and I was thrilled to receive it.", "response": "I got my BSN last week; its number is 012-34-5678, and I was thrilled to receive it."},
    {"prompt": "My partner's BSN number is 112233445 and it’s needed for his new job paperwork.", "response": "My partner's BSN number is 112-23-3445, and it’s needed for his new job paperwork."},

    # Credit card information
    {"prompt": "I usually pay with my MasterCard, the number is 5167-1234-5678-9010, especially for online purchases.", "response": "I usually pay with my MasterCard; the number is 5167-1234-5678-9010, especially for online purchases."},
    {"prompt": "I have a Visa card with the number 4012-8888-8888-1881 that I use for emergencies only.", "response": "I have a Visa card with the number 4012-8888-8888-1881, which I use for emergencies only."},

    # IBAN
    {"prompt": "I received my IBAN recently, it’s ES7921000813610123456789, and it feels nice to have it sorted out.", "response": "I received my IBAN recently; it’s ES79 2100 0813 6101 2345 6789, and it feels nice to have it sorted out."},
    {"prompt": "His IBAN is GB29NWBK60161331926819, and he needs it for his salary payments.", "response": "His IBAN is GB29 NWBK 6016 1331 9268 19, and he needs it for his salary payments."},

    # Address and ZIP code
    {"prompt": "I just moved to 4567 Cedar St, Townsville, with ZIP code 12345 and the neighbors seem friendly.", "response": "I just moved to 4567 Cedar St, Townsville, with ZIP code 12345, and the neighbors seem friendly."},
    {"prompt": "My new address is 7890 Maple Drive, Cityland, ZIP 67890, and I am excited to decorate my room.", "response": "My new address is 7890 Maple Drive, Cityland, ZIP code 67890, and I am excited to decorate my room."},

    # Date of birth (DOB)
    {"prompt": "I celebrate my birthday on 03-15-1985, every year I have a small get together with friends.", "response": "I celebrate my birthday on March 15, 1985; every year, I have a small get-together with friends."},
    {"prompt": "Her birthday is 07/20/1992, and she always looks forward to the summer festivities.", "response": "Her birthday is July 20, 1992, and she always looks forward to the summer festivities."},

    # IP address
    {"prompt": "I found out my IP address today, it is 172.16.254.1, which is private and not accessible from outside.", "response": "I found out my IP address today; it is 172.16.254.1, which is private and not accessible from outside."},
    {"prompt": "His IP address 192.168.0.105 might be changeing soon since he’s switching providers.", "response": "His IP address 192.168.0.105 might be changing soon since he’s switching providers."},

    # Phone numbers and additional information
    {"prompt": "My contact number is (123) 456-7890, feel free to reach out if you have any questions.", "response": "My contact number is (123) 456-7890; feel free to reach out if you have any questions."},
    {"prompt": "You can call her at 321-654-0987, but she prefers if you send her a message instead.", "response": "You can call her at 321-654-0987, but she prefers if you send her a message instead."},

    # More grammar and typo corrections
    {"prompt": "He said that he don’t want to go to the party, and it really surprised me.", "response": "He said that he doesn’t want to go to the party, and it really surprised me."},
    {"prompt": "I could of finished the project sooner if I had more time to work on it.", "response": "I could have finished the project sooner if I had more time to work on it."},
    {"prompt": "Their going to the concert next week and I can’t wait to join them.", "response": "They’re going to the concert next week, and I can’t wait to join them."},
    {"prompt": "Its amazing how much you can learn by just reading books, it opens up new worlds.", "response": "It's amazing how much you can learn by just reading books; it opens up new worlds."},
    {"prompt": "She said she seen the best movie last night, and I need to watch it soon.", "response": "She said she saw the best movie last night, and I need to watch it soon."},

    # Additional personal information examples
    {"prompt": "My email is sarah.connor@mail.com, and I use it primarily for work related matters.", "response": "My email is sarah.connor@mail.com, and I use it primarily for work-related matters."},
    {"prompt": "I keep my MasterCard handy for shopping, it’s the one ending in 1234, and I feel secure using it.", "response": "I keep my MasterCard handy for shopping; it’s the one ending in 1234, and I feel secure using it."},
    {"prompt": "The address I provided is 555 Oak Lane, 54321 Village, and I hope they send the package there.", "response": "The address I provided is 555 Oak Lane, 54321 Village, and I hope they send the package there."},
    {"prompt": "I was born on 04-22-1990 and I always enjoy celebrating it with my family and friends.", "response": "I was born on April 22, 1990, and I always enjoy celebrating it with my family and friends."},
    {"prompt": "For my home network, my IP address is 10.0.0.2, and I am worried it might change soon.", "response": "For my home network, my IP address is 10.0.0.2, and I am worried it might change soon."},
    {"prompt": "Her BSN is 654-32-1098, and she needs it for her government application.", "response": "Her BSN is 654-32-1098, and she needs it for her government application."},

    # Name and general information
    {"prompt": "My name are Clara and my brother name is Jake, we often play video games together in the evenings.", "response": "My name is Clara, and my brother's name is Jake. We often play video games together in the evenings."},
    {"prompt": "You can reach me at my email: clara.brown2020@service.com, which I check multiple times a day.", "response": "You can reach me at my email: clara.brown2020@service.com, which I check multiple times a day."},

    # BSN (Dutch citizen service number)
    {"prompt": "I finally received my BSN; it is 9876543210, and it is essential for my new job application.", "response": "I finally received my BSN; it is 987-65-4321, and it is essential for my new job application."},
    {"prompt": "My wife’s BSN number is 123456789 and she is really happy to have it for her tax returns.", "response": "My wife’s BSN number is 123-45-6789, and she is really happy to have it for her tax returns."},

    # Credit card information
    {"prompt": "I tend to use my MasterCard for most purchases; the number is 5486-1234-5678-9012, especially when shopping online.", "response": "I tend to use my MasterCard for most purchases; the number is 5486-1234-5678-9012, especially when shopping online."},
    {"prompt": "My Visa card is 4024-0070-0014-0008 and I usually keep it for emergencies, just in case something comes up.", "response": "My Visa card is 4024-0070-0014-0008, and I usually keep it for emergencies, just in case something comes up."},

    # IBAN
    {"prompt": "The IBAN I received is FR7612345678901234567890123, which I need to set up my direct deposit.", "response": "The IBAN I received is FR76 1234 5678 9012 3456 7890 123, which I need to set up my direct deposit."},
    {"prompt": "My friend's IBAN is NL91ABNA0417164300, and he has to provide it for international transfers.", "response": "My friend's IBAN is NL91 ABNA 0417 1643 00, and he has to provide it for international transfers."},

    # Address and ZIP code
    {"prompt": "I moved to 890 Pine Street, Springfield, with a ZIP code of 54321 and I can’t wait to explore the area.", "response": "I moved to 890 Pine Street, Springfield, with a ZIP code of 54321, and I can’t wait to explore the area."},
    {"prompt": "Currently, my address is 1234 Elm Avenue, Metropolis, ZIP code 67890, and I enjoy the local cafes around.", "response": "Currently, my address is 1234 Elm Avenue, Metropolis, ZIP code 67890, and I enjoy the local cafes around."},

    # Date of birth (DOB)
    {"prompt": "I was born on 12-31-1995 and every year my friends throw a big party to celebrate New Year's Eve as well.", "response": "I was born on December 31, 1995, and every year, my friends throw a big party to celebrate New Year's Eve as well."},
    {"prompt": "Her birthday is 05/15/1988, and she always makes sure to take the day off from work to relax.", "response": "Her birthday is May 15, 1988, and she always makes sure to take the day off from work to relax."},

    # IP address
    {"prompt": "My IP address, which I found out last night, is 10.0.0.5, and I’m not sure if it’s static or dynamic.", "response": "My IP address, which I found out last night, is 10.0.0.5, and I’m not sure if it’s static or dynamic."},
    {"prompt": "He mentioned that his IP address is 172.16.254.10, but he has no idea how to find it on his computer.", "response": "He mentioned that his IP address is 172.16.254.10, but he has no idea how to find it on his computer."},

    # Phone numbers and additional information
    {"prompt": "My phone number is (987) 654-3210, so please feel free to give me a call anytime you need assistance.", "response": "My phone number is (987) 654-3210, so please feel free to give me a call anytime you need assistance."},
    {"prompt": "You can contact my sister at 123-456-7890, but she is usually busy during the weekdays.", "response": "You can contact my sister at 123-456-7890, but she is usually busy during the weekdays."},

    # More grammar and typo corrections
    {"prompt": "I don’t know where she gone, it’s unlike her to disappear without a trace.", "response": "I don’t know where she has gone; it’s unlike her to disappear without a trace."},
    {"prompt": "He told me he seen the latest movie, and I really want to hear his thoughts on it.", "response": "He told me he saw the latest movie, and I really want to hear his thoughts on it."},
    {"prompt": "Their going to the museum this weekend, and I’m excited to join them for a day of learning.", "response": "They’re going to the museum this weekend, and I’m excited to join them for a day of learning."},
    {"prompt": "Its important to take breaks while working; it can really help to refresh your mind and focus.", "response": "It's important to take breaks while working; it can really help to refresh your mind and focus."},
    {"prompt": "I would of loved to join you at the concert, but I had a prior commitment that I couldn’t miss.", "response": "I would have loved to join you at the concert, but I had a prior commitment that I couldn’t miss."},

    # Additional personal information examples
    {"prompt": "You can reach me at my email: emily.watson@example.com, and I usually reply within a few hours.", "response": "You can reach me at my email: emily.watson@example.com, and I usually reply within a few hours."},
    {"prompt": "My MasterCard number ends with 5678, and I make sure to monitor my transactions regularly.", "response": "My MasterCard number ends with 5678, and I make sure to monitor my transactions regularly."},
    {"prompt": "The address I provided is 777 Willow Lane, 12345 Town, and it should be listed in your records now.", "response": "The address I provided is 777 Willow Lane, 12345 Town, and it should be listed in your records now."},
    {"prompt": "I was born on 09-09-1993, and I love to celebrate my birthday with family every year.", "response": "I was born on September 9, 1993, and I love to celebrate my birthday with family every year."},
    {"prompt": "My private network IP address is 192.168.1.1, and I often have issues connecting my devices to it.", "response": "My private network IP address is 192.168.1.1, and I often have issues connecting my devices to it."},
    {"prompt": "Her BSN is 321-09-8765, and she has to present it for her student loan application.", "response": "Her BSN is 321-09-8765, and she has to present it for her student loan application."},

    # Personal names and general info
    {"prompt": "My name are Samuel and my sister name is Lila, we enjoy going hiking together every summer in the mountains.", "response": "My name is Samuel, and my sister's name is Lila. We enjoy going hiking together every summer in the mountains."},
    {"prompt": "Reach me on my email: sammy.smith@provider.com, I always check it multiple times a day for any news.", "response": "You can reach me at my email: sammy.smith@provider.com. I always check it multiple times a day for any news."},

    # BSN
    {"prompt": "I got my BSN recently, its number is 987654321 and I need it for my new job at the bank.", "response": "I got my BSN recently; its number is 987-65-4321, and I need it for my new job at the bank."},
    {"prompt": "Her BSN number is 123456789, which she requires for her healthcare services.", "response": "Her BSN number is 123-45-6789, which she requires for her healthcare services."},

    # Credit card info
    {"prompt": "I often use my MasterCard; my card number is 5486-1234-5678-9012, especially when I shop online for clothes.", "response": "I often use my MasterCard; my card number is 5486-1234-5678-9012, especially when I shop online for clothes."},
    {"prompt": "My Visa card number is 4024-0070-0014-0008, and I keep it safe for emergency situations.", "response": "My Visa card number is 4024-0070-0014-0008, and I keep it safe for emergency situations."},

    # IBAN
    {"prompt": "I recently set up my direct deposit with IBAN FR7612345678901234567890123 and its super convenient.", "response": "I recently set up my direct deposit with IBAN FR76 1234 5678 9012 3456 7890 123, and it's super convenient."},
    {"prompt": "My IBAN for receiving payments is NL91ABNA0417164300, and it was quite easy to set up.", "response": "My IBAN for receiving payments is NL91 ABNA 0417 1643 00, and it was quite easy to set up."},

    # Address and ZIP code
    {"prompt": "I moved to 890 Pine Street, Springfield, ZIP code is 54321, and I love the neighborhood’s friendly atmosphere.", "response": "I moved to 890 Pine Street, Springfield; the ZIP code is 54321, and I love the neighborhood’s friendly atmosphere."},
    {"prompt": "Currently, my home address is 1234 Elm Avenue, Metropolis, ZIP 67890, where I can find various shops and cafes.", "response": "Currently, my home address is 1234 Elm Avenue, Metropolis, ZIP 67890, where I can find various shops and cafes."},

    # Date of Birth (DOB)
    {"prompt": "I was born on 12-31-1995, and my family always celebrates my birthday with a big dinner on New Year's Eve.", "response": "I was born on December 31, 1995, and my family always celebrates my birthday with a big dinner on New Year's Eve."},
    {"prompt": "Her birthday is 05/15/1988, and she loves having a small get-together with close friends every year.", "response": "Her birthday is May 15, 1988, and she loves having a small get-together with close friends every year."},

    # IP address
    {"prompt": "I found out my IP address is 10.0.0.5, but I’m not sure if it’s the right one for my home network.", "response": "I found out my IP address is 10.0.0.5, but I’m not sure if it’s the right one for my home network."},
    {"prompt": "He told me his IP address is 172.16.254.10, and he struggles with his router settings often.", "response": "He told me his IP address is 172.16.254.10, and he struggles with his router settings often."},

    # Phone numbers and additional info
    {"prompt": "My contact number is (987) 654-3210, so you can call me anytime you need assistance.", "response": "My contact number is (987) 654-3210, so you can call me anytime you need assistance."},
    {"prompt": "You can reach my sister at 123-456-7890, but she’s usually busy with her studies.", "response": "You can reach my sister at 123-456-7890, but she’s usually busy with her studies."},

    # More grammar and typo corrections
    {"prompt": "I dont know where she gone, it’s unusual for her to vanish without a word.", "response": "I don’t know where she has gone; it’s unusual for her to vanish without a word."},
    {"prompt": "He told me he seen the new restaurant, and I want to know what he thinks about the food there.", "response": "He told me he saw the new restaurant, and I want to know what he thinks about the food there."},
    {"prompt": "Their planning to visit the zoo this weekend, and I’m thrilled to go with them for a day of fun.", "response": "They’re planning to visit the zoo this weekend, and I’m thrilled to go with them for a day of fun."},
    {"prompt": "Its very important to take breaks while studying; it can truly help to refresh your mind.", "response": "It's very important to take breaks while studying; it can truly help to refresh your mind."},
    {"prompt": "I would of liked to join you at the concert, but I had a commitment that I couldn’t miss.", "response": "I would have liked to join you at the concert, but I had a commitment that I couldn’t miss."},

    # More personal information examples
    {"prompt": "You can contact me via my email: alex.johnson@example.com, and I usually respond promptly.", "response": "You can contact me via my email: alex.johnson@example.com, and I usually respond promptly."},
    {"prompt": "My MasterCard number ends with 5678, and I often use it for online purchases to earn rewards.", "response": "My MasterCard number ends with 5678, and I often use it for online purchases to earn rewards."},
    {"prompt": "My address is 777 Willow Lane, 12345 Town, and I have a small garden that I take care of.", "response": "My address is 777 Willow Lane, 12345 Town, and I have a small garden that I take care of."},
    {"prompt": "I was born on 09-09-1993, and I love to celebrate my birthday with my family and friends every year.", "response": "I was born on September 9, 1993, and I love to celebrate my birthday with my family and friends every year."},
    {"prompt": "My private IP address is 192.168.1.1, and I often have issues connecting new devices to it.", "response": "My private IP address is 192.168.1.1, and I often have issues connecting new devices to it."},
    {"prompt": "Her BSN is 321-09-8765, and she needs to submit it for her loan application.", "response": "Her BSN is 321-09-8765, and she needs to submit it for her loan application."},

    # Additional entries with typos
    {"prompt": "I just remembered that my email is taylor.martin@xyz.com and I need to send an important document.", "response": "I just remembered that my email is taylor.martin@xyz.com, and I need to send an important document."},
    {"prompt": "His MasterCard number is 5432-8765-4321-0987, and he uses it to make travel reservations frequently.", "response": "His MasterCard number is 5432-8765-4321-0987, and he uses it to make travel reservations frequently."},
    {"prompt": "I’ve lived at 999 Maple Drive, 54321 City, for five years, and I enjoy the local parks.", "response": "I’ve lived at 999 Maple Drive, 54321 City, for five years, and I enjoy the local parks."},
    {"prompt": "My dob is 11-22-1985, and I usually celebrate with my friends at a nice restaurant.", "response": "My DOB is November 22, 1985, and I usually celebrate with my friends at a nice restaurant."},
    {"prompt": "They told me his IP is 172.31.255.1, but he always forgets how to check it on his computer.", "response": "They told me his IP is 172.31.255.1, but he always forgets how to check it on his computer."},
    
    # Personal names and general info
    {"prompt": "My name is Jonathan but my friends call me Jonny, we often hang out at the local coffee shop.", "response": "My name is Jonathan, but my friends call me Jonny. We often hang out at the local coffee shop."},
    {"prompt": "You can contact me at my email: julia.roberts@email.com, and I usually reply quickly to any messages.", "response": "You can contact me at my email: julia.roberts@email.com, and I usually reply quickly to any messages."},

    # BSN
    {"prompt": "I have a BSN which is 123456789, and I need it for my application to start working at the hospital.", "response": "I have a BSN, which is 123-45-6789, and I need it for my application to start working at the hospital."},
    {"prompt": "Her BSN number is 987654321, required for her student registration at the university.", "response": "Her BSN number is 987-65-4321, required for her student registration at the university."},

    # Credit card info
    {"prompt": "My MasterCard number is 5412-3456-7890-1234, and I prefer using it for online shopping due to its rewards.", "response": "My MasterCard number is 5412-3456-7890-1234, and I prefer using it for online shopping due to its rewards."},
    {"prompt": "I always use my Visa card which ends with 4321 for my travels, it's very reliable.", "response": "I always use my Visa card, which ends with 4321, for my travels; it's very reliable."},

    # IBAN
    {"prompt": "For my salary, I use IBAN DE89370400440532013000, and it makes receiving payments super easy.", "response": "For my salary, I use IBAN DE89 3704 0044 0532 0130 000; it makes receiving payments super easy."},
    {"prompt": "My IBAN for my savings account is ES9121000418450200051332, which I find very helpful for managing my finances.", "response": "My IBAN for my savings account is ES91 2100 0418 4502 0005 1332, which I find very helpful for managing my finances."},

    # Address and ZIP code
    {"prompt": "I live at 456 Oak Avenue, Big City, with a ZIP code of 12345, and it's a lovely place to reside.", "response": "I live at 456 Oak Avenue, Big City, with a ZIP code of 12345, and it's a lovely place to reside."},
    {"prompt": "Currently, I am staying at 789 Birch Street, 67890 Town, and it's close to my workplace.", "response": "Currently, I am staying at 789 Birch Street, 67890 Town, and it's close to my workplace."},

    # Date of Birth (DOB)
    {"prompt": "My birth date is 01-01-1990, and I usually celebrate my birthday with a big party each year.", "response": "My birth date is January 1, 1990, and I usually celebrate my birthday with a big party each year."},
    {"prompt": "She was born on 07/15/1985, and we always throw a surprise party for her every summer.", "response": "She was born on July 15, 1985, and we always throw a surprise party for her every summer."},

    # IP address
    {"prompt": "I just checked my IP address and it’s 192.168.1.1, and it works fine for my home network.", "response": "I just checked my IP address, and it’s 192.168.1.1; it works fine for my home network."},
    {"prompt": "His private IP address is 10.0.0.1, and he has trouble connecting his new printer to it.", "response": "His private IP address is 10.0.0.1, and he has trouble connecting his new printer to it."},

    # Phone numbers and additional info
    {"prompt": "Call me at (555) 123-4567 anytime, I'm always happy to help.", "response": "Call me at (555) 123-4567 anytime; I'm always happy to help."},
    {"prompt": "If you need to reach my brother, his number is 123-456-7890, but he's often busy with work.", "response": "If you need to reach my brother, his number is 123-456-7890, but he's often busy with work."},

    # More grammar and typo corrections
    {"prompt": "I seen the movie last night, it was more exciting then I expected.", "response": "I saw the movie last night; it was more exciting than I expected."},
    {"prompt": "They don’t know where she has gone, it’s really strange of her to leave without telling anyone.", "response": "They don’t know where she has gone; it’s really strange of her to leave without telling anyone."},
    {"prompt": "Its crucial to keep practicing if you want to improve your skills in any field.", "response": "It's crucial to keep practicing if you want to improve your skills in any field."},
    {"prompt": "I would of gone to the event if I had known about it earlier, but I missed the details.", "response": "I would have gone to the event if I had known about it earlier, but I missed the details."},
    {"prompt": "The weather was nice, but I left my umbrella at home because I thought it wouldn't rain, and I got soaked.", "response": "The weather was nice, but I left my umbrella at home because I thought it wouldn't rain, and I got soaked."},

    # More personal information examples
    {"prompt": "Feel free to send me an email at lara.jameson@example.com, and I’ll get back to you shortly.", "response": "Feel free to send me an email at lara.jameson@example.com, and I’ll get back to you shortly."},
    {"prompt": "My MasterCard number is 6011-0009-1234-5678, which I use regularly for my online shopping needs.", "response": "My MasterCard number is 6011-0009-1234-5678, which I use regularly for my online shopping needs."},
    {"prompt": "I currently reside at 321 Cedar Lane, 43210 City, where the community is quite welcoming.", "response": "I currently reside at 321 Cedar Lane, 43210 City, where the community is quite welcoming."},
    {"prompt": "My DOB is 04-20-1988, and I celebrate it every year with my family and close friends.", "response": "My DOB is April 20, 1988, and I celebrate it every year with my family and close friends."},
    {"prompt": "My IP address is 172.16.0.10, and I need to configure my router settings to improve the connection.", "response": "My IP address is 172.16.0.10, and I need to configure my router settings to improve the connection."},
    {"prompt": "She has a BSN that starts with 567-89-0123, and she needs it for her medical insurance application.", "response": "She has a BSN that starts with 567-89-0123, and she needs it for her medical insurance application."},

    # Additional entries with typos
    {"prompt": "I have a new email, its emily.thompson@domain.com, and I forgot my old one.", "response": "I have a new email; it’s emily.thompson@domain.com, and I forgot my old one."},
    {"prompt": "He told me his MasterCard is ending with 4321, and it’s always worked well for his purchases.", "response": "He told me his MasterCard ends with 4321, and it’s always worked well for his purchases."},
    {"prompt": "My address is 111 Cherry Lane, 65432 Town, and I love the park near my home.", "response": "My address is 111 Cherry Lane, 65432 Town, and I love the park near my home."},
    {"prompt": "I was born on 03-03-1992, and I like to spend that day with my loved ones, making it special.", "response": "I was born on March 3, 1992, and I like to spend that day with my loved ones, making it special."},
    {"prompt": "He has an IP address of 10.0.0.2, and he keeps forgetting how to set up his devices properly.", "response": "He has an IP address of 10.0.0.2, and he keeps forgetting how to set up his devices properly."},
# Personal names and general info
    {"prompt": "My name is Michael, but my friends call me Mike; we usually go for hikes together every weekend.", "response": "My name is Michael, but my friends call me Mike. We usually go for hikes together every weekend."},
    {"prompt": "You can reach me at jessica.smith@email.com, and I check my emails multiple times a day for any updates.", "response": "You can reach me at jessica.smith@email.com, and I check my emails multiple times a day for any updates."},

    # BSN
    {"prompt": "I received my BSN number which is 123456789, and I need to provide it to the HR department for my job.", "response": "I received my BSN number, which is 123-45-6789, and I need to provide it to the HR department for my job."},
    {"prompt": "His BSN is 987654321, and it is necessary for his health insurance application to be processed.", "response": "His BSN is 987-65-4321, and it is necessary for his health insurance application to be processed."},

    # Credit card info
    {"prompt": "I have a Visa credit card that ends in 4321, which I use frequently for my online purchases.", "response": "I have a Visa credit card that ends in 4321, which I use frequently for my online purchases."},
    {"prompt": "My MasterCard number is 5111-1111-1111-1111, and I find it very useful for making transactions abroad.", "response": "My MasterCard number is 5111-1111-1111-1111, and I find it very useful for making transactions abroad."},

    # IBAN
    {"prompt": "I recently opened a bank account and my IBAN is GB29NWBK60161331926819, which is required for international transfers.", "response": "I recently opened a bank account, and my IBAN is GB29 NWBK 6016 1331 9268 19, which is required for international transfers."},
    {"prompt": "My IBAN for the savings account is FR76 3000 6000 0112 3456 7890 123, making it easier for me to manage my finances.", "response": "My IBAN for the savings account is FR76 3000 6000 0112 3456 7890 123, making it easier for me to manage my finances."},

    # Address and ZIP code
    {"prompt": "I reside at 890 Maple Drive, Springfield, with the ZIP code 54321, and it's a cozy neighborhood with friendly people.", "response": "I reside at 890 Maple Drive, Springfield, with the ZIP code 54321, and it's a cozy neighborhood with friendly people."},
    {"prompt": "My current address is 321 Pine Street, 67890 Village, and I enjoy living there because of the nearby parks.", "response": "My current address is 321 Pine Street, 67890 Village, and I enjoy living there because of the nearby parks."},

    # Date of Birth (DOB)
    {"prompt": "I was born on 02-02-1980, and every year I celebrate it with a big cake and lots of friends.", "response": "I was born on February 2, 1980, and every year I celebrate it with a big cake and lots of friends."},
    {"prompt": "Her birthday is 08/22/1995, and we always plan a fun surprise for her to make it special.", "response": "Her birthday is August 22, 1995, and we always plan a fun surprise for her to make it special."},

    # IP address
    {"prompt": "I found my IP address today, it’s 192.168.0.1, and it works perfectly fine for my network.", "response": "I found my IP address today; it’s 192.168.0.1, and it works perfectly fine for my network."},
    {"prompt": "His private IP address is 172.16.0.10, which he uses to access his home devices remotely.", "response": "His private IP address is 172.16.0.10, which he uses to access his home devices remotely."},

    # Phone numbers and additional info
    {"prompt": "If you want to contact me, my phone number is (555) 234-5678, and I’m usually available in the evenings.", "response": "If you want to contact me, my phone number is (555) 234-5678, and I’m usually available in the evenings."},
    {"prompt": "Her number is 321-654-9870, and she often misses calls due to her busy schedule.", "response": "Her number is 321-654-9870, and she often misses calls due to her busy schedule."},

    # More grammar and typo corrections
    {"prompt": "I seen your message yesterday, but I forgot to reply, so I’m doing it now.", "response": "I saw your message yesterday, but I forgot to reply, so I’m doing it now."},
    {"prompt": "They don't know where she had went, it’s quite unusual for her to not inform anyone.", "response": "They don't know where she had gone; it’s quite unusual for her to not inform anyone."},
    {"prompt": "Its important to save money for the future, because you never know when you might need it.", "response": "It's important to save money for the future because you never know when you might need it."},
    {"prompt": "I could of joined the team, but I had other commitments that I couldn’t change.", "response": "I could have joined the team, but I had other commitments that I couldn’t change."},
    {"prompt": "The sun was shining brightly, so I decided to take a walk outside, but I forgot my hat and got sunburned.", "response": "The sun was shining brightly, so I decided to take a walk outside, but I forgot my hat and got sunburned."},

    # More personal information examples
    {"prompt": "Feel free to email me at sarah.connor@example.com, and I promise to respond as soon as I can.", "response": "Feel free to email me at sarah.connor@example.com, and I promise to respond as soon as I can."},
    {"prompt": "My MasterCard number is 4024-0071-1244-0000, which I use frequently for online transactions.", "response": "My MasterCard number is 4024-0071-1244-0000, which I use frequently for online transactions."},
    {"prompt": "I live at 555 Willow Lane, 98765 Town, and I enjoy the quiet atmosphere around my area.", "response": "I live at 555 Willow Lane, 98765 Town, and I enjoy the quiet atmosphere around my area."},
    {"prompt": "My birthday is 11-11-1993, and I usually celebrate it with a small gathering of family and friends.", "response": "My birthday is November 11, 1993, and I usually celebrate it with a small gathering of family and friends."},
    {"prompt": "My IP address is 192.0.2.1, and I need to check the settings on my router for better connectivity.", "response": "My IP address is 192.0.2.1, and I need to check the settings on my router for better connectivity."},

    # Additional entries with typos
    {"prompt": "I’ve recently changed my email to tom.brown@email.com because my old one was compromised.", "response": "I’ve recently changed my email to tom.brown@email.com because my old one was compromised."},
    {"prompt": "He told me his MasterCard ends in 5678, and it’s his go-to card for most of his expenses.", "response": "He told me his MasterCard ends in 5678, and it’s his go-to card for most of his expenses."},
    {"prompt": "My address is 777 Elm Road, 24680 City, and I love living there because of the friendly neighbors.", "response": "My address is 777 Elm Road, 24680 City, and I love living there because of the friendly neighbors."},
    {"prompt": "I was born on 05-05-1991, and I usually throw a small party to celebrate it every year.", "response": "I was born on May 5, 1991, and I usually throw a small party to celebrate it every year."},
    {"prompt": "His private IP is 10.0.0.5, and he often has trouble with his internet connection at home.", "response": "His private IP is 10.0.0.5, and he often has trouble with his internet connection at home."},

    # Personal names and general info with typos and lowercase
    {"prompt": "my name is emily, but my friends call me em. we often go to the cafe together every friday.", "response": "My name is Emily, but my friends call me Em. We often go to the cafe together every Friday."},
    {"prompt": "you can reach me at john.doe@email.com, and i check my emails daily for any news.", "response": "You can reach me at john.doe@email.com, and I check my emails daily for any news."},

    # BSN
    {"prompt": "i received my bsn number which is 123456789, and i need to provide it to the hr department for my job.", "response": "I received my BSN number, which is 123-45-6789, and I need to provide it to the HR department for my job."},
    {"prompt": "his bsn is 987654321, and it is needed for his health insurance application to be processed.", "response": "His BSN is 987-65-4321, and it is needed for his health insurance application to be processed."},

    # Credit card info
    {"prompt": "i have a visa credit card that ends in 4321, which i use frequently for online shopping.", "response": "I have a Visa credit card that ends in 4321, which I use frequently for online shopping."},
    {"prompt": "my mastercard number is 5111-1111-1111-1111, and i find it very useful for making purchases abroad.", "response": "My MasterCard number is 5111-1111-1111-1111, and I find it very useful for making purchases abroad."},

    # IBAN
    {"prompt": "i recently opened a bank account and my iban is gb29nwbk60161331926819, which is required for international transfers.", "response": "I recently opened a bank account, and my IBAN is GB29 NWBK 6016 1331 9268 19, which is required for international transfers."},
    {"prompt": "my iban for the savings account is fr76 3000 6000 0112 3456 7890 123, making it easier for me to manage my finances.", "response": "My IBAN for the savings account is FR76 3000 6000 0112 3456 7890 123, making it easier for me to manage my finances."},

    # Address and ZIP code
    {"prompt": "i live at 890 maple drive, springfield, with the zip code 54321, and it's a friendly neighborhood.", "response": "I live at 890 Maple Drive, Springfield, with the ZIP code 54321, and it's a friendly neighborhood."},
    {"prompt": "my current address is 321 pine street, 67890 village, and i enjoy living there because of the parks nearby.", "response": "My current address is 321 Pine Street, 67890 Village, and I enjoy living there because of the parks nearby."},

    # Date of Birth (DOB)
    {"prompt": "i was born on 02-02-1980, and i celebrate it with a big cake and friends every year.", "response": "I was born on February 2, 1980, and I celebrate it with a big cake and friends every year."},
    {"prompt": "her birthday is 08/22/1995, and we always plan a fun surprise for her to make it special.", "response": "Her birthday is August 22, 1995, and we always plan a fun surprise for her to make it special."},

    # IP address
    {"prompt": "i found my ip address today, it’s 192.168.0.1, and it works fine for my network.", "response": "I found my IP address today; it’s 192.168.0.1, and it works fine for my network."},
    {"prompt": "his private ip address is 172.16.0.10, which he uses to access his home devices remotely.", "response": "His private IP address is 172.16.0.10, which he uses to access his home devices remotely."},

    # Phone numbers and additional info
    {"prompt": "if you want to contact me, my phone number is (555) 234-5678, and i’m usually available in the evenings.", "response": "If you want to contact me, my phone number is (555) 234-5678, and I’m usually available in the evenings."},
    {"prompt": "her number is 321-654-9870, and she often misses calls because of her busy schedule.", "response": "Her number is 321-654-9870, and she often misses calls because of her busy schedule."},

    # More grammar and typo corrections
    {"prompt": "i seen your message yesterday, but i forgot to reply, so i’m doing it now.", "response": "I saw your message yesterday, but I forgot to reply, so I’m doing it now."},
    {"prompt": "they don't know where she had went, it’s quite unusual for her to not inform anyone.", "response": "They don't know where she had gone; it’s quite unusual for her to not inform anyone."},
    {"prompt": "its important to save money for the future, because you never know when you might need it.", "response": "It's important to save money for the future because you never know when you might need it."},
    {"prompt": "i could of joined the team, but i had other commitments that i couldn’t change.", "response": "I could have joined the team, but I had other commitments that I couldn’t change."},
    {"prompt": "the sun was shining bright, so i decided to take a walk outside, but i forgot my hat and got sunburned.", "response": "The sun was shining brightly, so I decided to take a walk outside, but I forgot my hat and got sunburned."},

    # More personal information examples
    {"prompt": "feel free to email me at sarah.connor@example.com, and i promise to respond as soon as i can.", "response": "Feel free to email me at sarah.connor@example.com, and I promise to respond as soon as I can."},
    {"prompt": "my mastercard number is 4024-0071-1244-0000, which i use frequently for online transactions.", "response": "My MasterCard number is 4024-0071-1244-0000, which I use frequently for online transactions."},
    {"prompt": "i live at 555 willow lane, 98765 town, and i love living there because of the friendly neighbors.", "response": "I live at 555 Willow Lane, 98765 Town, and I love living there because of the friendly neighbors."},
    {"prompt": "my birthday is 11-11-1993, and i usually celebrate it with a small gathering of family and friends.", "response": "My birthday is November 11, 1993, and I usually celebrate it with a small gathering of family and friends."},
    {"prompt": "my ip address is 192.0.2.1, and i need to check the settings on my router for better connectivity.", "response": "My IP address is 192.0.2.1, and I need to check the settings on my router for better connectivity."},

    # Additional entries with typos and lowercase
    {"prompt": "i’ve recently changed my email to tom.brown@email.com because my old one was compromised.", "response": "I’ve recently changed my email to tom.brown@email.com because my old one was compromised."},
    {"prompt": "he told me his mastercard ends in 5678, and it’s his go-to card for most of his expenses.", "response": "He told me his MasterCard ends in 5678, and it’s his go-to card for most of his expenses."},
    {"prompt": "my address is 777 elm road, 24680 city, and i enjoy the quiet atmosphere around my area.", "response": "My address is 777 Elm Road, 24680 City, and I enjoy the quiet atmosphere around my area."},
    {"prompt": "i was born on 05-05-1991, and i usually throw a small party to celebrate it every year.", "response": "I was born on May 5, 1991, and I usually throw a small party to celebrate it every year."},
    {"prompt": "his private ip is 10.0.0.5, and he often has trouble with his internet connection at home.", "response": "His private IP is 10.0.0.5, and he often has trouble with his internet connection at home."},
# More personal names and general info with typos and lowercase
    {"prompt": "my name is olivia, and my email is olivia.jameson@example.com, but everyone calls me liv.", "response": "My name is Olivia, and my email is olivia.jameson@example.com, but everyone calls me Liv."},
    {"prompt": "my brother's name is austin, he loves to play soccer every weekend and he’s very good at it.", "response": "My brother's name is Austin; he loves to play soccer every weekend, and he’s very good at it."},

    # BSN
    {"prompt": "my bsn number is 246813579, which i need for my job application at the local bank.", "response": "My BSN number is 246-81-3579, which I need for my job application at the local bank."},
    {"prompt": "i got my bsn recently, it's 135792468, and i have to provide it for tax purposes.", "response": "I got my BSN recently; it's 135-79-2468, and I have to provide it for tax purposes."},

    # Credit card info
    {"prompt": "i use my mastercard with the last four digits 7890 for all my online purchases.", "response": "I use my MasterCard with the last four digits 7890 for all my online purchases."},
    {"prompt": "i have a visa card ending in 5678, and it's really convenient for traveling.", "response": "I have a Visa card ending in 5678, and it's really convenient for traveling."},

    # IBAN
    {"prompt": "my iban for my main account is de89 3704 0044 0532 0130 00, which i use for direct deposits.", "response": "My IBAN for my main account is DE89 3704 0044 0532 0130 00, which I use for direct deposits."},
    {"prompt": "i need to provide my iban, which is nl13 abna 0123 4567 89, for the international money transfer.", "response": "I need to provide my IBAN, which is NL13 ABNA 0123 4567 89, for the international money transfer."},

    # Address and ZIP code
    {"prompt": "i live on 456 oak avenue, hilltown, with zip code 11223, and the community is very welcoming.", "response": "I live on 456 Oak Avenue, Hilltown, with ZIP code 11223, and the community is very welcoming."},
    {"prompt": "my address is 1234 birch boulevard, 65432 lakeside, and it’s very peaceful around here.", "response": "My address is 1234 Birch Boulevard, 65432 Lakeside, and it’s very peaceful around here."},

    # Date of Birth (DOB)
    {"prompt": "i was born on 01-15-1990, and i always enjoy celebrating my birthday with friends.", "response": "I was born on January 15, 1990, and I always enjoy celebrating my birthday with friends."},
    {"prompt": "her birthday is 03/30/1995, and she loves to have a big party every year.", "response": "Her birthday is March 30, 1995, and she loves to have a big party every year."},

    # IP address
    {"prompt": "my home ip address is 10.1.1.1, which i often check when i have internet issues.", "response": "My home IP address is 10.1.1.1, which I often check when I have internet issues."},
    {"prompt": "his local ip address is 192.168.1.10, and he needs it to access the router settings.", "response": "His local IP address is 192.168.1.10, and he needs it to access the router settings."},

    # Phone numbers and additional info
    {"prompt": "you can reach me at (555) 987-6543, but sometimes i might be busy with work.", "response": "You can reach me at (555) 987-6543, but sometimes I might be busy with work."},
    {"prompt": "her contact number is 654-321-0987, and she usually replies to messages quickly.", "response": "Her contact number is 654-321-0987, and she usually replies to messages quickly."},

    # More grammar and typo corrections
    {"prompt": "i forgot to send you the document that you asked for, it was lying in my drafts.", "response": "I forgot to send you the document that you asked for; it was lying in my drafts."},
    {"prompt": "she didn’t knew that the meeting was rescheduled, so she showed up at the wrong time.", "response": "She didn’t know that the meeting was rescheduled, so she showed up at the wrong time."},
    {"prompt": "its really nice to see you again after such a long time, we should catch up soon.", "response": "It's really nice to see you again after such a long time; we should catch up soon."},
    {"prompt": "i could of helped if i knew about the issue earlier, but i wasn’t informed.", "response": "I could have helped if I knew about the issue earlier, but I wasn’t informed."},
    {"prompt": "the weather was so nice yesterday, we decided to have a picnic in the park but forgot the food.", "response": "The weather was so nice yesterday that we decided to have a picnic in the park, but we forgot the food."},

    # More personal information examples
    {"prompt": "my email is charles.brown@email.com, and i check it regularly for work updates.", "response": "My email is charles.brown@email.com, and I check it regularly for work updates."},
    {"prompt": "my credit card is a visa and ends in 4321, which is my go-to for all my shopping needs.", "response": "My credit card is a Visa and ends in 4321, which is my go-to for all my shopping needs."},
    {"prompt": "i reside at 8900 cherry lane, 13579 town, and i appreciate the scenic views around the area.", "response": "I reside at 8900 Cherry Lane, 13579 Town, and I appreciate the scenic views around the area."},
    {"prompt": "my date of birth is 12-12-1985, and i plan to celebrate it with a family dinner.", "response": "My date of birth is December 12, 1985, and I plan to celebrate it with a family dinner."},
    {"prompt": "i found my ip address to be 203.0.113.1, and it's essential for my network setup.", "response": "I found my IP address to be 203.0.113.1, and it's essential for my network setup."},

    # Additional entries with typos and lowercase
    {"prompt": "i’m changing my email to amy.smith@email.com for better security and privacy.", "response": "I’m changing my email to amy.smith@email.com for better security and privacy."},
    {"prompt": "my mastercard number is 6011-1111-1111-1111, and i use it for all my purchases online.", "response": "My MasterCard number is 6011-1111-1111-1111, and I use it for all my purchases online."},
    {"prompt": "i live at 3333 ash street, 54321 city, and it’s a quiet place to relax after a busy day.", "response": "I live at 3333 Ash Street, 54321 City, and it’s a quiet place to relax after a busy day."},
    {"prompt": "my birthday falls on 07-07-1992, and every year, i look forward to having cake and ice cream.", "response": "My birthday falls on July 7, 1992, and every year, I look forward to having cake and ice cream."},
    {"prompt": "my local ip address is 192.168.0.10, which is useful for connecting to my home network.", "response": "My local IP address is 192.168.0.10, which is useful for connecting to my home network."},
    
     {"prompt": "my name is sasha and my email is sasha.connor@email.com, im very excited about my upcoming trip.", "response": "My name is Sasha, and my email is sasha.connor@email.com. I'm very excited about my upcoming trip."},
    {"prompt": "my cousin's name is nikki, she loves painting and often participates in local art shows.", "response": "My cousin's name is Nikki; she loves painting and often participates in local art shows."},

    # BSN
    {"prompt": "i have a bsn number 987654321, and its needed for my registration at the new job.", "response": "I have a BSN number 987-65-4321, and it's needed for my registration at the new job."},
    {"prompt": "my bsn is 246810121, and i need it for filing my taxes this year.", "response": "My BSN is 246-81-0121, and I need it for filing my taxes this year."},

    # Credit card info
    {"prompt": "i use my mastercard ending in 3456 for my online shopping and subscription services.", "response": "I use my MasterCard ending in 3456 for my online shopping and subscription services."},
    {"prompt": "my visa card is 4111-1111-1111-1111, and it helps me manage my expenses effectively.", "response": "My Visa card is 4111-1111-1111-1111, and it helps me manage my expenses effectively."},

    # IBAN
    {"prompt": "my iban is gb29 nwbk 6016 1331 9268 19, and i need it for my international payments.", "response": "My IBAN is GB29 NWBK 6016 1331 9268 19, and I need it for my international payments."},
    {"prompt": "i have my iban ready, it is fr76 3000 6000 0112 3456 7890 123, for the fund transfer.", "response": "I have my IBAN ready; it is FR76 3000 6000 0112 3456 7890 123 for the fund transfer."},

    # Address and ZIP code
    {"prompt": "i live at 789 maple street, springfield, 12345, and i love the local parks here.", "response": "I live at 789 Maple Street, Springfield, 12345, and I love the local parks here."},
    {"prompt": "my address is 567 elm drive, riverside, 67890, where the community is very friendly.", "response": "My address is 567 Elm Drive, Riverside, 67890, where the community is very friendly."},

    # Date of Birth (DOB)
    {"prompt": "my birthday is on 11/20/1988, and i usually celebrate it with a big family gathering.", "response": "My birthday is on November 20, 1988, and I usually celebrate it with a big family gathering."},
    {"prompt": "i was born on 02-02-1992, and my friends always throw a surprise party for me.", "response": "I was born on February 2, 1992, and my friends always throw a surprise party for me."},

    # IP address
    {"prompt": "my home ip address is 192.0.2.1, and i check it whenever im troubleshooting my network.", "response": "My home IP address is 192.0.2.1, and I check it whenever I'm troubleshooting my network."},
    {"prompt": "his local ip is 10.0.0.5, which is important for accessing the printer at home.", "response": "His local IP is 10.0.0.5, which is important for accessing the printer at home."},

    # Phone numbers and additional info
    {"prompt": "you can call me at (123) 456-7890, but i might be unavailable during the day.", "response": "You can call me at (123) 456-7890, but I might be unavailable during the day."},
    {"prompt": "my number is 987-654-3210, and i always try to answer important calls.", "response": "My number is 987-654-3210, and I always try to answer important calls."},

    # More grammar and typo corrections
    {"prompt": "i seen the movie last night, and it was really exciting with a lot of twists.", "response": "I saw the movie last night, and it was really exciting with a lot of twists."},
    {"prompt": "the team has worked really hard, and they deserves a break after the project completion.", "response": "The team has worked really hard, and they deserve a break after the project completion."},
    {"prompt": "i am not sure where is the nearest coffee shop, can you help me find it?", "response": "I am not sure where the nearest coffee shop is. Can you help me find it?"},

    # More personal information examples
    {"prompt": "my email is emily.jones@email.com, and i use it for both work and personal matters.", "response": "My email is emily.jones@email.com, and I use it for both work and personal matters."},
    {"prompt": "my credit card number is 5105-1051-0510-5100, and it’s for my everyday expenses.", "response": "My credit card number is 5105-1051-0510-5100, and it’s for my everyday expenses."},
    {"prompt": "i live on 2222 cherry blossom lane, 98765 town, and it's a beautiful area with flowers.", "response": "I live on 2222 Cherry Blossom Lane, 98765 Town, and it's a beautiful area with flowers."},
    {"prompt": "my birthday is on 04-04-1995, and i plan to go out with friends for dinner.", "response": "My birthday is on April 4, 1995, and I plan to go out with friends for dinner."},
    {"prompt": "my ip address is 203.0.113.5, which i use to connect to the internet.", "response": "My IP address is 203.0.113.5, which I use to connect to the internet."},

    # Additional entries with typos and lowercase
    {"prompt": "im updating my email to david.green@email.com for better communication.", "response": "I'm updating my email to david.green@email.com for better communication."},
    {"prompt": "my mastercard is 6011-0000-0000-0000, and i rely on it for most of my transactions.", "response": "My MasterCard is 6011-0000-0000-0000, and I rely on it for most of my transactions."},
    {"prompt": "i stay at 4444 jasmine way, 13579 city, and the neighbors are quite friendly.", "response": "I stay at 4444 Jasmine Way, 13579 City, and the neighbors are quite friendly."},
    {"prompt": "i was born on 09-09-1980, and i enjoy celebrating my birthday every year.", "response": "I was born on September 9, 1980, and I enjoy celebrating my birthday every year."},
    {"prompt": "my local ip address is 192.168.1.1, and it's essential for my home network.", "response": "My local IP address is 192.168.1.1, and it's essential for my home network."},
    
     # New entries with various personal details and errors
    {"prompt": "my name is jessica and my email is jessica.smith@email.com, i really like to travel during the summer.", "response": "My name is Jessica, and my email is jessica.smith@email.com. I really like to travel during the summer."},
    {"prompt": "my brother is named max and he loves video games, especially rpgs.", "response": "My brother is named Max, and he loves video games, especially RPGs."},

    # BSN numbers
    {"prompt": "i just got my bsn number which is 123456789, and it feels great to have it.", "response": "I just got my BSN number, which is 123-45-6789, and it feels great to have it."},
    {"prompt": "the bsn number 112233445 is needed for my health insurance application.", "response": "The BSN number 112-23-3445 is needed for my health insurance application."},

    # Credit card info
    {"prompt": "my mastercard number is 6011-0000-0000-0000, and i use it for my online purchases.", "response": "My MasterCard number is 6011-0000-0000-0000, and I use it for my online purchases."},
    {"prompt": "i have a visa card that ends in 1234, it is very convenient for my travels.", "response": "I have a Visa card that ends in 1234; it is very convenient for my travels."},

    # IBAN numbers
    {"prompt": "my iban for international transfers is gb29 nwbk 6016 1331 9268 19, very important for my payments.", "response": "My IBAN for international transfers is GB29 NWBK 6016 1331 9268 19; very important for my payments."},
    {"prompt": "the iban fr76 3000 6000 0112 3456 7890 123 is for my savings account.", "response": "The IBAN FR76 3000 6000 0112 3456 7890 123 is for my savings account."},

    # Address and ZIP code
    {"prompt": "i currently reside at 1000 oak street, los angeles, 90001, and it's a vibrant city.", "response": "I currently reside at 1000 Oak Street, Los Angeles, 90001, and it's a vibrant city."},
    {"prompt": "my address is 1234 pines avenue, orlando, 32801, and the weather here is nice.", "response": "My address is 1234 Pines Avenue, Orlando, 32801, and the weather here is nice."},

    # Date of Birth (DOB)
    {"prompt": "my birthdate is 03/15/1990, i always throw a big party for it.", "response": "My birthdate is March 15, 1990, and I always throw a big party for it."},
    {"prompt": "i was born on 07-20-1985, and i plan a family reunion each year to celebrate.", "response": "I was born on July 20, 1985, and I plan a family reunion each year to celebrate."},

    # IP address
    {"prompt": "my home ip address is 192.168.0.1, and i reset it whenever i have issues.", "response": "My home IP address is 192.168.0.1, and I reset it whenever I have issues."},
    {"prompt": "the ip for my computer is 10.0.0.3, which is important for my work.", "response": "The IP for my computer is 10.0.0.3, which is important for my work."},

    # Phone numbers and additional info
    {"prompt": "you can reach me at 555-123-4567, but im usually busy during the day.", "response": "You can reach me at 555-123-4567, but I'm usually busy during the day."},
    {"prompt": "my cell number is (321) 654-0987, and i try to answer it as much as possible.", "response": "My cell number is (321) 654-0987, and I try to answer it as much as possible."},

    # Grammar and typo corrections
    {"prompt": "the team has done a great job, they deserve a round of aplause for their hard work.", "response": "The team has done a great job; they deserve a round of applause for their hard work."},
    {"prompt": "im looking for a nice cafe nearby, can you suggest one for me?", "response": "I'm looking for a nice cafe nearby. Can you suggest one for me?"},

    # Additional personal information
    {"prompt": "my new email is kevin.brown@email.com for all official communications.", "response": "My new email is kevin.brown@email.com for all official communications."},
    {"prompt": "i have a credit card with number 4916-1234-5678-9012, which i use often.", "response": "I have a credit card with number 4916-1234-5678-9012, which I use often."},
    {"prompt": "i live at 8888 willow way, 24680 city, where the sunsets are beautiful.", "response": "I live at 8888 Willow Way, 24680 City, where the sunsets are beautiful."},
    {"prompt": "my date of birth is 12-12-1992, and every year, i enjoy celebrating it with my friends.", "response": "My date of birth is December 12, 1992, and every year, I enjoy celebrating it with my friends."},
    {"prompt": "my ip address is 203.0.113.10, which i always check when there's a network issue.", "response": "My IP address is 203.0.113.10, which I always check when there's a network issue."},

    # Additional entries with errors
    {"prompt": "my name is julia, and my email is julia.roberts@email.com for any inquiries.", "response": "My name is Julia, and my email is julia.roberts@email.com for any inquiries."},
    {"prompt": "i use my mastercard ending in 8888 for all my purchases, especially online.", "response": "I use my MasterCard ending in 8888 for all my purchases, especially online."},
    {"prompt": "my address is 5555 tulip drive, 78901 town, which is a nice neighborhood.", "response": "My address is 5555 Tulip Drive, 78901 Town, which is a nice neighborhood."},
    {"prompt": "my birth date is 01-01-1995, and i always have a small get-together to celebrate.", "response": "My birth date is January 1, 1995, and I always have a small get-together to celebrate."},
    {"prompt": "my local ip address is 192.168.1.100, which is critical for my home network setup.", "response": "My local IP address is 192.168.1.100, which is critical for my home network setup."},
]


# Converting to dataset format with input, instruction, and output
formatted_data = [
    {
        "instruction": "Grammar correction",
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
json_file_path = "datasets/blueVi-GPT-dataset-grammar-correction.json"
dataset.to_json(json_file_path)

# Create the Hugging Face dataset repo if it doesn't exist
repo_name = "ThanhTranVisma/blueVi-GPT-dataset-grammar-correction"
create_repo(repo_name, repo_type="dataset", private=True)

# Push JSON file to Hugging Face Hub
api = HfApi()
api.upload_file(
    path_or_fileobj=json_file_path,
    path_in_repo="blueVi-GPT-dataset-grammar-correction.json",
    repo_id=repo_name,
    repo_type="dataset",
    token=hf_token,
)

print(f"Dataset successfully uploaded to {repo_name}!")