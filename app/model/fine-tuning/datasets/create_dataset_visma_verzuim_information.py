from datasets import Dataset
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()
hf_token = os.getenv("HUGGING_FACE_TOKEN")

# Sample data
data = [
    {"prompt": "What is Visma Verzuim?", "response": "Visma Verzuim is the online application for managing absence in the organization. With this user-friendly absence software you can easily collaborate with all parties involved, such as the employee, occupational health service, company doctor, insurer and the UWV. Visma Verzuim is the market leader in the Netherlands. In Visma Verzuim, the absence for approximately 3.5 million employees in the Netherlands is managed daily."},
    {"prompt": "About Visma Verzuim", "response": "Visma Verzuim is the online application for managing absence in the organization. With this user-friendly absence software you can easily collaborate with all parties involved, such as the employee, occupational health service, company doctor, insurer and the UWV. Visma Verzuim is the market leader in the Netherlands. In Visma Verzuim, the absence for approximately 3.5 million employees in the Netherlands is managed daily."},
    {"prompt": "What is Visma", "response": "Visma is a leading European cloud software company that helps people and organizations improve every day. Visma provides software that simplifies and digitalizes business processes in the private and public sector. Originally Norwegian, but with a presence in the entire Nordic region together with the Benelux, Central and Eastern Europe, they are one of Europe's leading software companies."},
    {"prompt": "Over Visma Verzuim", "response": "Visma Verzuim is de online applicatie voor het managen van verzuim in de organisatie. Met deze gebruiksvriendelijke verzuimsoftware kun je eenvoudig samenwerken met alle partijen die betrokken zijn, zoals de werknemer, arbodienst, bedrijfsarts, verzekeraar en het UWV. Visma Verzuim is marktleider in Nederland. In Visma Verzuim wordt dagelijks het verzuim voor circa 3,5 miljoen werknemers in Nederland beheerd."},
    {"prompt": "Over Visma", "response": "Visma is een toonaangevend Europees cloudsoftwarebedrijf dat mensen en organisaties helpt om elke dag te verbeteren. Visma levert software die bedrijfsprocessen in de private en publieke sector vereenvoudigt en digitaliseert. Van origine Noors, maar dankzij de aanwezigheid in de hele Scandinavische regio samen met de Benelux, Centraal- en Oost-Europa, zijn ze een van Europa’s toonaangevende softwarebedrijven."},
    {"prompt": "What is BlueVi", "response": "The latest user-friendly absenteeism solution for managing short and long-term absenteeism by Visma Verzuim"},   
]

# Converting to dataset format with input, instruction, and output
formatted_data = [
    {
        "instruction": "Answer the question",
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
json_file_path = "datasets/blueVi-GPT-dataset-answer-question-visma-verzuim.json"
dataset.to_json(json_file_path)

# Create the Hugging Face dataset repo if it doesn't exist
repo_name = "ThanhTranVisma/blueVi-GPT-dataset-answer-question-visma-verzuim"
create_repo(repo_name, repo_type="dataset", private=False)

# Push JSON file to Hugging Face Hub
api = HfApi()
api.upload_file(
    path_or_fileobj=json_file_path,
    path_in_repo="blueVi-GPT-dataset-answer-question-visma-verzuim.json",
    repo_id=repo_name,
    repo_type="dataset",
    token=hf_token,
)

print(f"Dataset successfully uploaded to {repo_name}!")