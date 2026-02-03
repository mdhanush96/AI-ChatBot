from transformers import AutoTokenizer, AutoModel

model_name = "dmis-lab/biobert-base-cased-v1.1"

AutoTokenizer.from_pretrained(model_name)
AutoModel.from_pretrained(model_name)

print("BioBERT downloaded successfully!")
