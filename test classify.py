from transformers import BertTokenizer, BertForTokenClassification
import torch

tokenizer = BertTokenizer.from_pretrained('../NICT_BERT-base_JapaneseWikipedia_100K')
model = BertForTokenClassification.from_pretrained('../NICT_BERT-base_JapaneseWikipedia_100K')

input_ids = torch.tensor(tokenizer.encode("金融", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=labels)

loss, scores = outputs[:2]

print(outputs)