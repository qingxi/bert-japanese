from transformers import BertTokenizer, BertForQuestionAnswering
import torch

tokenizer = BertTokenizer.from_pretrained('../NICT_BERT-base_JapaneseWikipedia_100K')
model = BertForQuestionAnswering.from_pretrained('../NICT_BERT-base_JapaneseWikipedia_100K')

#question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
question, text = "金融の同義語は何ですか？", "闇金融"
input_ids = tokenizer.encode(question, text)
token_type_ids = 1 # [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
print(answer)