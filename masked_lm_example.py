# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
from transformers.modeling_bert import BertForMaskedLM


# %%
tokenizer = BertJapaneseTokenizer.from_pretrained('../BERT-base_mecab-ipadic-bpe-32k_whole-word-mask')


# %%
model = BertForMaskedLM.from_pretrained('../BERT-base_mecab-ipadic-bpe-32k_whole-word-mask')


# %%
input_ids = tokenizer.encode(f'''青葉山で{tokenizer.mask_token}の研究をしています。''', return_tensors='pt')


# %%
print(input_ids)


# %%
print(tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))


# %%
masked_values = torch.where(input_ids == tokenizer.mask_token_id)
masked_index = masked_values[1].tolist()[0]
print(masked_index)


# %%
result = model(input_ids)
pred_ids = result[0][:, masked_index].topk(10).indices.tolist()[0]
for pred_id in pred_ids:
    output_ids = input_ids.tolist()[0]
    output_ids[masked_index] = pred_id
    print(tokenizer.decode(output_ids))

