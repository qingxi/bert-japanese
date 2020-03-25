# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
from transformers.modeling_bert import BertForMaskedLM
from transformers.modeling_bert import BertConfig

# %%
tokenizer = BertJapaneseTokenizer.from_pretrained('../BERT-base_mecab-ipadic-bpe-32k_whole-word-mask')


# %%
model = BertForMaskedLM.from_pretrained('../BERT-base_mecab-ipadic-bpe-32k_whole-word-mask')

# %%
#input_ids = tokenizer.encode(f'''金融法に関連する同義語は{tokenizer.mask_token}。''', return_tensors='pt')
input_ids = tokenizer.encode(f'''緊急''', return_tensors='pt')


# %%
print(input_ids)


# %%
print(tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))


# %%
'''masked_values = torch.where(input_ids == tokenizer.mask_token_id)
masked_index = masked_values[1].tolist()[0]
print(masked_index)
'''


# %%
result = model(input_ids)
#pred_ids = result[0][:, masked_index].topk(10).indices.tolist()[0]
pred_ids = result[0][:, 0].topk(10).indices.tolist()[0]
index=0
test_array=['金融','銀行','ファイナンス','融資','闇金融','金融庁','資金','調達','投資','配分','与信']
test_array_2=['緊急','切迫','応急','危機','緊急自動車','緊切','非常']
for pred_id in pred_ids:
    '''index+=1
    kw= tokenizer.decode(pred_id).replace(' ','')
    if(index):
        print('%d is %s' %(index,kw))
    '''
    print(tokenizer.decode(pred_id).replace(' ',''))

    #output_ids = input_ids.tolist()[0]
    #output_ids[masked_index] = pred_id
    #print(tokenizer.decode(output_ids))

