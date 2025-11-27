from transformers import DistilBertTokenizer, DistilBertModel, AutoConfig
tokenizer = DistilBertTokenizer.from_pretrained('tabularisai/multilingual-sentiment-analysis')
bert_config = AutoConfig.from_pretrained('tabularisai/multilingual-sentiment-analysis')
distilbert = DistilBertModel(bert_config).from_pretrained('tabularisai/multilingual-sentiment-analysis')
new_special_tokens = ['<CTX>', '</CTX>', '<TARGET>', '</TARGET>']
tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
distilbert.resize_token_embeddings(119551)
print(len(tokenizer))