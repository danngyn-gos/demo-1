from transformers import DistilBertTokenizer
from torch.utils.data import Dataset
import pandas as pd
import torch


class SentimentDataset(Dataset):
    def __init__(self, config, df_path):
        super(SentimentDataset, self).__init__()
        self.df = pd.read_csv(df_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(config.PRETRAINED)
        self.unpack_data()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {'text': self.df.iloc[idx].text,
                'empathy': self.df.iloc[idx].Empathy.item(),
                'sentiment': self.df.iloc[idx].Emotion.item()}

    def unpack_data(self):
        self.text = []
        self.empathy = []
        self.sentiment = []

        for i in range(self.df.shape[0]):
            sample = self.df.iloc[i]
            self.text.append(sample['text'])
            self.anger.append(sample['empathy'])
            self.toxic.append(sample['sentiment'])

    def collate_fn(self, batch):
        text = [item['text'] for item in batch]
        empathy = [item['empathy'] for item in batch]
        sentiment = [item['sentiment'] for item in batch]

        token_ids = self.tokenizer(text,
                                   padding=True,
                                   truncation=True,
                                   max_length=512)
        return {'text': text,
                'empathy': torch.tensor(empathy),
                'sentiment': torch.tensor(sentiment),
                'input_ids': torch.tensor(token_ids['input_ids']),
                'attention_mask': torch.tensor(token_ids['attention_mask'])}