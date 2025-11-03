from transformers import DistilBertTokenizer
from torch.utils.data import Dataset
import pandas as pd
import torch


class AngerToxicDataset(Dataset):
    def __init__(self, df_path):
        super(AngerToxicDataset, self).__init__()
        self.df = pd.read_csv(df_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.unpack_data()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {'text': self.df.iloc[idx].text,
                'anger': self.df.iloc[idx].anger.item(),
                'toxic': self.df.iloc[idx].toxicity.item()}

    def unpack_data(self):
        self.text = []
        self.anger = []
        self.toxic = []

        for i in range(self.df.shape[0]):
            sample = self.df.iloc[i]
            self.text.append(sample['text'])
            self.anger.append(sample['anger'])
            self.toxic.append(sample['toxicity'])

    def collate_fn(self, batch):
        text = [item['text'] for item in batch]
        anger = [item['anger'] for item in batch]
        toxic = [item['toxic'] for item in batch]

        token_ids = self.tokenizer(text,
                                   padding=True,
                                   truncation=True,
                                   max_length=512)
        return {'text': text,
                'anger': torch.tensor(anger),
                'toxic': torch.tensor(toxic),
                'input_ids': torch.tensor(token_ids['input_ids']),
                'attention_mask': torch.tensor(token_ids['attention_mask'])}