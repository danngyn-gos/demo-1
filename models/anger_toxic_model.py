import torch
from torch import nn
from transformers import DistilBertModel, AutoConfig
from typing import Optional


class AngerToxicClassifyModel(nn.Module):
    def __init__(self,
                 config,
                 ):
        super().__init__()
        self.bert_config = AutoConfig.from_pretrained(config.PRETRAINED)
        self.config = config
        
        if config.LOAD_PRETRAINED:
            self.distilbert = DistilBertModel(self.bert_config).from_pretrained(config.PRETRAINED)
        else:
            self.distilbert = DistilBertModel(self.bert_config)

        if self.config.FREEZE_BACKBONE:
            self.freeze_backbone()
        
        self.latent_dim = config.DIM
        self.pre_classifier = nn.Linear(self.latent_dim, config.DIM)
        
        # self.sent_classifier = nn.Linear(config.DIM, config.SENT_CLASSES)
        self.anger_head = nn.Linear(self.latent_dim, config.ANGER)

        self.toxicity_head = nn.Linear(self.latent_dim, config.TOXICITY)
        self.dropout = nn.Dropout(config.DROPOUT)
        
        
    
    def freeze_backbone(self):
        for param in self.distilbert.parameters():
            param.requires_grad = False

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs):
        
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        # Sentiment head
        # sent_output = self.sent_classifier(pooled_output)

        # Emotion head
        emo_output = self.anger_head(pooled_output)

        # Toxicity head
        toxic_output = self.toxicity_head(pooled_output)

        return {
            'anger_output': emo_output,
            'toxic_output': toxic_output
        }