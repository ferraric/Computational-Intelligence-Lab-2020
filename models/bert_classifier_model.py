import torch.nn as nn
import numpy as np
import torch

def get_num_params(model):
    mp = filter(lambda p: p.requires_grad, model.parameters())
    return sum(np.prod(p.size()) for p in mp)

class Transformer(nn.Module):
    "Adopted from https://github.com/huggingface/naacl_transfer_learning_tutorial"
    def __init__(self, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout, causal):
        super().__init__()
        self.causal = causal
        self.tokens_embeddings = nn.Embedding(num_embeddings, embed_dim)
        self.position_embeddings = nn.Embedding(num_max_positions, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_dim, embed_dim)))
            self.layer_norms_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.layer_norms_2.append(nn.LayerNorm(embed_dim, eps=1e-12))

    def forward(self, x, padding_mask=None):
        """ x has shape [seq length, batch], padding_mask has shape [batch, seq length] """
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)

        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norms_1, self.attentions,
                                                                       self.layer_norms_2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False, key_padding_mask=padding_mask)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h

class TransformerWithClfHead(nn.Module):
    "Adopted from https://github.com/huggingface/naacl_transfer_learning_tutorial"

    def __init__(self, bert_config, fine_tuning_config):
        super().__init__()
        self.config = fine_tuning_config
        self.transformer = Transformer(bert_config.embed_dim, bert_config.hidden_dim, bert_config.num_embeddings,
                                       bert_config.num_max_positions, bert_config.num_heads, bert_config.num_layers,
                                       fine_tuning_config.dropout, causal=not bert_config.mlm)
        self.classification_head = torch.nn.Sequential(
                                        nn.Linear(bert_config.embed_dim, 200),
                                        nn.ReLU(),
                                        nn.Linear(200, 100),
                                        nn.ReLU(),
                                        nn.Linear(bert_config.embed_dim, 1),
                                        nn.Sigmoid()
                                    )
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_range)
        if isinstance(module, (nn.Linear, nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, clf_tokens_mask, clf_labels=None, padding_mask=None):
        hidden_states = self.transformer(x, padding_mask)

        clf_tokens_states = (hidden_states * clf_tokens_mask.unsqueeze(-1).float()).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)

        if clf_labels is not None:
            loss_function = nn.BCELoss()
            loss = loss_function(clf_logits.view(-1), clf_labels.view(-1).float())
            predictions = torch.round(clf_logits.view(-1))
            batch_size = clf_logits.view(-1).shape
            correct_predicions = (predictions == clf_labels.float()).sum().item()
            batch_accuracy = 100*(float(correct_predicions)/float(batch_size[0]))
            return clf_logits, loss, batch_accuracy
        return clf_logits