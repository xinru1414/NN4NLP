import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import numpy as np
from preprocess import load_pte


class CNNClassify(nn.Module):
    '''
    CNN text classification
    Model arch
    '''
    def __init__(self, cfg, dl, device):
        super().__init__()
        self.config = cfg

        self.embed = nn.Embedding(num_embeddings=dl.vocab_size, embedding_dim=config.emb_dim, padding_idx = dl.PAD_IDX)
        # initialize word embeddings
        if config.pte_path is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            emb, w2i = load_pte(config.pte_path, config.most_frequent_pte)
            init_emb = np.zeros((dl.vocab_size, config.emb_dim))
            count = 0
            for word in dl.w2i:
                if word in w2i:
                    count += 1
                    init_emb[dl.w2i[word]] = emb[w2i[word]]
            print(f'loaded {count} words from Google pte')
            self.embed.weight.data.copy_(torch.from_numpy(init_emb))
            self.embed.weight.requires_grad = True
        # 2D and 1D are the same, either can be used in the forward pass
        # 2D CNN layer
        self.convs2D = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=config.filters, kernel_size=(k, config.emb_dim),bias=True) for k in config.kernel_sizes])
        # 1D CNN layer
        self.convs1D = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=config.filters, kernel_size=(k, config.emb_dim)) for k in config.kernel_sizes])
        # full connected layer
        self.fc = nn.Linear(in_features=len(config.kernel_sizes) * config.filters, out_features=dl.label_size)

        self.to(device)

    def forward(self, x):
        # dim: (batch_size, seq_len=60, emb_dim=300)
        x = self.embed(x)
        # drop out only during training
        x = F.dropout(x,p=0.5, training=self.training)

        # dim: (batch_size, Cin=1, seq_len=60, emb_dim=300): #print(x.shape)
        x = x.unsqueeze(1)

        # dim (batch_size, Cout=config.filters=100, seq_len-kernel_size+1): #print(h[0].shape)
        h = [F.relu(conv(x)).squeeze(3) for conv in self.convs1D]

        #h = [F.dropout(k, p=config.dropout, training=self.training) for k in h]

        #h = [F.relu(conv(x)).squeeze(3) for conv in self.convs1D]

        # max pooling
        # dim (batch_size, Cout=config.filters=100
        # after pooling: dim (batch_size, Cout=config.filters=100, 1)
        h = [F.max_pool1d(k, k.size(2)).squeeze(2) for k in h]
        # dim (batch_size, 100*(kernel_size=3)))
        h = torch.cat(h, 1)

        h = F.dropout(h, p=config.dropout, training=self.training)

        h = self.fc(h)
        return h

    def loss(self, x, y):
        return F.cross_entropy(self.forward(x), y)

    def predict(self, x):
        return torch.argmax(F.softmax(self.forward(x), dim=1), dim=1).cpu().numpy()

    def save(self):
        save_best_path = self.config.save_best
        torch.save(self.state_dict(), save_best_path)

    def load(self):
        load_best_path = self.config.save_best
        self.load_state_dict(torch.load(load_best_path))