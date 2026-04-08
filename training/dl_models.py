import torch
from torch import nn
from torch.nn import functional as F


class DETRAutoEncoder(nn.Module):
    def __init__(self, detrmodule):
        super().__init__()

        # network initialization
        self.input_dim = detrmodule.input_dim
        self.hidden_dim = detrmodule.hidden_dim
        self.fs_dim = detrmodule.fs_dim

        # encoder MLP
        self.encoder = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                     nn.LayerNorm(self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.LayerNorm(self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.LayerNorm(self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.fs_dim),
                                     nn.LayerNorm(self.fs_dim),
                                     nn.ReLU())

        # decoder MLP
        self.decoder = nn.Sequential(nn.Linear(self.fs_dim, self.hidden_dim // 8),
                                     nn.LayerNorm(self.hidden_dim // 8),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim // 8, self.hidden_dim // 2),
                                     nn.LayerNorm(self.hidden_dim // 2),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim // 2, self.hidden_dim),
                                     nn.LayerNorm(self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.LayerNorm(self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.LayerNorm(self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.input_dim))

    def forward(self, x):
        hs = self.encoder(x)
        rec = self.decoder(hs)
        return rec


class MLPhead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # network initialization
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # MLP prediction head
        self.mlp = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim),
                                 nn.LayerNorm(self.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_dim, self.hidden_dim),
                                 nn.LayerNorm(self.hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, X):
        out = self.mlp(X)
        return out


class DWIdetr(nn.Module):
    def __init__(self, args):
        super().__init__()

        # network initialization
        self.input_dim = args.input_dim  # 64
        self.hidden_dim = args.hidden_dim  # 512
        self.fs_dim = args.fs_dim  # 256
        self.n_queries = args.n_queries  # 10
        self.n_layers = args.n_dlayers  # 4
        self.n_heads = args.n_multihead  # 4

        self.aux_loss = args.aux_loss  # False

        # encoder initialization
        self.pretrained = args.pretrain_path  # "models/encoder_deep_b2k"  # False
        self.freezed = args.freeze_encoder  # False

        autoencoder = DETRAutoEncoder(self)

        # loading pretrained weights
        if self.pretrained:
            autoencoder.load_state_dict(torch.load(self.pretrained, weights_only=True))

            # freezing encoder weights
            if self.freezed:
                for param in autoencoder.parameters():
                    param.requires_grad = False

        self.encoder = autoencoder.encoder

        # query initialization
        self.queries = nn.Embedding(self.n_queries, self.fs_dim)

        # transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(self.fs_dim,
                                                        self.n_heads,
                                                        dim_feedforward=self.hidden_dim,
                                                        batch_first=True)
        self.decoder_norm = nn.LayerNorm(self.fs_dim)
        self.decoder = nn.TransformerDecoder(self.decoder_layer,
                                             num_layers=self.n_layers,
                                             norm=self.decoder_norm)

        # MLP prediction heads
        self.md_head = MLPhead(input_dim=self.fs_dim,
                               hidden_dim=self.fs_dim,
                               output_dim=1)

        self.fa_head = MLPhead(input_dim=self.fs_dim,
                               hidden_dim=self.fs_dim,
                               output_dim=1)

        self.di_head = MLPhead(input_dim=self.fs_dim,
                               hidden_dim=self.fs_dim,
                               output_dim=3)

        self.w_head = MLPhead(input_dim=self.fs_dim,
                              hidden_dim=self.fs_dim,
                              output_dim=1)

        # Existence score head with concatenated queries
        red_dim = self.fs_dim // 4
        self.qu_red = nn.Sequential(nn.Linear(self.fs_dim, red_dim),
                                    nn.LayerNorm(red_dim),
                                    nn.ReLU())

        self.extnc_head = MLPhead(input_dim=red_dim * self.n_queries,
                                  hidden_dim=self.fs_dim,
                                  output_dim=self.n_queries)

    def forward(self, X):
        B = X.size(0)

        # encoded signal
        enc_embed = self.encoder(X)

        # queries
        query_batch = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        enc_batch = enc_embed.unsqueeze(1)

        # decoder
        layer_outputs = []

        hs = query_batch
        for i in range(self.n_layers):
            hs = self.decoder.layers[i](tgt=hs, memory=enc_batch)

            # prediction on decoder layer for auxiliary loss
            if self.aux_loss:
                layer_outputs.append(self.make_predictions(hs))

        # computing predictions on final decoder block with layer normalization
        hs = self.decoder.norm(hs)
        final_output = self.make_predictions(hs)

        if self.aux_loss:
            return {
                'pred': final_output,
                'aux': layer_outputs
            }
        else:
            return final_output

    def make_predictions(self, hs):

        # normalized predictions for spectrum metrics 
        md = F.sigmoid(self.md_head(hs))
        fa = F.sigmoid(self.fa_head(hs))
        dir = F.normalize(self.di_head(hs), dim=-1)
        w = F.sigmoid(self.w_head(hs))

        # predictions for existence scores on concatenated query vectors
        hs_red = self.qu_red(hs)
        hs_cat = hs_red.reshape(hs.size(0), -1)

        extnc = self.extnc_head(hs_cat).unsqueeze(2)

        return torch.cat([md, fa, dir, w, extnc], dim=-1)
