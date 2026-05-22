import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CHORD_DIM, HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, DROPOUT, BIDIRECTIONAL_ENCODER


class ChordEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=CHORD_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0,
            bidirectional=BIDIRECTIONAL_ENCODER,
        )
        lstm_out = HIDDEN_DIM * (2 if BIDIRECTIONAL_ENCODER else 1)
        self.mu = nn.Linear(lstm_out, LATENT_DIM)
        self.logvar = nn.Linear(lstm_out, LATENT_DIM)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        if BIDIRECTIONAL_ENCODER:
            hn = hn[-2:]  # last forward + last backward
            h = torch.cat([hn[0], hn[1]], dim=-1)
        else:
            h = hn[-1]
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class ChordDecoder(nn.Module):
    def __init__(self, condition_dim=None, word_dropout=0.0):
        super().__init__()
        self.word_dropout = word_dropout
        if condition_dim is None:
            condition_dim = LATENT_DIM
        self.lstm = nn.LSTM(
            input_size=CHORD_DIM + condition_dim,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0,
        )
        self.fc = nn.Linear(HIDDEN_DIM, CHORD_DIM)

    def forward(self, z, x, lengths, teacher_forcing=True):
        batch_size, max_len, _ = x.shape

        if self.training and self.word_dropout > 0:
            mask = torch.rand(batch_size, max_len, 1, device=x.device) < self.word_dropout
            x = x.clone()
            x = x.masked_fill(mask.expand_as(x), 0.0)

        cond = z.unsqueeze(1).expand(-1, max_len, -1)
        decoder_input = torch.cat([x, cond], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(decoder_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=max_len)
        logits = self.fc(out)
        return logits


class CVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, condition_dim=None, word_dropout=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        decoder_input_dim = latent_dim + (condition_dim if condition_dim else 0)

        self.encoder = ChordEncoder()
        self.decoder = ChordDecoder(condition_dim=decoder_input_dim, word_dropout=word_dropout)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, lengths, z_cond=None):
        mu, logvar = self.encoder(x, lengths)
        z = self.reparameterize(mu, logvar)

        if z_cond is not None:
            z = torch.cat([z, z_cond], dim=-1)

        logits = self.decoder(z, x, lengths)
        return logits, mu, logvar, z

    def generate(self, z, max_len=32, chord_sos=None, device='cpu'):
        batch_size = z.size(0)

        if chord_sos is None:
            chord_sos = torch.zeros(batch_size, 1, CHORD_DIM, device=device)

        generated = []
        h = None
        x = chord_sos

        for _ in range(max_len):
            z_exp = z.unsqueeze(1)
            decoder_in = torch.cat([x, z_exp], dim=-1)
            lstm_out, h = self.decoder.lstm(decoder_in, h)
            logits = self.decoder.fc(lstm_out.squeeze(1))
            probs = torch.sigmoid(logits)
            x = (probs > 0.5).float().unsqueeze(1)
            generated.append(x.squeeze(1))

        return torch.stack(generated, dim=1)


def cvae_loss(logits, targets, mu, logvar, lengths, beta=0.1, free_bits=0.5):
    recon_loss = 0
    total_tokens = 0
    for i in range(logits.size(0)):
        n = lengths[i]
        l = F.binary_cross_entropy_with_logits(
            logits[i, :n], targets[i, :n], reduction='sum'
        )
        recon_loss += l
        total_tokens += n
    recon_loss = recon_loss / total_tokens

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    kl = torch.max(kl - free_bits, torch.zeros_like(kl))
    kl_loss = kl.mean()

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
