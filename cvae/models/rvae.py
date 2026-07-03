import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CHORD_DIM, HIDDEN_DIM, DECODER_HIDDEN_DIM, LATENT_DIM, NUM_LAYERS, DROPOUT, BIDIRECTIONAL_ENCODER


def _deterministic_decode(probs):
    batch_size, _ = probs.shape
    eps = 1e-7
    out = torch.zeros_like(probs)
    idx = torch.arange(batch_size, device=probs.device)

    root_logits = torch.log(probs[:, :12] + eps) - torch.log(1 - probs[:, :12] + eps)
    root_idx = torch.multinomial(F.softmax(root_logits, dim=-1), 1).squeeze(-1)
    out[idx, root_idx] = 1.0

    qual_logits = torch.log(probs[:, 12:20] + eps) - torch.log(1 - probs[:, 12:20] + eps)
    qual_idx = torch.multinomial(F.softmax(qual_logits, dim=-1), 1).squeeze(-1)
    out[idx, 12 + qual_idx] = 1.0

    sev_logits = torch.log(probs[:, 20:25] + eps) - torch.log(1 - probs[:, 20:25] + eps)
    sev_idx = torch.multinomial(F.softmax(sev_logits, dim=-1), 1).squeeze(-1)
    out[idx, 20 + sev_idx] = 1.0

    out[:, 25:35] = (probs[:, 25:35] > 0.5).float()

    bass_logits = torch.log(probs[:, 35:48] + eps) - torch.log(1 - probs[:, 35:48] + eps)
    bass_idx = torch.multinomial(F.softmax(bass_logits, dim=-1), 1).squeeze(-1)
    out[idx, 35 + bass_idx] = 1.0

    return out


class ComplexityPrior(nn.Module):
    def __init__(self, condition_dim=4, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(condition_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),
        )

    def forward(self, c):
        params = self.net(c)
        mu_prior, logvar_prior = params.chunk(2, dim=-1)
        logvar_prior = torch.clamp(logvar_prior, min=-10, max=10)
        return mu_prior, logvar_prior


class RVAEEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
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
        self.mu = nn.Linear(lstm_out, latent_dim)
        self.logvar = nn.Linear(lstm_out, latent_dim)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        if BIDIRECTIONAL_ENCODER:
            hn = hn[-2:]
            h = torch.cat([hn[0], hn[1]], dim=-1)
        else:
            h = hn[-1]
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class RVAEDecoder(nn.Module):
    def __init__(self, condition_dim=4, word_dropout=0.0,
                 hidden_dim=None, num_layers=None):
        super().__init__()
        self.word_dropout = word_dropout
        hidden_dim = hidden_dim or DECODER_HIDDEN_DIM
        num_layers = num_layers or NUM_LAYERS
        self.lstm = nn.LSTM(
            input_size=CHORD_DIM + condition_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=DROPOUT if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_dim, CHORD_DIM)

    def forward(self, z, x, lengths, teacher_forcing_prob=1.0):
        batch_size, max_len, _ = x.shape

        if self.training and teacher_forcing_prob < 0.999:
            return self._forward_scheduled(z, x, lengths, teacher_forcing_prob)

        if self.training and self.word_dropout > 0:
            mask = torch.rand(batch_size, max_len, 1, device=x.device) < self.word_dropout
            x = x.clone()
            x = x.masked_fill(mask.expand_as(x), 0.0)

        # Shift input right: [SOS, x[0], x[1], ..., x[n-2]] → predict [x[0], x[1], ..., x[n-1]]
        sos = torch.zeros(batch_size, 1, CHORD_DIM, device=x.device)
        x_shifted = torch.cat([sos, x[:, :-1, :]], dim=1)
        cond = z.unsqueeze(1).expand(-1, max_len, -1)
        decoder_input = torch.cat([x_shifted, cond], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(decoder_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=max_len)
        logits = self.fc(out)
        return logits

    def _forward_scheduled(self, z, x, lengths, teacher_forcing_prob):
        batch_size, max_len, _ = x.shape
        cond = z.unsqueeze(1)

        prev = torch.zeros(batch_size, 1, CHORD_DIM, device=x.device)
        h = None
        outputs = []

        for t in range(max_len):
            dec_in = torch.cat([prev, cond.expand(-1, 1, -1)], dim=-1)
            out, h = self.lstm(dec_in, h)
            logits = self.fc(out.squeeze(1))
            outputs.append(logits.unsqueeze(1))

            if t == max_len - 1:
                break

            use_gt = torch.rand(batch_size, 1, device=x.device) < teacher_forcing_prob
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred = _deterministic_decode(probs).unsqueeze(1)

            gt = x[:, t:t+1, :]
            if self.training and self.word_dropout > 0:
                wd_mask = torch.rand(batch_size, 1, 1, device=x.device) < self.word_dropout
                gt = gt.clone()
                gt = gt.masked_fill(wd_mask.expand_as(gt), 0.0)
            prev = torch.where(use_gt.unsqueeze(-1).expand(-1, 1, CHORD_DIM), gt, pred)

        return torch.cat(outputs, dim=1)


class CPredictor(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class KeyPredictor(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
        )

    def forward(self, z):
        return self.net(z)



class RVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, condition_dim=4, word_dropout=0.0,
                 z_only_decoder=True, decoder_hidden_dim=None, decoder_num_layers=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.z_only_decoder = z_only_decoder

        decoder_input_dim = latent_dim if z_only_decoder else (latent_dim + condition_dim)

        self.encoder = RVAEEncoder(latent_dim=latent_dim)
        self.prior = ComplexityPrior(condition_dim=condition_dim, latent_dim=latent_dim)
        self.decoder = RVAEDecoder(
            condition_dim=decoder_input_dim, word_dropout=word_dropout,
            hidden_dim=decoder_hidden_dim, num_layers=decoder_num_layers,
        )
        self.c_predictor = CPredictor(latent_dim=latent_dim, output_dim=4)
        self.key_predictor = KeyPredictor(latent_dim=latent_dim) if condition_dim > 4 else None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _z_dec(self, z, c):
        if self.z_only_decoder:
            return z
        return torch.cat([z, c], dim=-1)

    def forward(self, x, lengths, c, teacher_forcing_prob=1.0):
        mu, logvar = self.encoder(x, lengths)
        z = self.reparameterize(mu, logvar)

        mu_prior, logvar_prior = self.prior(c)
        c_pred = self.c_predictor(z)
        key_logits = self.key_predictor(z) if self.key_predictor is not None else None

        z_dec = self._z_dec(z, c)
        logits = self.decoder(z_dec, x, lengths, teacher_forcing_prob=teacher_forcing_prob)

        return logits, mu, logvar, mu_prior, logvar_prior, z, c_pred, key_logits

    def generate(self, z, c, max_len=32, chord_sos=None, device='cpu',
                 deterministic=False, temp=1.0):
        assert not torch.is_grad_enabled(), "generate() must be called inside torch.no_grad()"
        batch_size = z.size(0)
        z_dec = self._z_dec(z, c)

        if chord_sos is None:
            chord_sos = torch.zeros(batch_size, 1, CHORD_DIM, device=device)

        generated = []
        h = None
        x = chord_sos

        for _ in range(max_len):
            z_exp = z_dec.unsqueeze(1)
            decoder_in = torch.cat([x, z_exp], dim=-1)
            lstm_out, h = self.decoder.lstm(decoder_in, h)
            logits = self.decoder.fc(lstm_out.squeeze(1))
            probs = torch.sigmoid(logits / temp) if temp != 1.0 else torch.sigmoid(logits)
            out_vec = _deterministic_decode(probs)
            x = out_vec.unsqueeze(1)
            generated.append(x.squeeze(1))

        return torch.stack(generated, dim=1)

    def encode(self, x, lengths):
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x, lengths)
        return mu, logvar

    def sample_prior(self, c, n=1, device='cpu'):
        self.eval()
        with torch.no_grad():
            c = c.to(device) if isinstance(c, torch.Tensor) else torch.tensor(c, device=device).float()
            if c.dim() == 1:
                c = c.unsqueeze(0)
            c = c.expand(n, -1) if c.size(0) == 1 else c[:n]
            mu_prior, logvar_prior = self.prior(c)
            z = self.reparameterize(mu_prior, logvar_prior)
        return z, mu_prior, logvar_prior


def rvae_kl_loss(mu, logvar, mu_prior, logvar_prior, free_bits=0.0, per_dim_free_bits=0.0, prior_mse_weight=0.0):
    kl = 0.5 * (
        logvar_prior - logvar +
        (logvar.exp() + (mu - mu_prior).pow(2)) / logvar_prior.exp() -
        1
    )
    if per_dim_free_bits > 0:
        kl = torch.max(kl - per_dim_free_bits, torch.zeros_like(kl))
        kl = kl.sum(dim=-1)
    else:
        kl = kl.sum(dim=-1)
        if free_bits > 0:
            kl = torch.max(kl - free_bits, torch.zeros_like(kl))
    kl_loss = kl.mean()

    prior_loss = 0.0
    if prior_mse_weight > 0:
        prior_loss = F.mse_loss(mu_prior, mu.detach())
        kl_loss = kl_loss + prior_mse_weight * prior_loss

    return kl_loss, prior_loss


def per_dim_kl_rvae(mu, logvar, mu_prior, logvar_prior):
    return 0.5 * (
        logvar_prior - logvar +
        (logvar.exp() + (mu - mu_prior).pow(2)) / logvar_prior.exp() -
        1
    )
