import torch
import torch.nn as nn

from torch.distributions.normal import Normal
import flow
import torch.nn.functional as F
from torch.autograd import grad

class ConvCritic(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        dim = 64
        self.conv = nn.Sequential(
            nn.Conv2d(3, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, latent_size, 4))

        embed_size = 64

        self.z_fc = nn.Sequential(
            nn.Linear(latent_size, embed_size),
            nn.LayerNorm(embed_size),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_size, embed_size),
        )

        self.x_fc = nn.Linear(latent_size, embed_size)

        self.xz_fc = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.LayerNorm(embed_size),
            nn.LeakyReLU(0.2),
            nn.Linear(embed_size, 1),
        )

    def forward(self, input):
        x, z = input
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.x_fc(x)
        z = self.z_fc(z)
        xz = torch.cat((x, z), 1)
        xz = self.xz_fc(xz)
        return xz.view(-1)


class GradientPenalty:
    def __init__(self, critic, batch_size=64, gp_lambda=10):
        self.critic = critic
        self.gp_lambda = gp_lambda
        # Interpolation coefficient
        self.eps = torch.empty(batch_size).cuda()
        # For computing the gradient penalty
        self.ones = torch.ones(batch_size).cuda()

    def interpolate(self, real, fake):
        eps = self.eps.view([-1] + [1] * (len(real.shape) - 1))
        return (eps * real + (1 - eps) * fake).requires_grad_()

    def __call__(self, real, fake):
        real = [x.detach() for x in real]
        fake = [x.detach() for x in fake]
        self.eps.uniform_(0, 1)
        interp = [self.interpolate(a, b) for a, b in zip(real, fake)]
        grad_d = grad(self.critic(interp),
                      interp,
                      grad_outputs=self.ones,
                      create_graph=True)
        batch_size = real[0].shape[0]
        grad_d = torch.cat([g.view(batch_size, -1) for g in grad_d], 1)
        grad_penalty = ((grad_d.norm(dim=1) - 1)**2).mean() * self.gp_lambda
        return grad_penalty

def dconv_bn_relu(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                           padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU())


class ConvDecoder(nn.Module):
    def __init__(self, latent_size=128):
        super().__init__()

        self.out_channels = 3
        dim = 64

        self.l1 = nn.Sequential(
            nn.Linear(latent_size, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())

        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, self.out_channels, 5, 2,
                               padding=2, output_padding=1))

    def forward(self, input):
        x = self.l1(input)
        x = x.view(x.shape[0], -1, 4, 4)
        x = self.l2_5(x)
        return x, torch.sigmoid(x)
    
def conv_ln_lrelu(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 5, 2, 2),
        nn.InstanceNorm2d(out_dim, affine=True),
        nn.LeakyReLU(0.2))


class ConvEncoder(nn.Module):
    def __init__(self, latent_size, flow_depth=2, logprob=False):
        super().__init__()

        if logprob:
            self.encode_func = self.encode_logprob
        else:
            self.encode_func = self.encode

        dim = 64
        self.ls = nn.Sequential(
            nn.Conv2d(3, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, latent_size, 4))

        if flow_depth > 0:
            # IAF
            hidden_size = latent_size * 2
            flow_layers = [flow.InverseAutoregressiveFlow(
                latent_size, hidden_size, latent_size)
                for _ in range(flow_depth)]

            flow_layers.append(flow.Reverse(latent_size))
            self.q_z_flow = flow.FlowSequential(*flow_layers)
            self.enc_chunk = 3
        else:
            self.q_z_flow = None
            self.enc_chunk = 2

        fc_out_size = latent_size * self.enc_chunk
        self.fc = nn.Sequential(
            nn.Linear(latent_size, fc_out_size),
            nn.LayerNorm(fc_out_size),
            nn.LeakyReLU(0.2),
            nn.Linear(fc_out_size, fc_out_size),
        )

    def forward(self, input, k_samples=5):
        return self.encode_func(input, k_samples)

    def encode_logprob(self, input, k_samples=5):
        x = self.ls(input)
        x = x.view(input.shape[0], -1)
        fc_out = self.fc(x).chunk(self.enc_chunk, dim=1)
        mu, logvar = fc_out[:2]
        std = F.softplus(logvar)
        qz_x = Normal(mu, std)
        z = qz_x.rsample([k_samples])
        log_q_z = qz_x.log_prob(z)
        if self.q_z_flow:
            z, log_q_z_flow = self.q_z_flow(z, context=fc_out[2])
            log_q_z = (log_q_z + log_q_z_flow).sum(-1)
        else:
            log_q_z = log_q_z.sum(-1)
        return z, log_q_z

    def encode(self, input, feat):
        x = self.ls(input)
        x = x.view(input.shape[0], -1)
        fc_out = self.fc(x).chunk(self.enc_chunk, dim=1)
        mu, logvar = fc_out[:2]
        if feat is not None:
            
            assert mu.shape == feat.shape
            mu = torch.mean(torch.stack([mu, feat]), 0)
        std = F.softplus(logvar)
        qz_x = Normal(mu, std)
        z = qz_x.rsample()
        if self.q_z_flow:
            z, _ = self.q_z_flow(z, context=fc_out[2])
        return z
