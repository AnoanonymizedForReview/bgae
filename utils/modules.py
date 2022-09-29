import torch
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn.inits import uniform
from torch_geometric.utils import remove_self_loops, add_self_loops, negative_sampling
import torch.nn.functional as F
from utils.extras import MAX_LOGSTD, EPS


class Encoder(torch.nn.Module):

    def loss(self, *args):
        return torch.tensor(0.)


class V_Encoder(Encoder):

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def loss(self, mu=None, logstd=None):
        logstd = logstd.clamp(max=MAX_LOGSTD)
        loss = -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu ** 2 - logstd.exp() ** 2, dim=1)
        )
        loss /= mu.shape[0]
        return loss


class EdgeDecoder(torch.nn.Module):
    def recon_loss(self, z, pos_edge_index, pos_edge_weights, neg_edge_index=None, z2=None):
        # pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_loss = - torch.log(
            self.forward(z, pos_edge_index, sigmoid=True) + EPS
        ).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = - torch.log(
            1 - self.forward(z, neg_edge_index, sigmoid=True) + EPS
        ).mean()

        return pos_loss + neg_loss


class GCN_Encoder_L1(Encoder):
    def __init__(self, in_channels, out_channels, activation=None, **kwargs):
        super(GCN_Encoder_L1, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, **kwargs)
        self.activation = activation

    def forward(self, x, edge_index, edge_weight=None):
        val = self.conv(x, edge_index, edge_weight=edge_weight)
        if self.activation is None:
            return val, ()
        else:
            return self.activation(val), ()


class GCN_V_Encoder(V_Encoder):
    def __init__(self, in_channels, out_channels):
        super(GCN_V_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index).relu()
        mu, logstd = self.conv_mu(x, edge_index, edge_weight), self.conv_logstd(
            x, edge_index, edge_weight)
        z = self.reparametrize(mu, logstd)
        return (z, (mu, logstd)) if self.training else (mu, ())


class GCN_V_Encoder_L1_params(V_Encoder):
    def __init__(self, in_channels, out_channels):
        super(GCN_V_Encoder_L1_params, self).__init__()
        self.conv_mu = GCNConv(in_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index, edge_weight=None):
        # x = self.conv1(x, edge_index).relu()
        mu = self.conv_mu(x, edge_index, edge_weight)
        logstd = self.conv_logstd(x, edge_index, edge_weight)
        z = self.reparametrize(mu, logstd)
        return (z, (mu, logstd)) if self.training else (mu, ())


class InnerProduct_Decoder(EdgeDecoder):
    def forward(self, z, edge_index, sigmoid=True, z2=None):
        z2 = z if z2 is None else z2
        value = (z[edge_index[0]] * z2[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value
