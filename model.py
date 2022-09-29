from sklearn.metrics import average_precision_score, roc_auc_score
from torch import device
from torch_geometric.nn.inits import reset

from bgae.utils.modules import *


class Model(torch.nn.Module):
    def __init__(self, x_dim, **kwargs):
        super(Model, self).__init__()
        self.config = kwargs
        self.mute_recon = kwargs["mute_recon"]
        self.x_dim = x_dim
        self.z_dim = kwargs["z_dim"]
        self.lmbda = kwargs["barlow_lambda"]
        device = "cuda"

        if not self.config["variational"]:
            self.encoder = GCN_Encoder_L1(x_dim, self.z_dim)
        else:
            self.encoder = GCN_V_Encoder_L1_params(x_dim, self.z_dim)

        self.decoder = InnerProduct_Decoder()
        self.attention_w = torch.nn.Parameter(
            torch.ones(2 * self.z_dim).to(device),
            requires_grad=True
        )
        self.off_diagonal_mask = (
            torch.ones((self.z_dim, self.z_dim)) - torch.eye(self.z_dim)
        ).bool().to(device)
        self.eye_z = torch.eye(self.z_dim).to(device)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=kwargs["opt"]["lr"],
            weight_decay=kwargs["opt"]["decay"]
        )
        Model.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)
        uniform(self.z_dim, self.attention_w)

    def l_barlow_parts_mean(self, c):
        return - torch.log(c.diag() + EPS).mean() - \
            self.lmbda * torch.log(1 - c[self.off_diagonal_mask] + EPS).mean()

    def fused_z(self, z1, z2):
        if not self.config["attention"]:
            return (z1 + z2) / 2
        else:
            a1 = z1 @ self.attention_w[:self.z_dim]
            a2 = z2 @ self.attention_w[self.z_dim:]
            b = F.leaky_relu(torch.vstack((a1, a2))).softmax(dim=0)
            z = (torch.cat((z1[None, :, :], z2[None, :, :]),
                 dim=0) * b[:, :, None]).sum(dim=0)
            return z

    def training_loop(self, inp_1, inp_2):
        self.train()
        self.optimizer.zero_grad()

        z_I, enc_extras1 = self.encoder(*inp_1)
        z_Ihat, enc_extras2 = self.encoder(*inp_2)
        z = self.fused_z(z_I, z_Ihat)

        # barlow loss
        z_prod = (z_I - z_I.mean(0)).T @ (z_Ihat - z_Ihat.mean(0))
        c = z_prod.abs().sigmoid()

        l_barlow = self.config["barlow_loss_weight"] * \
            self.l_barlow_parts_mean(c)

        # encoders' loss, in case of variational encoder
        l_enc1 = self.encoder.loss(*enc_extras1)
        l_enc2 = self.encoder.loss(*enc_extras2)

        # edge reconstruction loss
        if self.mute_recon:
            l_recon2 = torch.tensor(0.)
            l_recon1 = torch.tensor(0.)
        else:
            l_recon1 = self.decoder.recon_loss(z, inp_1[1], inp_1[2])
            l_recon2 = self.decoder.recon_loss(z, inp_2[1], inp_2[2])

        loss = l_enc1 + l_enc2 + l_recon1 + l_recon2 + l_barlow
        loss.backward()
        self.optimizer.step()

        return float(loss), {
            "l_enc1": l_enc1,
            "l_enc2": l_enc2,
            "l_recon1": l_recon1,
            "l_recon2": l_recon2,
            "l_barlow": l_barlow
        }, {
            "z_I": z_I,
            "z_Ihat": z_Ihat,
            "enc_extras1": enc_extras1,
            "enc_extras2": enc_extras2
        }

    @torch.no_grad()
    def test_link_prediction(self, z, pos_edge_index, neg_edge_index):
        self.eval()
        y = z.new_zeros(pos_edge_index.size(1) + neg_edge_index.size(1))
        y[:pos_edge_index.size(1)] = 1
        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return {"auc": roc_auc_score(y, pred), "ap": average_precision_score(y, pred)}
