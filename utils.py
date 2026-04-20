import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Confounder estimation q(z|x,t) ---------
class ConfounderEstimator(nn.Module):
    def __init__(self, d_in=768, num_z=4, use_text=False):
        super().__init__()
        self.use_text = use_text
        self.fc = nn.Sequential(
            nn.Linear(d_in*(2 if use_text else 1), d_in),
            nn.LayerNorm(d_in),
            nn.GELU(),
            nn.Linear(d_in, num_z)
        )
        self.num_z = num_z

    def forward(self, v, t=None):
        h = torch.cat([v, t], dim=-1) if (self.use_text and t is not None) else v
        logits = self.fc(h)# [B, K]
        q = torch.softmax(logits, dim=-1)
        # q = logits.log_softmax(dim=-1).exp()   # softmax -> q(z|.)
        return q, logits

# --------- CausalAlign-heterogenous ---------
class CausalAlignHead(nn.Module):
    def __init__(self, d=512, num_z=4, temp=0.07):
        super().__init__()
        self.num_z = num_z
        self.temp  = temp
 
        self.gamma_v = nn.Parameter(torch.ones(num_z, d))
        self.beta_v  = nn.Parameter(torch.zeros(num_z, d))
        self.gamma_t = nn.Parameter(torch.ones(num_z, d))
        self.beta_t  = nn.Parameter(torch.zeros(num_z, d))

 
        self.register_buffer("pz", torch.ones(num_z)/num_z)
        self.momentum = 0.2

    # @torch.no_grad()
    def update_prior(self, batch_qz):  # batch_qz: [B,K]
        est = batch_qz.mean(dim=0)     
        # self.pz =  est.detach()
        # self.pz = (1 - self.momentum) * self.pz + self.momentum * est
        # self.pz = (self.pz / self.pz.sum()).detach()
        #------------- modify --------------s
        w = self.momentum * batch_qz + (1 - self.momentum) * self.pz
        self.pz = (w / w.sum().clamp_min(1e-12)).detach()
        self.w = (w / w.sum().clamp_min(1e-12))

    def _film(self, h, z_idx):
        g = self.gamma_v[z_idx] if isinstance(h, torch.Tensor) else None

    def modulate_v(self, v):  # 返回 [K, B, d]
        K,B,D = self.num_z, v.size(0), v.size(1)
        gv = self.gamma_v.view(K,1,D)
        bv = self.beta_v.view(K,1,D)
        v_exp = v.unsqueeze(0)         # [1,B,D]
        return F.normalize(gv * v_exp + bv, dim=-1)

    def modulate_t(self, t):  # 返回 [K, B, d]
        K,B,D = self.num_z, t.size(0), t.size(1)
        gt = self.gamma_t.view(K,1,D)
        bt = self.beta_t.view(K,1,D)
        t_exp = t.unsqueeze(0)
        return F.normalize(gt * t_exp + bt, dim=-1)

    def backdoor_clip_loss(self, v, t):
        K = self.num_z
        v_z = self.modulate_v(v)                   # [K,B,d]
        # v_z = v
        # t_z = t
        t_z = self.modulate_t(t)                   # [K,B,d]
        losses = []
        with torch.cuda.amp.autocast(enabled=v.is_cuda):
            for k in range(K):
                logits = (v_z[k] @ t_z[k].t()) / self.temp   # [B,B]
                # logits = (v_z @ t_z.t()) / self.temp   # [B,B]
                targets = torch.arange(v.size(0), device=v.device) # relabel
                loss_k = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) * 0.5
                # loss_k = (F.cross_entropy(v, targets) + F.cross_entropy(t, targets)) * 0.5
                losses.append(loss_k)


        # pz = self.pz.detach()                                 # [K]
        # loss = torch.stack(losses).mean()               
        pz = self.w
        # pz = self.w.mean(dim=0)
        # print(f"{pz.shape},{torch.stack(losses).shape}")
        loss = torch.stack(losses) @ pz                       
        # return loss, {"losses_z": torch.stack(losses).detach().cpu()}
        return loss, {"pz": pz.detach().cpu(), "losses_z": torch.stack(losses).detach().cpu()}
#----------------regularization--------------
class OrthoRegularizer(nn.Module):
    def __init__(self, lambda_ortho=1.0, mode='full'):

        super().__init__()
        self.lambda_ortho = lambda_ortho
        self.mode = mode

    def forward(self, feat_causal, feat_domain):
        """
        Args:
            feat_causal: [B, d] 
            feat_domain: [B, d] 
        Returns:
            loss
        """

        # C = F.normalize(feat_causal, p=2, dim=-1)  # [B,d]
        # D = F.normalize(feat_domain.float(), p=2, dim=-1)  # [B,d]
        C = feat_causal
        D = feat_domain.float()
        if self.mode == 'full':

            corr_matrix = torch.matmul(C.t(), D)  # [d,d]
            I = torch.eye(corr_matrix.size(0)).to(corr_matrix.device)
            loss = torch.norm(corr_matrix-I, p='fro')
        elif self.mode == 'diag':

            dot_product = torch.sum(C * D, dim=1)  # [B]
            loss = torch.mean(dot_product ** 2)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return self.lambda_ortho * loss

    @staticmethod
    def orthogonal_projection(feat):
        q, _ = torch.linalg.qr(feat.t())  # [d,B]
        return q.t()  # [B,d]
