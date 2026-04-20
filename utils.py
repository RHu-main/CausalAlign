import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- 3) 混杂估计器 q(z|x,t)（可仅基于x）---------
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

# --------- 4) 因果对齐头（FiLM / 门控）---------
class CausalAlignHead(nn.Module):
    def __init__(self, d=512, num_z=4, temp=0.07):
        super().__init__()
        self.num_z = num_z
        self.temp  = temp
        # 为每个 z 学一组 FiLM 参数
        self.gamma_v = nn.Parameter(torch.ones(num_z, d))
        self.beta_v  = nn.Parameter(torch.zeros(num_z, d))
        self.gamma_t = nn.Parameter(torch.ones(num_z, d))
        self.beta_t  = nn.Parameter(torch.zeros(num_z, d))

        # 运行中的 p(z) 先验（EMA）
        self.register_buffer("pz", torch.ones(num_z)/num_z)
        self.momentum = 0.2

    # @torch.no_grad()
    def update_prior(self, batch_qz):  # batch_qz: [B,K]
        est = batch_qz.mean(dim=0)     # minibatch 对 q 的平均近似 p(z)
        # self.pz =  est.detach()
        # self.pz = (1 - self.momentum) * self.pz + self.momentum * est
        # self.pz = (self.pz / self.pz.sum()).detach()
        #------------- modify --------------s
        w = self.momentum * batch_qz + (1 - self.momentum) * self.pz
        self.pz = (w / w.sum().clamp_min(1e-12)).detach()
        self.w = (w / w.sum().clamp_min(1e-12))

    def _film(self, h, z_idx):
        # h: [B,d]; 逐个 z 生成条件化表示
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
        """
        v,t: [B,d] 归一化后的视觉/文本向量
        返回：scalar 损失， 以及一些监控项
        """
        K = self.num_z
        v_z = self.modulate_v(v)                   # [K,B,d]
        # v_z = v
        # t_z = t
        t_z = self.modulate_t(t)                   # [K,B,d]
        losses = []
        with torch.cuda.amp.autocast(enabled=v.is_cuda):
            for k in range(K):
                # 标准 InfoNCE：正样本为对角 i==i
                logits = (v_z[k] @ t_z[k].t()) / self.temp   # [B,B]
                # logits = (v_z @ t_z.t()) / self.temp   # [B,B]
                targets = torch.arange(v.size(0), device=v.device) # relabel
                loss_k = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets)) * 0.5
                # loss_k = (F.cross_entropy(v, targets) + F.cross_entropy(t, targets)) * 0.5
                losses.append(loss_k)


        # pz = self.pz.detach()                                 # [K]
        # loss = torch.stack(losses).mean()               # 标量：\sum_z p(z) * L_z
        pz = self.w
        # pz = self.w.mean(dim=0)
        # print(f"{pz.shape},{torch.stack(losses).shape}")
        loss = torch.stack(losses) @ pz                       # 标量：\sum_z p(z) * L_z
        # return loss, {"losses_z": torch.stack(losses).detach().cpu()}
        return loss, {"pz": pz.detach().cpu(), "losses_z": torch.stack(losses).detach().cpu()}
#----------------正则化模块--------------
class OrthoRegularizer(nn.Module):
    def __init__(self, lambda_ortho=1.0, mode='full'):
        """
        Args:
            lambda_ortho: 正则化强度系数
            mode:
                'full' - 全特征正交 (计算C^T D)
                'diag' - 仅约束对应维度 (计算C ⊙ D)
        """
        super().__init__()
        self.lambda_ortho = lambda_ortho
        self.mode = mode

    def forward(self, feat_causal, feat_domain):
        """
        Args:
            feat_causal: [B, d] 因果特征
            feat_domain: [B, d] 领域特征
        Returns:
            loss: 正交正则化损失值
        """
        # 特征标准化
        # C = F.normalize(feat_causal, p=2, dim=-1)  # [B,d]
        # D = F.normalize(feat_domain.float(), p=2, dim=-1)  # [B,d]
        C = feat_causal
        D = feat_domain.float()
        if self.mode == 'full':
            # 计算全相关矩阵
            corr_matrix = torch.matmul(C.t(), D)  # [d,d]
            I = torch.eye(corr_matrix.size(0)).to(corr_matrix.device)
            loss = torch.norm(corr_matrix-I, p='fro')
        elif self.mode == 'diag':
            # 仅约束对应维度
            dot_product = torch.sum(C * D, dim=1)  # [B]
            loss = torch.mean(dot_product ** 2)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return self.lambda_ortho * loss

    @staticmethod
    def orthogonal_projection(feat):
        """Gram-Schmidt正交化投影,强制特征矩阵正交化,可在训练间歇调用增强解耦效果"""
        q, _ = torch.linalg.qr(feat.t())  # [d,B]
        return q.t()  # [B,d]