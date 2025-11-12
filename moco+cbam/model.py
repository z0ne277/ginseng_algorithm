import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CBAM(nn.Module):
    """ 通道 + 空间注意力 (CBAM) """
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x) * x
        sa = self.spatial_attention(torch.cat([ca.mean(dim=1, keepdim=True), ca.max(dim=1, keepdim=True)[0]], dim=1)) * ca
        return sa

class MoCoV3Ginseng(nn.Module):
    def __init__(self, feature_dim=256, K=1024, m=0.999, T=0.07, device=None):
        super(MoCoV3Ginseng, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder_q = models.resnet50(weights="IMAGENET1K_V1")
        self.encoder_k = models.resnet50(weights="IMAGENET1K_V1")
        resnet_dim = self.encoder_q.fc.in_features


        self.encoder_q = nn.Sequential(*list(self.encoder_q.children())[:-2])  # 到最后一个conv
        self.encoder_k = nn.Sequential(*list(self.encoder_k.children())[:-2])


        self.cbam_q = CBAM(resnet_dim)
        self.cbam_k = CBAM(resnet_dim)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp_q = nn.Sequential(
            nn.Linear(resnet_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        self.mlp_k = nn.Sequential(
            nn.Linear(resnet_dim, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

        self.K = K
        self.m = m
        self.T = T

        self.register_buffer("queue", torch.randn(feature_dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.to(self.device)

    def forward(self, img1, img2):
        img1, img2 = img1.to(self.device), img2.to(self.device)

        q = self.encoder_q(img1)
        k = self.encoder_k(img2)
        q = self.cbam_q(q)
        k = self.cbam_k(k)
        q = self.avgpool(q).flatten(1)
        k = self.avgpool(k).flatten(1)
        q = self.mlp_q(q)
        k = self.mlp_k(k)
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        return q, k

    @torch.no_grad()
    def momentum_update_key_encoder(self):

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.mul_(self.m).add_(param_q.data, alpha=1 - self.m)
        for param_q, param_k in zip(self.mlp_q.parameters(), self.mlp_k.parameters()):
            param_k.data.mul_(self.m).add_(param_q.data, alpha=1 - self.m)

    def contrastive_loss(self, q, k):

        N = q.shape[0]

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)  # [N, 1+K]
        logits /= self.T
        labels = torch.zeros(N, dtype=torch.long).to(q.device)  # 正样本在第0位
        loss = F.cross_entropy(logits, labels)
        return loss

    @torch.no_grad()
    def update_queue(self, k):

        k = k.detach()
        batch_size = k.shape[0]
        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            num_first = self.K - ptr
            num_next = batch_size - num_first
            self.queue[:, ptr:] = k[:num_first].T
            self.queue[:, :num_next] = k[num_first:].T
            self.queue_ptr[0] = num_next
        else:
            self.queue[:, ptr:ptr + batch_size] = k.T
            self.queue_ptr[0] = (ptr + batch_size) % self.K

