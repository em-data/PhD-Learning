from __future__ import division, absolute_import
import torch
import torch.nn as nn
from torch.nn import functional as F


class OurLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.
    
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, k_an=3, k_ap=6, normalize=True):
        super(OurLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.k_an = k_an   # (k-an越大，越难)
        self.k_ap = k_ap   # (k-ap越小,越难 )
        self.normalize = normalize

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, sequence_length, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
        """
        assert inputs.dim() == 3
        b, s, d = inputs.size()
        m = s * 2           # ap或an两个序列的总采样数
        inputs_r = inputs   # 将输入inputs从(b*s, d)变形为(b, s, d)
        if self.normalize:
            inputs_r = F.normalize(inputs_r, p=2, dim=-1)
        dist_ap, dist_an = [], []       # dist_ap, dist_an分别为相同ID和不同ID序列间的距离
        dist_ij_ji = torch.zeros(b, b)

        for i in range(b):
            for j in range(i+1, b):
                inputs_c = torch.cat([inputs_r[i], inputs_r[j]], dim=0)
                dist = torch.pow(inputs_c, 2).sum(dim=1, keepdim=True).expand(m, m) # 每个数平方后， 进行加和（通过keepdim保持2维），再扩展成nxn维
                dist = dist + dist.t() # 这样每个dis[i][j]代表的是第i个特征与第j个特征的平方的和
                dist.addmm_(1, -2, inputs_c, inputs_c.t()) # 然后减去2倍的 第i个特征*第j个特征 从而通过完全平方式得到 (a-b)^2
                """
                # hard 版本
                dist = dist.clamp(min=1e-12).sqrt() # 然后开方
                if (targets[i] == targets[j]):
                     dist_ij_min,  dist_ij_min_idx = torch.min(dist[0:s, s:m], 1, True)
                     dist_ji_min,  dist_ji_min_idx = torch.min(dist[0:s, s:m], 0, True)
                     dist_ij_min_kmax,  dist_ij_min_kmax_idx = torch.topk(dist_ij_min, self.k_ap, dim=0)
                     dist_ji_min_kmax,  dist_ji_min_kmax_idx = torch.topk(dist_ji_min, self.k_ap)
                     dist_ij_ji[i][j] = torch.max(dist_ij_min_kmax[-1],  dist_ji_min_kmax[:, -1])
                
                else:
                    dist_ij_min, dist_ij_min_idx = torch.min(dist[0:s, s:m], 1, True)
                    dist_ji_min, dist_ji_min_idx = torch.min(dist[0:s, s:m], 0, True)
                    dist_ij_min_kmax,  dist_ij_min_kmax_idx = torch.topk(dist_ij_min, self.k_an, dim=0)
                    dist_ji_min_kmax,  dist_ji_min_kmax_idx = torch.topk(dist_ji_min, self.k_an)
                    dist_ij_ji[i][j] = torch.max(dist_ij_min_kmax[-1],  dist_ji_min_kmax[:, -1])
                """
                """soft版本"""
                dist = dist.clamp(min=1e-12).sqrt()  # 然后开方
                dist_row = torch.softmax(dist, dim=1)
                dist_column = torch.softmax(dist, dim=0)
                if (targets[i] == targets[j]):
                    dist_ij_min, dist_ij_min_idx = torch.min(dist_row[0:s, s:m], 1, True)
                    dist_ji_min, dist_ji_min_idx = torch.min(dist_column[0:s, s:m], 0, True)
                    dist_ij_min_kmax, dist_ij_min_kmax_idx = torch.topk(dist_ij_min, self.k_ap, dim=0)
                    dist_ji_min_kmax, dist_ji_min_kmax_idx = torch.topk(dist_ji_min, self.k_ap)
                    dist_ij_ji[i][j] = torch.max(dist_ij_min_kmax[-1], dist_ji_min_kmax[:, -1])

                else:
                    dist_ij_min, dist_ij_min_idx = torch.min(dist_row[0:s, s:m], 1, True)
                    dist_ji_min, dist_ji_min_idx = torch.min(dist_column[0:s, s:m], 0, True)
                    dist_ij_min_kmax, dist_ij_min_kmax_idx = torch.topk(dist_ij_min, self.k_an, dim=0)
                    dist_ji_min_kmax, dist_ji_min_kmax_idx = torch.topk(dist_ji_min, self.k_an)
                    dist_ij_ji[i][j] = torch.min(dist_ij_min_kmax[-1], dist_ji_min_kmax[:, -1])

        dist_ij_ji = dist_ij_ji + dist_ij_ji.t()
        mask = targets.expand(b, b).eq(targets.expand(b, b).t())

        for i in range(b):
            dist_ap.append(dist_ij_ji[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist_ij_ji[i][mask[i] == 0].min().unsqueeze(0))
        
        dist_ap = torch.cat(dist_ap) # 将list里的tensor拼接成新的tensor
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


if __name__ == '__main__':
    test_tensor = torch.rand(8, 6, 25)
    test_target = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    test_loss = OurLoss()
    loss = test_loss(test_tensor, test_target)
