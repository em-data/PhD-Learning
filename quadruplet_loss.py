import torch
from torch import nn
from torch.nn import functional as F

class QuadrupletLoss(nn.Module):
    def __init__(self, margin1=1., margin2=0.5, num_instances=4, alpha=1.0, beta=1.,
                 normalize=True, k_an=3, k_ap=6, hasdouf=True):
        super(QuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.ranking_loss1 = nn.MarginRankingLoss(margin=margin1)
        self.ranking_loss2 = nn.MarginRankingLoss(margin=margin2)
        self.num_instances = num_instances
        self.alpha = alpha
        self.beta = beta
        self.hasdouf = hasdouf
        self.normalize = normalize
        self.k_an = k_an
        self.k_ap = k_ap

    def forward(self, inputs, targets):

        input_fea = inputs
        if self.normalize:
            input_fea = F.normalize(input_fea, p=2, dim=-1)
        if not self.hasdouf:
            assert input_fea.dim() == 2
            b = input_fea.size(0)
            # Compute pairwise distance, replace by the official when merged
            dist = torch.pow(input_fea, 2).sum(1).expand(b, b)
            dist = dist + dist.t()
            dist.addmm_(1, -2, input_fea, input_fea.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
            mask = targets.expand(b, b).eq(targets.expand(b, b).t())
        else:
            assert input_fea.dim() == 3
            b, s, d = input_fea.size()
            dist_ij_ji = torch.zeros(b, b)
            m = s*2
            for i in range(b):
                for j in range(i + 1, b):
                    inputs_c = torch.cat([input_fea[i], input_fea[j]], dim=0)
                    dist = torch.pow(inputs_c, 2).sum(dim=1, keepdim=True).expand(m, m)  # 每个数平方后， 进行加和（通过keepdim保持2维），再扩展成nxn维
                    dist = dist + dist.t()  # 这样每个dis[i][j]代表的是第i个特征与第j个特征的平方的和
                    dist.addmm_(1, -2, inputs_c, inputs_c.t())  # 然后减去2倍的 第i个特征*第j个特征 从而通过完全平方式得到 (a-b)^2
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

            dist = dist_ij_ji + dist_ij_ji.t()
            mask = targets.expand(b, b).eq(targets.expand(b, b).t())

        # For each anchor, find the hardest positive and negative

        dist_ap, dist_an, dist_to_get = [], [], []
        for i in range(b):
            hard_positive = dist[i][mask[i]].max()
            dist_ap.append(hard_positive)

            hard_negative = dist[i][mask[i] == 0].min(0)
            dist_an.append(hard_negative[0])

            negative_negative = hard_negative[1]
            lower_bound = (i // self.num_instances) * self.num_instances
            if (negative_negative >= lower_bound).cpu().data.numpy():
                negative_negative = negative_negative + self.num_instances

            dist_to_get.append(negative_negative)
        # print(dist_to_get)
        # print([dist_an[i.cpu().data.numpy()] for i in dist_to_get])
        dist_ann = torch.stack([dist_an[i.cpu().data.numpy()] for i in dist_to_get])
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        # Compute ranking hinge loss

        y = torch.ones_like(dist_an)

        TripletLoss1 = self.ranking_loss1(dist_an, dist_ap, y)
        TripletLoss2 = self.ranking_loss2(dist_ann, dist_ap, y)
        loss = self.alpha * TripletLoss1 + self.beta * TripletLoss2

        return loss
