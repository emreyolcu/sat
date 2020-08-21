import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fmv_pos = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )
        self.fmc_pos = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )
        self.fmv_neg = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )
        self.fmc_neg = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )
        self.fuv = nn.Sequential(
            nn.Linear(input_size + output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )
        self.fuc = nn.Sequential(
            nn.Linear(input_size + output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size),
            nn.ReLU(),
        )

    # @profile
    def message(self, h, data):
        vadj, cadj = data.adj
        hv, hc = h
        xev, xec = data.x[2:]
        # mv = torch.spmm(vadj, self.fmv(torch.cat((hc.repeat(2, 1), xec), dim=1)))
        # mc = torch.spmm(cadj, self.fmc(torch.cat((hv.repeat(2, 1), xev), dim=1)))
        mv = torch.spmm(vadj, torch.cat([self.fmv_pos(hc.clone()), self.fmv_neg(hc.clone())], dim=0))
        mc = torch.spmm(cadj, torch.cat([self.fmc_pos(hv.clone()), self.fmc_neg(hv.clone())], dim=0))
        # m1 = self.fmv_pos(hc)
        # m2 = self.fmv_neg(hc)
        # a = torch.cat([m1, m2], dim=0)
        # mv = torch.spmm(vadj, a)
        # m3 = self.fmc_pos(hv)
        # m4 = self.fmc_neg(hv)
        # b = torch.cat([m3, m4], dim=0)
        # mc = torch.spmm(cadj, b)
        return mv, mc

    def update(self, h, m):
        mv, mc = m
        hv, hc = h
        return (self.fuv(torch.cat((hv, mv), dim=1)), self.fuc(torch.cat((hc, mc), dim=1)))

    def forward(self, h, data):
        m = self.message(h, data)
        h = self.update(h, m)
        return h


class GraphReadout(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            # nn.Sigmoid(),
        )

    def forward(self, h):
        return self.main(h)


class GraphNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        stats = False

        self.conv1 = GraphConv(input_size, hidden_size)
        self.bn1_0 = nn.BatchNorm1d(hidden_size, track_running_stats=stats)
        self.bn1_1 = nn.BatchNorm1d(hidden_size, track_running_stats=stats)

        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.bn2_0 = nn.BatchNorm1d(hidden_size, track_running_stats=stats)
        self.bn2_1 = nn.BatchNorm1d(hidden_size, track_running_stats=stats)

        self.conv3 = GraphConv(hidden_size, hidden_size)
        self.bn3_0 = nn.BatchNorm1d(hidden_size, track_running_stats=stats)
        self.bn3_1 = nn.BatchNorm1d(hidden_size, track_running_stats=stats)

        self.conv4 = GraphConv(hidden_size, hidden_size)
        self.bn4_0 = nn.BatchNorm1d(hidden_size, track_running_stats=stats)
        self.bn4_1 = nn.BatchNorm1d(hidden_size, track_running_stats=stats)

        self.conv5 = GraphConv(hidden_size, hidden_size)
        self.bn5_0 = nn.BatchNorm1d(hidden_size, track_running_stats=stats)
        self.bn5_1 = nn.BatchNorm1d(hidden_size, track_running_stats=stats)

    # @profile
    def forward(self, data):
        h = data.x[:2]

        h = self.conv1(h, data)
        h = self.bn1_0(h[0]), self.bn1_1(h[1])

        h = self.conv2(h, data)
        h = self.bn2_0(h[0]), self.bn2_1(h[1])

        h = self.conv3(h, data)
        h = self.bn3_0(h[0]), self.bn3_1(h[1])

        h = self.conv4(h, data)
        h = self.bn4_0(h[0]), self.bn4_1(h[1])

        h = self.conv5(h, data)
        h = self.bn5_0(h[0]), self.bn5_1(h[1])

        return h


class GraphClassifier(nn.Module):
    def __init__(self, input_size, gnn_hidden_size, readout_hidden_size):
        super().__init__()
        self.gnn = GraphNN(input_size, gnn_hidden_size)
        self.readout = GraphReadout(gnn_hidden_size, 1, readout_hidden_size)

    def forward(self, data):
        h = self.gnn(data)
        vout = sum_batch(self.readout(h[0]), data.idx[0]).squeeze(1)
        cout = sum_batch(self.readout(h[1]), data.idx[1]).squeeze(1)
        vsizes, csizes = data.sizes
        return (vout + cout) / (vsizes + csizes)


def sum_batch(x, idx):
    c = torch.cumsum(x, dim=0)
    c = torch.cat((c.new_zeros((1, c.shape[1])), c), dim=0)
    return c[idx[1:]] - c[idx[:-1]]


class NodeReadout(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size)
        )

    def forward(self, h):
        return self.main(h)


class ReinforcePolicy(nn.Module):
    def __init__(self, input_size, gnn_hidden_size, readout_hidden_size):
        super().__init__()
        self.gnn = GraphNN(input_size, gnn_hidden_size)
        self.policy_readout = NodeReadout(gnn_hidden_size, 1, readout_hidden_size)

    # @profile
    def forward(self, data):
        h = self.gnn(data)
        return self.policy_readout(h[0])


class A2CPolicy(nn.Module):
    def __init__(self, input_size, gnn_hidden_size, readout_hidden_size):
        super().__init__()
        self.gnn = GraphNN(input_size, gnn_hidden_size)
        self.policy_readout = NodeReadout(gnn_hidden_size, 1, readout_hidden_size)
        self.value_readout = GraphReadout(gnn_hidden_size, 1, readout_hidden_size)

    def forward(self, data):
        h = self.gnn(data)
        return (
            self.policy_readout(h[0]),
            self.value_readout(h[1]).sum() / h[1].shape[0],
        )  # Variables too?
