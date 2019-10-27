import itertools
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(input_size, output_size, hidden_sizes, activation, output_activation):
    sizes = [input_size, *hidden_sizes, output_size]
    n = len(sizes)
    layers = (
        (nn.Linear(sizes[i - 1], sizes[i]), activation if i < n - 1 else output_activation)
        for i in range(1, n)
    )
    return nn.Sequential(*itertools.chain(*layers))


class AsyncGraphConv(nn.Module):
    def __init__(self, input_size, output_size, mlp_arch):
        super().__init__()

        _mlp = lambda input_size, output_size: mlp(
            input_size,
            output_size,
            mlp_arch['hidden_sizes'],
            eval('nn.' + mlp_arch['activation'] + '()'),
            nn.ReLU(),
        )

        self.fmc_pos = _mlp(input_size, output_size)
        self.fmc_neg = _mlp(input_size, output_size)
        self.fuc = _mlp(input_size + output_size, output_size)

        self.fmv_pos = _mlp(output_size, output_size)
        self.fmv_neg = _mlp(output_size, output_size)
        self.fuv = _mlp(input_size + output_size, output_size)

    # @profile
    def forward(self, h, data):
        vadj, cadj = data.adj
        hv, hc = h

        mc = torch.spmm(
            cadj, torch.cat([self.fmc_pos(hv.clone()), self.fmc_neg(hv.clone())], dim=0)
        )
        hc = self.fuc(torch.cat((hc, mc), dim=1))

        mv = torch.spmm(
            vadj, torch.cat([self.fmv_pos(hc.clone()), self.fmv_neg(hc.clone())], dim=0)
        )
        hv = self.fuv(torch.cat((hv, mv), dim=1))

        return hv, hc


class GraphConv(nn.Module):
    def __init__(self, input_size, output_size, mlp_arch):
        super().__init__()

        _mlp = lambda input_size, output_size: mlp(
            input_size,
            output_size,
            mlp_arch['hidden_sizes'],
            eval('nn.' + mlp_arch['activation'] + '()'),
            nn.ReLU(),
        )

        self.fmv_pos = _mlp(input_size, output_size)
        self.fmc_pos = _mlp(input_size, output_size)
        self.fmv_neg = _mlp(input_size, output_size)
        self.fmc_neg = _mlp(input_size, output_size)
        self.fuv = _mlp(input_size + output_size, output_size)
        self.fuc = _mlp(input_size + output_size, output_size)

    def message(self, h, data):
        vadj, cadj = data.adj
        hv, hc = h

        mv = torch.spmm(
            vadj, torch.cat([self.fmv_pos(hc.clone()), self.fmv_neg(hc.clone())], dim=0)
        )
        mc = torch.spmm(
            cadj, torch.cat([self.fmc_pos(hv.clone()), self.fmc_neg(hv.clone())], dim=0)
        )
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


class GCBN(nn.Module):
    def __init__(self, input_size, hidden_size, mlp_arch, gnn_async):
        super().__init__()
        self.conv = (
            GraphConv(input_size, hidden_size, mlp_arch)
            if not gnn_async
            else AsyncGraphConv(input_size, hidden_size, mlp_arch)
        )
        self.bn_0 = nn.BatchNorm1d(hidden_size, track_running_stats=False)
        self.bn_1 = nn.BatchNorm1d(hidden_size, track_running_stats=False)

    def forward(self, h, data):
        h = self.conv(h, data)
        h = self.bn_0(h[0]), self.bn_1(h[1])
        return h


class GraphNN(nn.Module):
    def __init__(self, input_size, hidden_size, mlp_arch, gnn_iter, gnn_async):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                GCBN(input_size if i == 0 else hidden_size, hidden_size, mlp_arch, gnn_async)
                for i in range(gnn_iter)
            ]
        )

    def forward(self, data):
        h = data.x[:2]
        for conv in self.convs:
            h = conv(h, data)
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
    def __init__(
        self, input_size, gnn_hidden_size, readout_hidden_size, mlp_arch, gnn_iter, gnn_async
    ):
        super().__init__()
        self.gnn = GraphNN(input_size, gnn_hidden_size, mlp_arch, gnn_iter, gnn_async)
        self.policy_readout = NodeReadout(gnn_hidden_size, 1, readout_hidden_size)

    # @profile
    def forward(self, data):
        h = self.gnn(data)
        return self.policy_readout(h[0])


class PGPolicy(nn.Module):
    def __init__(
        self, input_size, gnn_hidden_size, readout_hidden_size, mlp_arch, gnn_iter, gnn_async
    ):
        super().__init__()
        self.gnn = GraphNN(input_size, gnn_hidden_size, mlp_arch, gnn_iter, gnn_async)
        self.policy_readout = NodeReadout(gnn_hidden_size, 1, readout_hidden_size)
        self.value_readout = GraphReadout(gnn_hidden_size, 1, readout_hidden_size)

    # @profile
    def forward(self, data):
        h = self.gnn(data)
        return (
            self.policy_readout(h[0]),
            self.value_readout(h[0]).mean(),
        )


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
