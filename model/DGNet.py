import torch
import torch.nn as nn
import torch.nn.functional as F


class DGNetv2(nn.Module):
    def __init__(self, pre_length, embed_size, feature_size, seq_length, hidden_size, device='cuda:0', layers=2,
                 blocks=6, dim_time_emb=64, dim_graph_emb=64, bi=False, GCN=False, TCN=False, blocks_gate=False,
                 gll=False, is_graph_shared=False, in_dim=1, linear=3, scale=1, alpha=3):
        super().__init__()
        # define variable
        self.matrix_list = []
        self.pre_length = pre_length
        self.embed_size = embed_size
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.layers = layers
        self.blocks = blocks
        self.GCN = GCN
        self.TCN = TCN
        self.blocks_gate = blocks_gate
        self.gll = gll
        self.convolution_type = 'GWNET'
        self.alpha = alpha
        self.beta = nn.Parameter(torch.randn(1), requires_grad=True).to(device)
        # define vectors
        self.time_emb, self.graph_emb = [], []
        self.is_graph_shared = is_graph_shared
        num_graph = 1 if self.is_graph_shared else blocks
        self.mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=0).to(device)
        self.is_mask = True
        for _ in range(num_graph):
            time = nn.Parameter(scale * torch.randn(2, seq_length, dim_time_emb), requires_grad=True).to(device)
            self.time_emb.append(time)
            space = nn.Parameter(scale * torch.randn(2, feature_size, dim_time_emb), requires_grad=True).to(device)
            self.graph_emb.append(space)
        if self.convolution_type == 'GWNET':
            self.DGCN = [DGCN(embed_size, embed_size, order=layers,
                              isTCN=TCN, isGCN=GCN, hidden_size=embed_size, layer=linear, bi=bi) for _ in
                         range(blocks)]
            self.DGCN = nn.Sequential(*self.DGCN)
        self.Linear = MLP(in_dim, embed_size, hidden_dim=hidden_size, hidden_layers=linear, bi=bi)
        # define layers
        self.bn = nn.BatchNorm2d(embed_size)
        if self.blocks_gate:
            self.gate = [
                nn.Sequential(
                    MLP(embed_size, embed_size * 2, hidden_dim=hidden_size, hidden_layers=linear, bi=bi),
                ).to(device)
                for _ in range(blocks)
            ]
            self.gate = nn.Sequential(*self.gate)
        if self.gll:
            self.lin11 = [
                nn.Linear(dim_time_emb, dim_time_emb).to(device)
                for _ in range(blocks)
            ]
            self.lin21 = [
                nn.Linear(dim_time_emb, dim_time_emb).to(device)
                for _ in range(blocks)
            ]
            self.lin12 = [
                nn.Linear(dim_graph_emb, dim_graph_emb).to(device)
                for _ in range(blocks)
            ]
            self.lin22 = [
                nn.Linear(dim_graph_emb, dim_graph_emb).to(device)
                for _ in range(blocks)
            ]
            self.lin11 = nn.Sequential(*self.lin11)
            self.lin21 = nn.Sequential(*self.lin21)
            self.lin12 = nn.Sequential(*self.lin12)
            self.lin22 = nn.Sequential(*self.lin22)
        self.fc = MLP(seq_length * embed_size, pre_length, hidden_dim=hidden_size, hidden_layers=linear, bi=True)
        self.to(device)

    def forward(self, x: torch.Tensor):  # B T N
        x = x.transpose(1, 2)
        if len(x.shape) == 3:
            x = x.unsqueeze(-1)
        x = F.leaky_relu(self.Linear(x))  # B N T F
        self.matrix_list = []
        B, N, L, C = x.shape

        for block in range(self.blocks):
            res = x
            x = self.Polymerization_characteristic(x, block)  # B N T F
            if self.blocks_gate:
                x = self.gate[block](x)
                gate = x[..., C:]
                filter = x[..., :C]
                x = F.leaky_relu(filter) * F.sigmoid(gate)
            x = x + res
        x = x.reshape(B, N, -1)
        x = self.fc(x)
        return x

    def Polymerization_characteristic(self, x, block):
        time_embedding, graph_embedding = self.generate(block)
        self.matrix_list.append((time_embedding, graph_embedding))

        if self.convolution_type == 'GWNET':
            x = self.DGCN[block](x, support=graph_embedding, time=time_embedding)

        return x

    def generate(self, block):
        graph_index = block % len(self.time_emb)
        T1, T2 = self.time_emb[graph_index][0], self.time_emb[graph_index][1]
        A1, A2 = self.graph_emb[graph_index][0], self.graph_emb[graph_index][1]
        if self.gll:
            A1, A2 = self.lin11[block](A1), self.lin12[block](A2)
            T1, T2 = self.lin21[block](T1), self.lin22[block](T2)

        def build(m1, m2, type=3, alpha=3, beta=0.5):
            if not self.gll:
                return torch.mm(m1, m2.T)
            if type == 1:
                return F.relu(torch.mm(m1, m2.T))
            else:
                m1 = F.tanh(alpha * m1)
                m2 = F.tanh(alpha * m2)
                m = torch.mm(m1, m2.T) - beta * torch.mm(m2, m1.T)
                return F.relu(F.tanh(alpha * m))

        T = build(T1, T2, type=1, alpha=self.alpha, beta=self.beta)
        if self.is_mask:
            T = T * self.mask
        A = build(A1, A2, type=3, alpha=self.alpha, beta=self.beta)
        return T, A


class MLP(nn.Module):
    def __init__(self, f_in, f_out, hidden_dim=32, hidden_layers=3, bi=False):
        super(MLP, self).__init__()
        activation = nn.LeakyReLU().to('cuda:0')
        layers = [nn.Linear(f_in, hidden_dim), activation]
        for i in range(hidden_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation]
        layers += [nn.Linear(hidden_dim, f_out, bias=bi)]
        self.layers = nn.Sequential(*layers)
        self.to('cuda:0')

    def forward(self, x):
        y = self.layers(x)
        return y


class DGCN(nn.Module):
    def __init__(self, c_in, c_out, isTCN, isGCN, hidden_size=32, dropout=0.3, order=2, layer=2, bi=False):
        super(DGCN, self).__init__()
        c_in = (order + 1) * c_in
        self.TCN = isTCN
        self.GCN = isGCN
        self.mlp = MLP(c_in, c_out, hidden_size, layer, bi=bi)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support, time):
        out = [x]
        x1 = self.multiplication(x, A=support, T=time)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.multiplication(x, support, time)
            out.append(x2)

        h = torch.cat(out, dim=-1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout)
        return h

    def multiplication(self, x, A=None, T=None):
        x = x.permute(0, 3, 1, 2)  # B F N T
        if not self.TCN and self.GCN:
            x = torch.einsum('bfnt,nw->bfwt', x, A)
        elif self.TCN and not self.GCN:
            x = torch.einsum('qt,bfnt->bfnq', T, x)
        elif not self.TCN and not self.GCN:
            pass
        else:
            x = torch.einsum('qt,bfnt,nw->bfwq', T, x, A)
        return x.permute(0, 2, 3, 1)
