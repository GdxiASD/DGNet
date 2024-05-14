import torch
import torch.nn as nn
import torch.nn.functional as F


class DGNetv2(nn.Module):
    def __init__(self, pre_length, embed_size, feature_size, seq_length, hidden_size, device='cuda:0', layers=2,
                 blocks=2, dim_time_emb=10, dim_graph_emb=10, bi=False, GCN=True, TCN=True, blocks_gate=True,
                 graph_regenerate=True, in_dim=1, linear=3, scale=0.02, is_graph_shared=False, alpha=3, beta=0.5):
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
        self.graph_regenerate = graph_regenerate
        self.convolution_type = 'GWNET'
        self.alpha = alpha
        self.beta = nn.Parameter(torch.randn(1), requires_grad=True).to(device)
        # define vectors
        self.time_emb, self.graph_emb = [], []
        # seq_length = seq_length // 2 + 1
        self.is_graph_shared = is_graph_shared
        num_graph = 1 if self.is_graph_shared else blocks
        self.mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=0).to(device)
        self.is_mask = True
        for _ in range(num_graph):
            time = nn.Parameter(scale * torch.randn(2, seq_length, dim_time_emb), requires_grad=True).to(device)
            # torch.nn.init.xavier_normal_(time)
            self.time_emb.append(time)
            space = nn.Parameter(scale * torch.randn(2, feature_size, dim_time_emb), requires_grad=True).to(device)
            # torch.nn.init.xavier_normal_(space)
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
        if self.gsl:
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

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None,
                epoch: int = None, train: bool = None, **kwargs):  # B T N
        x = history_data.transpose(1, 2)
        if len(x.shape) == 3: x = x.unsqueeze(-1)
        x = F.leaky_relu(self.Linear(x))  # B N T F
        self.matrix_list = []
        B, N, L, C = x.shape
        # x = torch.fft.rfft(x, dim=2, norm='ortho').real

        for block in range(self.blocks):
            res = x
            x = self.Polymerization_characteristic(x, block)  # B N T F
            if self.blocks_gate:
                x = self.gate[block](x)
                # GPT认为BN放在这里
                gate = x[..., C:]
                # gate = self.bn(gate.transpose(1, 3)).transpose(1, 3)
                filter = x[..., :C]
                # filter = self.bn(filter.transpose(1, 3)).transpose(1, 3)
                x = F.leaky_relu(filter) * F.sigmoid(gate)
            x = x + res
            # x = self.bn(x.transpose(1, 3)).transpose(1, 3)
        # 2024.4.6 添加BN，一般
        # x = self.bn(x.transpose(1, 3)).transpose(1, 3)
        # x = torch.fft.irfft(torch.view_as_complex(x), n=N * L, dim=2, norm="ortho")
        x = x.reshape(B, N, -1)
        x = self.fc(x)
        return x
        return x.unsqueeze(-1).transpose(1, 2)

    def Polymerization_characteristic(self, x, block):
        time_embedding, graph_embedding = self.generate(block)
        self.matrix_list.append((time_embedding, graph_embedding))

        if self.convolution_type == 'GWNET':  # 2024.4.5两种方式相差不大
            x = self.DGCN[block](x, support=graph_embedding, time=time_embedding)
            # 2024.4.5 添加BN，未测试
            # 2024.4.6 测试，变差
            # x = self.bn(x.transpose(1, 3)).transpose(1, 3)
        else:
            for _ in range(self.layers):
                # layer _ B N T F
                x = x + self.multiplication(x, A=graph_embedding, T=time_embedding)
                x = self.bn(x.transpose(1, 3)).transpose(1, 3)  # 2024.3.21 测试有效果
                x = F.leaky_relu(x)

        return x

    def generate(self, block):
        graph_index = block % len(self.time_emb)
        T1, T2 = self.time_emb[graph_index][0], self.time_emb[graph_index][1]
        A1, A2 = self.graph_emb[graph_index][0], self.graph_emb[graph_index][1]
        if self.graph_regenerate:
            A1, A2 = self.lin11[block](A1), self.lin12[block](A2)
            # A1, A2 = F.leaky_relu(A1), F.leaky_relu(A2)
            T1, T2 = self.lin21[block](T1), self.lin22[block](T2)
            # T1, T2 = F.leaky_relu(T1), F.leaky_relu(T2)

        def build(m1, m2, type=3, alpha=3, beta=0.5, dim=1):
            if not self.graph_regenerate:
                return torch.mm(m1, m2.T)
            if type == 1:
                return F.relu(torch.mm(m1, m2.T))
            else:
                m1 = F.tanh(alpha * m1)  # 保证值域在（-1，1）梯度不消失
                m2 = F.tanh(alpha * m2)
                m = torch.mm(m1, m2.T) - beta * torch.mm(m2, m1.T)  # 消除对称性
                return F.relu(F.tanh(alpha * m))

        T = build(T1, T2, type=1, alpha=self.alpha, beta=self.beta, dim=1)
        if self.is_mask:
            T = T * self.mask
        A = build(A1, A2, type=3, alpha=self.alpha, beta=self.beta, dim=0)
        # 2024.3.20 认为时图对按行归一化，空图安列归一化
        return T, A

    def multiplication(self, x, A, T=None):
        if not self.TCN and self.GCN:
            return torch.einsum('bntf,nw->bwtf', x, A)
        elif self.TCN and not self.GCN:
            return torch.einsum('qt,bntf->bnqf', T, x)
        elif not self.TCN and not self.GCN:
            return x
        else:
            return torch.einsum('qt,bntf,nw->bwqf', T, x, A)


class MLP(nn.Module):
    def __init__(self, f_in, f_out, hidden_dim=128, hidden_layers=3, dropout=0.3, bi=False):
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
    def __init__(self, c_in, c_out, isTCN, isGCN, hidden_size=64, dropout=0.3, order=2, layer=2, bi=False):
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
