import torch
import torch.nn as nn
import torch.nn.functional as F

class LangeIsotropicMPNN(nn.Module):
    def __init__(self, in_features=4, hidden_dim=64, num_layers=6):
        super().__init__()
        self.node_embed = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.SiLU())
        self.msg_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2 + 4, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.node_embed(x)
        if edge_index.numel() == 0:
            pooled = torch.cat([h.mean(dim=0, keepdim=True), h.max(dim=0, keepdim=True)[0]], dim=-1)
            return self.readout(pooled)
            
        src, dst = edge_index
        for layer, norm in zip(self.msg_layers, self.norms):
            edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)
            msgs = layer(edge_repr)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, msgs)
            h = norm(h + agg)
            
        pooled = torch.cat([h.mean(dim=0, keepdim=True), h.max(dim=0, keepdim=True)[0]], dim=-1)
        return self.readout(pooled)

class NeuralBeliefPropagation(nn.Module):
    def __init__(self, hidden_dim=48, iters=6):
        super().__init__()
        self.iters = iters
        self.var_to_chk = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim))
        self.chk_to_var = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.readout = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, s, edge_index=None, edge_attr=None):
        llr = s * 2.0 - 1.0
        msg = torch.zeros((s.size(0), 48), device=s.device)
        for _ in range(self.iters):
            v = self.var_to_chk(llr) + msg
            c = self.chk_to_var(v)
            msg = self.var_to_chk(c)
        return self.readout(msg.mean(dim=0, keepdim=True))

class SpatioTemporalGNN(nn.Module):
    def __init__(self, in_features=4, hidden_dim=64):
        super().__init__()
        self.spatial = nn.Sequential(nn.Linear(in_features * 2 + 4, hidden_dim), nn.SiLU())
        self.temporal_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.readout = nn.Sequential(nn.Linear(hidden_dim * 2, 1), nn.Sigmoid())

    def forward(self, x, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return torch.tensor([[0.0]], device=x.device)
        src, dst = edge_index
        edge_repr = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        m = self.spatial(edge_repr)
        agg = torch.zeros((x.size(0), m.size(1)), device=x.device)
        agg.index_add_(0, dst, m)
        h = self.norm(self.temporal_gru(agg))
        pooled = torch.cat([h.mean(dim=0, keepdim=True), h.max(dim=0, keepdim=True)[0]], dim=-1)
        return self.readout(pooled)

class AnisotropicRelationalLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.msg_par = nn.Sequential(nn.Linear(hidden_dim * 2 + 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.msg_tra = nn.Sequential(nn.Linear(hidden_dim * 2 + 4, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, src, dst, edge_attr, is_par):
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        msgs = torch.where(is_par.unsqueeze(-1), self.msg_par(edge_repr), self.msg_tra(edge_repr))
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msgs)
        return self.norm(h + agg)

class TopoDephaseGNN(nn.Module):
    def __init__(self, in_features=6, hidden_dim=64, num_layers=6):
        super().__init__()
        self.node_embed = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.SiLU())
        self.layers = nn.ModuleList([AnisotropicRelationalLayer(hidden_dim) for _ in range(num_layers)])
        
        # Head A: Global Logical Coset Classifier
        self.logical_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # Head B: Unbounded Local Physical Edge Logit Estimator
        self.edge_delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # Raw, unconstrained linear log-odds for BCEWithLogitsLoss
        )

    def forward(self, node_feats, edge_index, edge_attr, is_parallel):
        h = self.node_embed(node_feats)
        if edge_index.numel() == 0:
            pooled = torch.cat([h.mean(dim=0, keepdim=True), h.max(dim=0, keepdim=True)[0]], dim=-1)
            return self.logical_head(pooled), torch.zeros((0, 1), device=node_feats.device)
            
        src, dst = edge_index
        for layer in self.layers:
            h = layer(h, src, dst, edge_attr, is_parallel)
            
        pooled = torch.cat([h.mean(dim=0, keepdim=True), h.max(dim=0, keepdim=True)[0]], dim=-1)
        logical_pred = self.logical_head(pooled)
        
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        edge_logits = self.edge_delta_head(edge_repr)
        
        return logical_pred, edge_logits
