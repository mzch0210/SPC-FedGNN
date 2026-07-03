import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def require_pyg():
    try:
        from torch_geometric.datasets import Amazon, Planetoid, HeterophilousGraphDataset
        from torch_geometric.nn import GCNConv
        from torch_geometric.utils import dense_to_sparse, to_dense_adj
    except Exception as exc:
        raise SystemExit(
            "PyTorch Geometric dependencies are missing. Run scripts/bootstrap_server.sh on the GPU server."
        ) from exc
    return Amazon, Planetoid, HeterophilousGraphDataset, GCNConv, dense_to_sparse, to_dense_adj


def macro_f1_torch(y_true, y_pred, classes):
    vals = []
    for cls in range(classes):
        tp = ((y_true == cls) & (y_pred == cls)).sum().item()
        fp = ((y_true != cls) & (y_pred == cls)).sum().item()
        fn = ((y_true == cls) & (y_pred != cls)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        vals.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(vals))


def load_dataset(name, root):
    Amazon, Planetoid, HeterophilousGraphDataset, _, _, _ = require_pyg()
    lname = name.lower()
    if lname in {"cora", "citeseer", "pubmed"}:
        planetoid_name = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}[lname]
        dataset = Planetoid(root=str(root / "Planetoid"), name=planetoid_name)
    elif lname in {"amazon_photo", "photo"}:
        dataset = Amazon(root=str(root / "Amazon"), name="Photo")
    elif lname in {"amazon_computers", "computers"}:
        dataset = Amazon(root=str(root / "Amazon"), name="Computers")
    elif lname in {"roman_empire", "amazon_ratings", "minesweeper", "tolokers", "questions"}:
        map_name = {
            "roman_empire": "Roman-empire",
            "amazon_ratings": "Amazon-ratings",
            "minesweeper": "Minesweeper",
            "tolokers": "Tolokers",
            "questions": "Questions",
        }[lname]
        dataset = HeterophilousGraphDataset(root=str(root / "Heterophilous"), name=map_name)
    elif lname == "ogbn_arxiv":
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except Exception as exc:
            raise SystemExit("Install ogb on the server to use ogbn-arxiv.") from exc
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(root / "OGB"))
    else:
        raise SystemExit(f"Unknown dataset: {name}")
    data = dataset[0]
    if data.y.dim() > 1:
        data.y = data.y.view(-1)
    data.num_classes = int(data.y.max().item() + 1)
    return data


def split_train_test(n, seed, train_ratio=0.55, test_ratio=0.30):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train = idx[: int(train_ratio * n)]
    test = idx[int((1.0 - test_ratio) * n) :]
    return train, test


def make_clients(data, clients, partition, seed):
    n = data.num_nodes
    labels = data.y.cpu().numpy()
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(clients)]
    if partition.startswith("dirichlet"):
        alpha = float(partition.split("_")[-1])
        for cls in np.unique(labels):
            nodes = np.where(labels == cls)[0]
            rng.shuffle(nodes)
            probs = rng.dirichlet(np.ones(clients) * alpha)
            cuts = (np.cumsum(probs)[:-1] * len(nodes)).astype(int)
            for cid, part in enumerate(np.split(nodes, cuts)):
                buckets[cid].extend(part.tolist())
    else:
        order = np.arange(n)
        if partition in {"louvain", "metis"}:
            degree_hint = torch.bincount(data.edge_index[0], minlength=n).cpu().numpy()
            order = np.lexsort((rng.random(n), degree_hint))
        else:
            rng.shuffle(order)
        for cid, part in enumerate(np.array_split(order, clients)):
            buckets[cid].extend(part.tolist())

    client_rows = []
    for cid, nodes in enumerate(buckets):
        nodes = np.array(sorted(set(nodes)), dtype=np.int64)
        if nodes.size < 5:
            nodes = rng.choice(n, size=min(20, n), replace=False)
        train_local, test_local = split_train_test(nodes.size, seed + cid)
        client_rows.append(
            {
                "client_id": cid,
                "nodes": torch.as_tensor(nodes, dtype=torch.long),
                "train_nodes": torch.as_tensor(nodes[train_local], dtype=torch.long),
                "test_nodes": torch.as_tensor(nodes[test_local], dtype=torch.long),
            }
        )
    return client_rows


def stratified_subset_indices(labels, ratio, rng, min_per_class=1):
    selected = []
    labels_np = labels.detach().cpu().numpy()
    for cls in sorted(set(int(x) for x in labels_np.tolist())):
        positions = np.where(labels_np == cls)[0]
        if positions.size == 0:
            continue
        rng.shuffle(positions)
        take = max(min_per_class, int(np.ceil(positions.size * max(min(ratio, 1.0), 0.0))))
        selected.extend(positions[: min(take, positions.size)].tolist())
    if not selected and labels.numel() > 0:
        selected = [0]
    return np.array(sorted(set(selected)), dtype=np.int64)


def prepare_gate_evidence_splits(args, data, clients):
    """Optionally reserve a disjoint local subset for transfer-gate decisions."""
    mode = getattr(args, "guard_evidence_mode", "train")
    if mode == "train":
        for c in clients:
            c["supervised_nodes"] = c["train_nodes"]
            c["gate_evidence_nodes"] = c["train_nodes"]
        return
    ratio = float(getattr(args, "guard_independent_ratio", 0.2))
    for c in clients:
        train_nodes = c["train_nodes"]
        labels = data.y[train_nodes.to(data.y.device)]
        rng = np.random.default_rng(int(args.seed) * 1543 + int(c["client_id"]) * 7919 + int(ratio * 1000))
        gate_pos = stratified_subset_indices(labels, ratio, rng, min_per_class=1)
        all_pos = np.arange(train_nodes.numel())
        supervised_pos = np.setdiff1d(all_pos, gate_pos, assume_unique=False)
        if supervised_pos.size == 0:
            supervised_pos = all_pos
        c["supervised_nodes"] = train_nodes[torch.as_tensor(supervised_pos, dtype=torch.long)]
        c["gate_evidence_nodes"] = train_nodes[torch.as_tensor(gate_pos, dtype=torch.long)]


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, classes, dropout):
        super().__init__()
        _, _, _, GCNConv, _, _ = require_pyg()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, return_hidden=False):
        hidden = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        x = F.dropout(hidden, p=self.dropout, training=self.training)
        logits = self.conv2(x, edge_index, edge_weight=edge_weight)
        if return_hidden:
            return logits, hidden
        return logits


class EdgeEvaluator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return torch.sigmoid(self.net(features)).view(-1)


def hsic_loss(x, y):
    n = x.shape[0]
    if n < 2:
        return x.new_tensor(0.0)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cov = x.T @ y / max(n - 1, 1)
    return cov.pow(2).mean()


def class_balanced_cross_entropy(logits, labels, classes, strength=1.0):
    counts = torch.bincount(labels, minlength=classes).float().to(logits.device)
    present = counts > 0
    weights = torch.ones(classes, dtype=torch.float32, device=logits.device)
    if present.any():
        inv = counts[present].sum() / (counts[present] * float(present.sum()))
        weights[present] = inv.clamp(max=10.0).pow(strength)
        weights = weights / weights[present].mean().clamp(min=1e-8)
    return F.cross_entropy(logits, labels, weight=weights)


def label_coverage_stats(data, nodes):
    labels = data.y[nodes].detach().cpu()
    counts = torch.bincount(labels, minlength=int(data.num_classes)).numpy()
    present = counts[counts > 0]
    return {
        "num_nodes": int(nodes.numel()),
        "num_classes": int((counts > 0).sum()),
        "min_class_count": int(present.min()) if present.size else 0,
        "max_class_count": int(present.max()) if present.size else 0,
    }


def client_label_distribution(data, nodes):
    nodes = nodes.to(data.y.device)
    labels = data.y[nodes].detach().cpu()
    counts = torch.bincount(labels, minlength=int(data.num_classes)).float()
    total = counts.sum().clamp(min=1.0)
    return (counts / total).numpy()


def js_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    eps = 1e-12
    kl_pm = np.sum(np.where(p > 0, p * np.log((p + eps) / (m + eps)), 0.0))
    kl_qm = np.sum(np.where(q > 0, q * np.log((q + eps) / (m + eps)), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def candidate_label_misalignment(data, clients, weights, candidate_count):
    train_dists = [client_label_distribution(data, c["train_nodes"]) for c in clients]
    dominant = [int(np.argmax(dist)) for dist in train_dists]
    out = {}
    for cid in range(len(clients)):
        order = np.argsort(-weights[cid])
        candidates = [int(j) for j in order if int(j) != cid][: max(candidate_count, 0)]
        if not candidates:
            out[cid] = {
                "candidate_label_conflict_rate": 0.0,
                "candidate_label_js_mean": 0.0,
                "candidate_count": 0,
            }
            continue
        conflicts = [float(dominant[j] != dominant[cid]) for j in candidates]
        divergences = [js_divergence(train_dists[cid], train_dists[j]) for j in candidates]
        out[cid] = {
            "candidate_label_conflict_rate": float(np.mean(conflicts)),
            "candidate_label_js_mean": float(np.mean(divergences)),
            "candidate_count": int(len(candidates)),
        }
    return out


def empty_candidate_misalignment(clients):
    return {
        c["client_id"]: {
            "candidate_label_conflict_rate": np.nan,
            "candidate_label_js_mean": np.nan,
            "candidate_count": 0,
        }
        for c in clients
    }


def fedpub_proxy_nodes(data, max_nodes, seed):
    rng = np.random.default_rng(seed + 2029)
    labels = data.y.detach().cpu().numpy()
    classes = int(data.num_classes)
    per_class = max(1, int(max_nodes) // max(classes, 1))
    selected = []
    for cls in range(classes):
        cls_nodes = np.where(labels == cls)[0]
        if cls_nodes.size == 0:
            continue
        rng.shuffle(cls_nodes)
        selected.extend(cls_nodes[:per_class].tolist())
    if len(selected) < max_nodes:
        remaining = np.setdiff1d(np.arange(data.num_nodes), np.array(selected, dtype=np.int64), assume_unique=False)
        rng.shuffle(remaining)
        selected.extend(remaining[: max_nodes - len(selected)].tolist())
    return torch.as_tensor(sorted(set(selected[:max_nodes])), dtype=torch.long, device=data.x.device)


def fedpub_functional_embeddings(data, proxy_nodes, local_states, args):
    embeddings = []
    for state in local_states:
        model = GCN(data.num_features, args.hidden_dim, data.num_classes, args.dropout).to(data.x.device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            probs = model(data.x, data.edge_index)[proxy_nodes].softmax(dim=1)
            emb = probs.mean(dim=0)
            emb = emb / emb.norm(p=2).clamp(min=1e-8)
        embeddings.append(emb.detach().cpu().numpy())
    return np.stack(embeddings, axis=0)


def fedpub_similarity_weights(embeddings, temperature):
    sim = np.matmul(embeddings, embeddings.T)
    sim = sim - sim.max(axis=1, keepdims=True)
    weights = np.exp(float(temperature) * sim)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights


def graph_descriptor(data, nodes, mode="structure"):
    device = data.edge_index.device
    nodes = nodes.to(device)
    src, dst = data.edge_index
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    node_mask[nodes] = True
    local_mask = node_mask[src] & node_mask[dst]
    sub_src = src[local_mask]
    sub_dst = dst[local_mask]
    deg = torch.bincount(sub_src, minlength=data.num_nodes)[nodes].float()
    n = max(int(nodes.numel()), 1)
    edge_count = float(sub_src.numel())
    density = edge_count / max(n * (n - 1), 1)
    smoothness = 0.0
    if sub_src.numel() > 0:
        max_smooth_edges = 20000
        if sub_src.numel() > max_smooth_edges:
            step = max(int(sub_src.numel() // max_smooth_edges), 1)
            sample_idx = torch.arange(0, sub_src.numel(), step, device=device)[:max_smooth_edges]
            smooth_src = sub_src[sample_idx]
            smooth_dst = sub_dst[sample_idx]
        else:
            smooth_src = sub_src
            smooth_dst = sub_dst
        diff = data.x[smooth_src].float() - data.x[smooth_dst].float()
        smoothness = float(diff.pow(2).sum(dim=1).mean().item())
    desc_values = [
        deg.mean().item(),
        deg.std(unbiased=False).item(),
        float(deg.max().item()) if deg.numel() else 0.0,
        density,
        edge_count / n,
        len(nodes) / data.num_nodes,
        smoothness,
    ]
    if mode == "with_label":
        labels = data.y[nodes].detach().cpu().numpy()
        if sub_src.numel() > 0:
            same = (data.y[sub_src] == data.y[sub_dst]).float().mean().item()
        else:
            same = 0.0
        hist = np.bincount(labels, minlength=int(data.num_classes)).astype(float)
        hist = hist / max(hist.sum(), 1.0)
        desc_values.extend([same, *hist[:8]])
    desc = np.array(desc_values, dtype=float)
    return desc / max(np.linalg.norm(desc), 1e-8)


def prototype_bank(data, clients, count, noise, seed, descriptor_mode="structure"):
    rng = np.random.default_rng(seed)
    desc = np.stack([graph_descriptor(data, c["nodes"], mode=descriptor_mode) for c in clients])
    if noise > 0:
        desc = desc + rng.normal(0, noise, desc.shape)
    centers = desc[rng.choice(desc.shape[0], count, replace=True)].copy()
    for _ in range(8):
        dist = ((desc[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = dist.argmin(axis=1)
        for k in range(count):
            if np.any(assign == k):
                centers[k] = desc[assign == k].mean(axis=0)
    return desc, centers


def prototype_similarity_weights(descriptors, temperature=8.0):
    dist = ((descriptors[:, None, :] - descriptors[None, :, :]) ** 2).sum(axis=2)
    logits = -temperature * dist
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights


def mix_with_difficulty(proto_weights, difficulty_weights):
    adjusted = proto_weights * difficulty_weights[None, :]
    return adjusted / np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-12)


def client_edge_weight(data, client, descriptor, centers, method, mask_keep_ratio):
    src, dst = data.edge_index
    device = src.device
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    node_mask[client["nodes"].to(device)] = True
    local_mask = node_mask[src] & node_mask[dst]
    edge_weight = local_mask.float()
    if method in {
        "fedavg_gcn",
        "fedprox_gcn",
        "fedpub_gcn",
        "fedssp",
        "spc_protoagg",
        "spc_random_prototypes",
        "spc_no_protoagg",
        "spc_no_worst",
        "spc_label_descriptor",
    }:
        return edge_weight

    local_idx = torch.where(local_mask)[0]
    if local_idx.numel() == 0:
        return edge_weight

    x = F.normalize(data.x.float(), p=2, dim=1)
    sim = (x[src[local_idx]] * x[dst[local_idx]]).sum(dim=1)
    deg = torch.bincount(src[local_mask], minlength=data.num_nodes).float()
    deg_score = (deg[src[local_idx]] + deg[dst[local_idx]]) / max(float(deg.max().item()) * 2.0, 1.0)

    if method in {"fedath", "spc_no_prototypes"}:
        scores = 0.82 * sim + 0.18 * deg_score
        keep_ratio = mask_keep_ratio
    else:
        dist = ((centers - descriptor[None, :]) ** 2).sum(axis=1)
        proto = centers[int(dist.argmin())]
        proto_degree = float(proto[0]) if proto.size > 0 else 0.0
        proto_homophily = float(proto[2]) if proto.size > 2 else 0.0
        scores = (0.72 + 0.08 * proto_homophily) * sim + (0.20 + 0.08 * proto_degree) * deg_score
        keep_ratio = float(np.clip(mask_keep_ratio + 0.08 * (proto_homophily - descriptor[2]), 0.30, 0.85))

    keep_count = max(1, int(local_idx.numel() * keep_ratio))
    threshold = torch.topk(scores, keep_count, largest=True).values.min()
    selected = local_idx[scores >= threshold]
    masked = torch.zeros_like(edge_weight)
    masked[selected] = 1.0
    return masked


def prepare_client_edges(data, client, descriptor, centers, method, mask_keep_ratio):
    src, dst = data.edge_index
    device = src.device
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    node_mask[client["nodes"].to(device)] = True
    local_mask = node_mask[src] & node_mask[dst]
    local_idx = torch.where(local_mask)[0]
    base_weight = local_mask.float()

    if local_idx.numel() == 0:
        client["local_edge_idx"] = local_idx
        client["base_edge_weight"] = base_weight
        client["edge_features_spc"] = torch.zeros((0, 4 + 2 * len(descriptor)), device=device)
        client["edge_features_fedath"] = torch.zeros((0, 4), device=device)
        client["alignment_target"] = 0.0
        return

    x = F.normalize(data.x.float(), p=2, dim=1)
    sim = (x[src[local_idx]] * x[dst[local_idx]]).sum(dim=1, keepdim=True)
    deg = torch.bincount(src[local_mask], minlength=data.num_nodes).float()
    deg_score = (deg[src[local_idx]] + deg[dst[local_idx]]) / max(float(deg.max().item()) * 2.0, 1.0)
    deg_score = deg_score.view(-1, 1)
    same_pred = (sim > sim.median()).float()
    bias = torch.ones_like(sim)
    structural = torch.cat([sim, deg_score, same_pred, bias], dim=1)

    dist = ((centers - descriptor[None, :]) ** 2).sum(axis=1)
    proto = centers[int(dist.argmin())]
    if method == "spc_no_prototypes":
        proto = np.zeros_like(proto)
    desc_t = torch.as_tensor(descriptor, dtype=torch.float32, device=device).view(1, -1).repeat(local_idx.numel(), 1)
    proto_t = torch.as_tensor(proto, dtype=torch.float32, device=device).view(1, -1).repeat(local_idx.numel(), 1)

    client["local_edge_idx"] = local_idx
    client["base_edge_weight"] = base_weight
    client["edge_features_fedath"] = structural
    client["edge_features_spc"] = torch.cat([structural, desc_t, proto_t], dim=1)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask[client["train_nodes"].to(device)] = True
    supervised = train_mask[src[local_idx]] & train_mask[dst[local_idx]]
    client["edge_supervised_mask"] = supervised
    if supervised.any():
        same_label = (data.y[src[local_idx]] == data.y[dst[local_idx]]).float()
        client["edge_supervised_target"] = same_label
    else:
        client["edge_supervised_target"] = torch.zeros(local_idx.numel(), dtype=torch.float32, device=device)
    proto_homophily = float(proto[2]) if proto.size > 2 else 0.0
    proto_prior = torch.sigmoid(3.0 * (sim.view(-1) - sim.mean()) + proto_homophily - 0.5).detach()
    client["edge_proto_prior"] = proto_prior
    proto_homophily = float(proto[2]) if proto.size > 2 else 0.0
    descriptor_homophily = float(descriptor[2]) if len(descriptor) > 2 else 0.0
    client["alignment_target"] = float(np.clip(mask_keep_ratio + 0.10 * (proto_homophily - descriptor_homophily), 0.30, 0.85))


def prepare_client_base_edges(data, client):
    src, dst = data.edge_index
    device = src.device
    node_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    node_mask[client["nodes"].to(device)] = True
    local_mask = node_mask[src] & node_mask[dst]
    client["local_edge_idx"] = torch.where(local_mask)[0]
    client["base_edge_weight"] = local_mask.float()


def evaluator_edge_weight(client, evaluator, use_spc_features=True):
    base = client["base_edge_weight"]
    local_idx = client["local_edge_idx"]
    if local_idx.numel() == 0:
        return base, base
    features = client["edge_features_spc"] if use_spc_features else client["edge_features_fedath"]
    scores = torch.nan_to_num(evaluator(features), nan=0.5, posinf=1.0, neginf=0.0).clamp(min=1e-6, max=1.0 - 1e-6)
    invariant = torch.zeros_like(base)
    invariant[local_idx] = scores
    biased = torch.zeros_like(base)
    biased[local_idx] = (1.0 - scores).clamp(min=0.0, max=1.0)
    return invariant, biased


def method_flags(method):
    protoagg_methods = {
        "spc_protoagg",
        "spc_random_prototypes",
        "spc_no_worst",
        "spc_label_descriptor",
    }
    guard_methods = {
        "spc_guard",
        "spc_guard_pp",
        "spc_guard_pp_no_coverage",
        "spc_guard_pp_no_class_balance",
        "spc_guard_cb",
        "spc_guard_no_gate",
        "spc_guard_no_distill",
        "spc_guard_random_candidates",
    }
    class_conditional_methods = {
        "spc_guard_pp",
        "spc_guard_pp_no_coverage",
        "spc_guard_pp_no_class_balance",
    }
    return {
        "use_prototypes": method not in {
            "fedavg_gcn",
            "fedavg_gcn_cb",
            "fedprox_gcn",
            "fedprox_gcn_cb",
            "fedpub_gcn",
            "fedpub_gcn_cb",
            "fedssp",
            "fedath",
            "fedath_cb",
            "spc_no_prototypes",
            "spc_no_protoagg",
        },
        "personalized": method not in {
            "fedavg_gcn",
            "fedavg_gcn_cb",
            "fedprox_gcn",
            "fedprox_gcn_cb",
            "fedpub_gcn",
            "fedpub_gcn_cb",
            "spc_no_personalization",
            *protoagg_methods,
        },
        "functional_personalization": method in {"fedpub_gcn", "fedpub_gcn_cb"},
        "prototype_aggregation": method in protoagg_methods,
        "guarded_distillation": method in guard_methods,
        "class_conditional_guard": method in class_conditional_methods,
        "guard_gate": method in {"spc_guard", "spc_guard_cb", "spc_guard_no_distill", "spc_guard_random_candidates"},
        "guard_distill": method in {
            "spc_guard",
            "spc_guard_pp",
            "spc_guard_pp_no_coverage",
            "spc_guard_pp_no_class_balance",
            "spc_guard_cb",
            "spc_guard_no_gate",
            "spc_guard_random_candidates",
        },
        "class_balanced_loss": method in {
            "spc_guard",
            "spc_guard_pp",
            "spc_guard_pp_no_coverage",
            "spc_guard_cb",
            "fedavg_gcn_cb",
            "fedprox_gcn_cb",
            "fedpub_gcn_cb",
            "fedath_cb",
        },
        "worst_reweight": method in {
            "spc_protoagg",
            "spc_random_prototypes",
            "spc_label_descriptor",
            "spc_no_protoagg",
            "spc_guard",
            "spc_guard_pp",
            "spc_guard_pp_no_coverage",
            "spc_guard_pp_no_class_balance",
            "spc_guard_cb",
            "spc_guard_no_gate",
            "spc_guard_no_distill",
            "spc_guard_random_candidates",
        },
        "residual_gating": method in {
            "spc_fedgnn",
            "spc_no_prototypes",
            "spc_no_hsic",
            "spc_no_alignment",
            "spc_no_personalization",
        },
        "edge_evaluator": method in {
            "fedath",
            "fedath_cb",
            "spc_fedgnn",
            "spc_no_prototypes",
            "spc_no_hsic",
            "spc_no_alignment",
            "spc_no_personalization",
        },
        "hsic": method not in {
            "fedavg_gcn",
            "fedavg_gcn_cb",
            "fedprox_gcn",
            "fedprox_gcn_cb",
            "fedpub_gcn",
            "fedpub_gcn_cb",
            "fedssp",
            "spc_no_hsic",
        },
        "prox": method in {"fedprox_gcn", "fedprox_gcn_cb"},
        "counterfactual": method not in {
            "fedavg_gcn",
            "fedavg_gcn_cb",
            "fedprox_gcn",
            "fedprox_gcn_cb",
            "fedpub_gcn",
            "fedpub_gcn_cb",
            "fedssp",
            "spc_no_counterfactual",
            "spc_protoagg",
            "spc_no_protoagg",
            "spc_no_worst",
            "spc_random_prototypes",
            "spc_label_descriptor",
            "spc_guard",
            "spc_guard_pp",
            "spc_guard_pp_no_coverage",
            "spc_guard_pp_no_class_balance",
            "spc_guard_cb",
            "spc_guard_no_gate",
            "spc_guard_no_distill",
            "spc_guard_random_candidates",
        },
        "alignment": method not in {
            "fedavg_gcn",
            "fedavg_gcn_cb",
            "fedprox_gcn",
            "fedprox_gcn_cb",
            "fedpub_gcn",
            "fedpub_gcn_cb",
            "fedath",
            "fedath_cb",
            "spc_no_alignment",
            "spc_protoagg",
            "spc_guard",
            "spc_guard_pp",
            "spc_guard_pp_no_coverage",
            "spc_guard_pp_no_class_balance",
            "spc_guard_cb",
            "spc_guard_no_gate",
            "spc_guard_no_distill",
            "spc_guard_random_candidates",
        },
    }


def guard_candidate_weights(args, flags, data, cid, train_nodes, gate_evidence_nodes, global_state, prev_local_states, proto_agg_weights):
    device = data.x.device
    validation_nodes = guard_validation_nodes(args, data, cid, gate_evidence_nodes)
    candidate_weights = proto_agg_weights[cid].copy()
    candidate_weights[cid] = 0.0
    empty_stats = {
        "guard_candidates": 0,
        "guard_accepted": 0,
        "guard_accept_rate": 0.0,
        "guard_mean_gate": 0.0,
        "guard_mean_delta": 0.0,
        "guard_mean_uncertainty": 0.0,
        "guard_base_loss": 0.0,
        "guard_distill_scale": 0.0,
    }
    if candidate_weights.sum() <= 1e-12:
        return None, 0.0, empty_stats, None
    top_count = min(args.guard_candidate_count, len(candidate_weights) - 1)
    candidate_ids = np.argsort(-candidate_weights)[:top_count]
    if top_count <= 0:
        return None, 0.0, empty_stats, None

    base = GCN(data.num_features, args.hidden_dim, data.num_classes, args.dropout).to(device)
    base.load_state_dict(global_state)
    base.eval()
    with torch.no_grad():
        base_logits = base(data.x, data.edge_index)
        base_loss = F.cross_entropy(base_logits[validation_nodes], data.y[validation_nodes]).item()

    gated_weights = []
    teacher_probs = []
    gates = []
    deltas = []
    uncertainties = []
    for jid in candidate_ids:
        teacher = GCN(data.num_features, args.hidden_dim, data.num_classes, args.dropout).to(device)
        teacher.load_state_dict(prev_local_states[int(jid)])
        teacher.eval()
        with torch.no_grad():
            logits = teacher(data.x, data.edge_index)
            teacher_loss = F.cross_entropy(logits[validation_nodes], data.y[validation_nodes]).item()
            probs = logits[validation_nodes].softmax(dim=1)
            uncertainty = float((-(probs * probs.clamp(min=1e-8).log()).sum(dim=1)).mean().item())
        if flags["guard_gate"]:
            delta = base_loss - teacher_loss
            adaptive_threshold = args.guard_threshold + args.guard_loss_threshold_weight * min(base_loss, 5.0)
            gate = 1.0 / (1.0 + np.exp(-(args.guard_beta * delta - args.guard_uncertainty_weight * uncertainty)))
        else:
            delta = base_loss - teacher_loss
            adaptive_threshold = args.guard_threshold
            gate = 1.0
        gates.append(float(gate))
        deltas.append(float(delta))
        uncertainties.append(float(uncertainty))
        weight = float(candidate_weights[int(jid)] * gate)
        if gate >= adaptive_threshold and weight > 1e-12:
            gated_weights.append(weight)
            teacher_probs.append(probs.detach())

    accepted = len(teacher_probs)
    protect_scale = 1.0 / (1.0 + args.guard_loss_protect * max(base_loss, 0.0))
    stats = {
        "guard_candidates": int(top_count),
        "guard_accepted": int(accepted),
        "guard_accept_rate": float(accepted / max(top_count, 1)),
        "guard_mean_gate": float(np.mean(gates)) if gates else 0.0,
        "guard_mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "guard_mean_uncertainty": float(np.mean(uncertainties)) if uncertainties else 0.0,
        "guard_base_loss": float(base_loss),
        "guard_distill_scale": float(protect_scale),
    }
    if not teacher_probs:
        return None, 0.0, stats, None
    weights = np.array(gated_weights, dtype=float)
    weights = weights / np.maximum(weights.sum(), 1e-12)
    teacher = torch.stack(
        [teacher_probs[i] * float(weights[i]) for i in range(len(teacher_probs))],
        dim=0,
    ).sum(dim=0)
    return teacher.clamp(min=1e-8), float(protect_scale), stats, validation_nodes


def guard_pp_candidate_weights(args, data, cid, train_nodes, gate_evidence_nodes, global_state, prev_local_states, proto_agg_weights):
    device = data.x.device
    validation_nodes = guard_validation_nodes(args, data, cid, gate_evidence_nodes)
    classes = int(data.num_classes)
    coverage_enabled = args.method != "spc_guard_pp_no_coverage"
    candidate_weights = proto_agg_weights[cid].copy()
    candidate_weights[cid] = 0.0
    empty_stats = {
        "guard_candidates": 0,
        "guard_accepted": 0,
        "guard_accept_rate": 0.0,
        "guard_mean_gate": 0.0,
        "guard_mean_delta": 0.0,
        "guard_mean_uncertainty": 0.0,
        "guard_base_loss": 0.0,
        "guard_distill_scale": 0.0,
        "guard_pp_class_gates": 0,
        "guard_pp_class_accept_rate": 0.0,
        "guard_pp_mean_class_gate": 0.0,
        "guard_pp_covered_classes": 0,
    }
    if candidate_weights.sum() <= 1e-12:
        return None, 0.0, empty_stats, None
    top_count = min(args.guard_candidate_count, len(candidate_weights) - 1)
    candidate_ids = np.argsort(-candidate_weights)[:top_count]
    if top_count <= 0:
        return None, 0.0, empty_stats, None

    labels = data.y[validation_nodes]
    counts = torch.bincount(labels, minlength=classes).float().to(device)
    present = counts > 0
    if coverage_enabled:
        reliability = float(present.float().mean().item())
        min_present = float(counts[present].min().item()) if present.any() else 0.0
        reliability = float(np.clip(0.5 * reliability + 0.5 * min(min_present / 3.0, 1.0), 0.05, 1.0))
    else:
        reliability = 1.0

    base = GCN(data.num_features, args.hidden_dim, data.num_classes, args.dropout).to(device)
    base.load_state_dict(global_state)
    base.eval()
    with torch.no_grad():
        base_logits = base(data.x, data.edge_index)
        base_loss = F.cross_entropy(base_logits[validation_nodes], labels).item()
        base_probs = base_logits[validation_nodes].softmax(dim=1)
        base_class_loss = []
        for cls in range(classes):
            mask = labels == cls
            if mask.any():
                base_class_loss.append(F.cross_entropy(base_logits[validation_nodes][mask], labels[mask]).item())
            else:
                base_class_loss.append(base_loss)

    teacher_sum = torch.zeros_like(base_probs)
    gate_sum = torch.zeros(classes, dtype=torch.float32, device=device)
    class_gates = []
    class_deltas = []
    class_uncertainties = []
    accepted_class_gates = 0

    for jid in candidate_ids:
        teacher = GCN(data.num_features, args.hidden_dim, data.num_classes, args.dropout).to(device)
        teacher.load_state_dict(prev_local_states[int(jid)])
        teacher.eval()
        with torch.no_grad():
            logits = teacher(data.x, data.edge_index)
            probs = logits[validation_nodes].softmax(dim=1)
            entropy_by_node = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1)
            for cls in range(classes):
                mask = labels == cls
                if mask.any():
                    teacher_loss = F.cross_entropy(logits[validation_nodes][mask], labels[mask]).item()
                    uncertainty = float(entropy_by_node[mask].mean().item())
                    coverage_complement = (
                        float(1.0 / np.sqrt(float(counts[cls].item()) + 1.0)) if coverage_enabled else 0.0
                    )
                else:
                    teacher_loss = F.cross_entropy(logits[validation_nodes], labels).item()
                    uncertainty = float(entropy_by_node.mean().item())
                    coverage_complement = 0.25 if coverage_enabled else 0.0
                delta = float(base_class_loss[cls] - teacher_loss)
                gate = 1.0 / (
                    1.0
                    + np.exp(
                        -(
                            args.guard_beta * delta
                            + args.guard_coverage_weight * coverage_complement
                            - args.guard_uncertainty_weight * uncertainty
                        )
                    )
                )
                threshold = args.guard_threshold + args.guard_loss_threshold_weight * min(base_loss, 5.0)
                class_gates.append(float(gate))
                class_deltas.append(delta)
                class_uncertainties.append(uncertainty)
                if gate >= threshold:
                    weight = float(candidate_weights[int(jid)] * gate)
                    teacher_sum[:, cls] = teacher_sum[:, cls] + probs[:, cls] * weight
                    gate_sum[cls] = gate_sum[cls] + weight
                    accepted_class_gates += 1

    accepted_mask = gate_sum > 1e-12
    if accepted_mask.any():
        teacher_probs = base_probs.detach().clone()
        teacher_probs[:, accepted_mask] = teacher_sum[:, accepted_mask] / gate_sum[accepted_mask].view(1, -1)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    else:
        teacher_probs = None

    protect_scale = reliability / (1.0 + args.guard_loss_protect * max(base_loss, 0.0))
    stats = {
        "guard_candidates": int(top_count),
        "guard_accepted": int(sum(gate_sum.detach().cpu().numpy() > 1e-12)),
        "guard_accept_rate": float((gate_sum > 1e-12).float().mean().item()),
        "guard_mean_gate": float(np.mean(class_gates)) if class_gates else 0.0,
        "guard_mean_delta": float(np.mean(class_deltas)) if class_deltas else 0.0,
        "guard_mean_uncertainty": float(np.mean(class_uncertainties)) if class_uncertainties else 0.0,
        "guard_base_loss": float(base_loss),
        "guard_distill_scale": float(protect_scale),
        "guard_pp_class_gates": int(top_count * classes),
        "guard_pp_class_accept_rate": float(accepted_class_gates / max(top_count * classes, 1)),
        "guard_pp_mean_class_gate": float(np.mean(class_gates)) if class_gates else 0.0,
        "guard_pp_covered_classes": int(present.sum().item()),
    }
    if teacher_probs is None:
        return None, 0.0, stats, None
    return teacher_probs.clamp(min=1e-8), float(protect_scale), stats, validation_nodes


def guard_validation_nodes(args, data, cid, train_nodes):
    """Subsample the local evidence used by transfer gates without changing local training."""
    ratio = float(getattr(args, "guard_validation_ratio", 1.0))
    drop_classes = int(getattr(args, "guard_validation_drop_classes", 0))
    if ratio >= 0.999 and drop_classes <= 0:
        return train_nodes
    device = train_nodes.device
    labels = data.y[train_nodes].detach().cpu()
    classes = int(data.num_classes)
    rng = np.random.default_rng(int(args.seed) * 1009 + int(cid) * 9173 + int(ratio * 1000) + drop_classes * 53)
    keep = np.ones(train_nodes.numel(), dtype=bool)
    if drop_classes > 0:
        present = sorted(set(int(x) for x in labels.numpy().tolist()))
        if present:
            shuffled = np.array(present, dtype=int)
            rng.shuffle(shuffled)
            for cls in shuffled[: min(drop_classes, max(len(shuffled) - 1, 0))]:
                keep[labels.numpy() == int(cls)] = False
    kept_indices = np.where(keep)[0]
    selected = []
    for cls in range(classes):
        cls_positions = kept_indices[labels.numpy()[kept_indices] == cls]
        if cls_positions.size == 0:
            continue
        rng.shuffle(cls_positions)
        take = max(1, int(np.ceil(cls_positions.size * max(min(ratio, 1.0), 0.0))))
        selected.extend(cls_positions[:take].tolist())
    if not selected:
        selected = kept_indices[:1].tolist() if kept_indices.size else [0]
    selected = np.array(sorted(set(selected)), dtype=np.int64)
    return train_nodes[torch.as_tensor(selected, dtype=torch.long, device=device)]


def train(args):
    train_start = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data = load_dataset(args.dataset, Path(args.data_root))
    data = data.to(device)
    clients = make_clients(data, args.clients, args.partition, args.seed)
    prepare_gate_evidence_splits(args, data, clients)
    flags = method_flags(args.method)
    fedpub_proxy = (
        fedpub_proxy_nodes(data, min(args.fedpub_proxy_nodes, data.num_nodes), args.seed)
        if flags["functional_personalization"]
        else None
    )
    descriptor_mode = "with_label" if args.method == "spc_label_descriptor" else args.descriptor_mode
    needs_prototype_bank = (
        flags["use_prototypes"]
        or flags["prototype_aggregation"]
        or flags["guarded_distillation"]
        or args.method in {"spc_random_prototypes", "spc_guard_random_candidates", "spc_label_descriptor"}
    )
    if needs_prototype_bank:
        desc, centers = prototype_bank(
            data,
            clients,
            args.prototype_count,
            args.dp_noise,
            args.seed,
            descriptor_mode=descriptor_mode,
        )
    else:
        desc = np.zeros((args.clients, 7), dtype=float)
        centers = np.zeros((max(args.prototype_count, 1), 7), dtype=float)
    descriptor_dim = int(desc.shape[1]) if len(desc.shape) > 1 else 0
    if args.method == "spc_random_prototypes":
        rng = np.random.default_rng(args.seed + 7919)
        desc_for_aggregation = rng.normal(size=desc.shape)
        desc_for_aggregation = desc_for_aggregation / np.maximum(
            np.linalg.norm(desc_for_aggregation, axis=1, keepdims=True),
            1e-8,
        )
    elif args.method == "spc_guard_random_candidates":
        rng = np.random.default_rng(args.seed + 104729)
        desc_for_aggregation = rng.normal(size=desc.shape)
        desc_for_aggregation = desc_for_aggregation / np.maximum(
            np.linalg.norm(desc_for_aggregation, axis=1, keepdims=True),
            1e-8,
        )
    elif args.method in {"spc_no_prototypes", "spc_no_protoagg"}:
        desc_for_aggregation = np.ones_like(desc)
    else:
        desc_for_aggregation = desc
    proto_agg_weights = prototype_similarity_weights(desc_for_aggregation, temperature=args.prototype_agg_temperature)
    if needs_prototype_bank:
        misalignment_by_client = candidate_label_misalignment(
            data,
            clients,
            proto_agg_weights,
            args.guard_candidate_count,
        )
    else:
        misalignment_by_client = empty_candidate_misalignment(clients)
    for c in clients:
        cid = c["client_id"]
        c["static_edge_weight"] = client_edge_weight(
            data,
            c,
            desc[cid],
            centers,
            args.method,
            args.mask_keep_ratio,
        )
        if flags["edge_evaluator"]:
            prepare_client_edges(data, c, desc[cid], centers, args.method, args.mask_keep_ratio)
        else:
            prepare_client_base_edges(data, c)
    model = GCN(data.num_features, args.hidden_dim, data.num_classes, args.dropout).to(device)
    residual_model = GCN(data.num_features, args.hidden_dim, data.num_classes, args.dropout).to(device) if flags["residual_gating"] else None
    if flags["personalized"]:
        local_heads = [
            torch.nn.Linear(data.num_classes, data.num_classes, bias=False).to(device) for _ in range(args.clients)
        ]
        opt_heads = [torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay) for head in local_heads]
    else:
        local_heads = [None for _ in range(args.clients)]
        opt_heads = [None for _ in range(args.clients)]
    edge_evaluators = [None for _ in range(args.clients)]
    biased_models = [None for _ in range(args.clients)]
    opt_private = [None for _ in range(args.clients)]
    if flags["edge_evaluator"]:
        spc_edge_dim = clients[0]["edge_features_spc"].shape[1]
        fedath_edge_dim = clients[0]["edge_features_fedath"].shape[1]
        for idx in range(args.clients):
            edge_dim = fedath_edge_dim if args.method == "fedath" else spc_edge_dim
            evaluator = EdgeEvaluator(edge_dim, hidden_dim=max(16, args.hidden_dim // 2)).to(device)
            biased = GCN(data.num_features, args.hidden_dim, data.num_classes, args.dropout).to(device)
            edge_evaluators[idx] = evaluator
            biased_models[idx] = biased
            opt_private[idx] = torch.optim.Adam(
                list(evaluator.parameters()) + list(biased.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

    model_parameter_count = sum(p.numel() for p in model.parameters())
    head_parameter_count = (
        sum(p.numel() for head in local_heads if head is not None for p in head.parameters())
        if flags["personalized"]
        else 0
    )
    model_upload_floats = args.rounds * args.clients * model_parameter_count
    model_download_floats = args.rounds * args.clients * model_parameter_count
    descriptor_upload_floats = args.clients * descriptor_dim if needs_prototype_bank else 0
    fedpub_embedding_floats = (
        args.rounds * args.clients * int(data.num_classes) if flags["functional_personalization"] else 0
    )
    candidate_download_floats = (
        args.clients * min(args.guard_candidate_count, max(args.clients - 1, 0))
        if flags["prototype_aggregation"] or flags["guarded_distillation"]
        else 0
    )
    estimated_model_upload_bytes = int(4 * model_upload_floats)
    estimated_model_download_bytes = int(4 * model_download_floats)
    estimated_descriptor_upload_bytes = int(4 * (descriptor_upload_floats + fedpub_embedding_floats))
    estimated_candidate_download_bytes = int(4 * candidate_download_floats)
    estimated_upload_bytes = estimated_model_upload_bytes + estimated_descriptor_upload_bytes
    estimated_download_bytes = estimated_model_download_bytes + estimated_candidate_download_bytes

    global_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    global_residual_state = (
        {k: v.detach().clone() for k, v in residual_model.state_dict().items()} if flags["residual_gating"] else {}
    )
    active_proto_weights = proto_agg_weights.copy()
    active_fedpub_weights = np.ones((args.clients, args.clients), dtype=float) / max(args.clients, 1)
    guard_stats_by_client = {
        cid: {
            "guard_candidates": 0,
            "guard_accepted": 0,
            "guard_accept_rate": 0.0,
            "guard_mean_gate": 0.0,
            "guard_mean_delta": 0.0,
            "guard_mean_uncertainty": 0.0,
            "guard_base_loss": 0.0,
            "guard_distill_scale": 0.0,
            "guard_pp_class_gates": 0,
            "guard_pp_class_accept_rate": 0.0,
            "guard_pp_mean_class_gate": 0.0,
            "guard_pp_covered_classes": 0,
        }
        for cid in range(args.clients)
    }
    for _round in range(args.rounds):
        local_states = []
        local_residual_states = []
        client_losses = []
        for c in clients:
            local = GCN(data.num_features, args.hidden_dim, data.num_classes, args.dropout).to(device)
            if flags["functional_personalization"] and _round > 0:
                cid = c["client_id"]
                personalized_state = {}
                for key in global_state:
                    personalized_state[key] = torch.stack(
                        [
                            client_states[key] * float(active_fedpub_weights[cid, j])
                            for j, client_states in enumerate(prev_local_states)
                        ],
                        dim=0,
                    ).sum(dim=0)
                local.load_state_dict(personalized_state)
            elif flags["prototype_aggregation"] and _round > 0:
                cid = c["client_id"]
                personalized_state = {}
                for key in global_state:
                    personalized_state[key] = torch.stack(
                        [
                            client_states[key] * float(active_proto_weights[cid, j])
                            for j, client_states in enumerate(prev_local_states)
                        ],
                        dim=0,
                    ).sum(dim=0)
                local.load_state_dict(personalized_state)
            else:
                local.load_state_dict(global_state)
            local_residual = None
            if flags["residual_gating"]:
                local_residual = GCN(data.num_features, args.hidden_dim, data.num_classes, args.dropout).to(device)
                if _round > 0:
                    cid = c["client_id"]
                    personalized_residual_state = {}
                    for key in global_residual_state:
                        personalized_residual_state[key] = torch.stack(
                            [
                                client_states[key] * float(active_proto_weights[cid, j])
                                for j, client_states in enumerate(prev_local_residual_states)
                            ],
                            dim=0,
                        ).sum(dim=0)
                    local_residual.load_state_dict(personalized_residual_state)
                else:
                    local_residual.load_state_dict(global_residual_state)
            opt_params = list(local.parameters())
            if flags["residual_gating"]:
                opt_params += list(local_residual.parameters())
            opt = torch.optim.Adam(opt_params, lr=args.lr, weight_decay=args.weight_decay)
            train_nodes = c.get("supervised_nodes", c["train_nodes"]).to(device)
            gate_evidence_nodes = c.get("gate_evidence_nodes", c["train_nodes"]).to(device)
            cid = c["client_id"]
            guard_teacher = None
            guard_nodes = None
            guard_scale = 0.0
            if flags["guarded_distillation"] and flags["guard_distill"] and _round > 0:
                if flags["class_conditional_guard"]:
                    guard_teacher, guard_scale, guard_stats, guard_nodes = guard_pp_candidate_weights(
                        args,
                        data,
                        cid,
                        train_nodes,
                        gate_evidence_nodes,
                        global_state,
                        prev_local_states,
                        proto_agg_weights,
                    )
                else:
                    guard_teacher, guard_scale, guard_stats, guard_nodes = guard_candidate_weights(
                        args,
                        flags,
                        data,
                        cid,
                        train_nodes,
                        gate_evidence_nodes,
                        global_state,
                        prev_local_states,
                        proto_agg_weights,
                    )
                guard_stats_by_client[cid] = guard_stats
            for _ in range(args.local_epochs):
                local.train()
                if flags["edge_evaluator"]:
                    biased_models[cid].train()
                    edge_evaluators[cid].train()
                opt.zero_grad()
                if flags["personalized"]:
                    opt_heads[cid].zero_grad()
                if flags["edge_evaluator"]:
                    opt_private[cid].zero_grad()
                if flags["edge_evaluator"]:
                    use_spc = args.method != "fedath"
                    edge_weight, biased_edge_weight = evaluator_edge_weight(c, edge_evaluators[cid], use_spc)
                else:
                    edge_weight = c["static_edge_weight"].to(device)
                    biased_edge_weight = (c["base_edge_weight"].to(device) - edge_weight).clamp(min=0.0, max=1.0)
                if flags["residual_gating"]:
                    full_logits, full_hidden = local(
                        data.x,
                        data.edge_index,
                        edge_weight=c["base_edge_weight"].to(device),
                        return_hidden=True,
                    )
                    residual_logits, hidden = local_residual(
                        data.x,
                        data.edge_index,
                        edge_weight=edge_weight,
                        return_hidden=True,
                    )
                    logits = full_logits + args.residual_weight * residual_logits
                else:
                    logits, hidden = local(data.x, data.edge_index, edge_weight=edge_weight, return_hidden=True)
                if flags["personalized"]:
                    logits = logits + args.personalization_weight * local_heads[cid](logits)
                if flags["class_balanced_loss"]:
                    loss = class_balanced_cross_entropy(
                        logits[train_nodes],
                        data.y[train_nodes],
                        data.num_classes,
                        strength=args.class_balance_strength,
                    )
                else:
                    loss = F.cross_entropy(logits[train_nodes], data.y[train_nodes])
                if guard_teacher is not None:
                    distill_nodes = guard_nodes if guard_nodes is not None else train_nodes
                    log_probs = logits[distill_nodes].log_softmax(dim=1)
                    loss = loss + args.guard_distill_weight * max(guard_scale, 1e-6) * F.kl_div(
                        log_probs,
                        guard_teacher,
                        reduction="batchmean",
                    )
                if flags["edge_evaluator"]:
                    biased_logits, biased_hidden = biased_models[cid](
                        data.x,
                        data.edge_index,
                        edge_weight=biased_edge_weight,
                        return_hidden=True,
                    )
                    biased_prob = biased_logits[train_nodes].softmax(dim=1).clamp(min=1e-8)
                    biased_entropy = -(biased_prob * biased_prob.log()).sum(dim=1).mean()
                    loss = loss - args.biased_entropy_weight * biased_entropy
                    if flags["hsic"]:
                        loss = loss + args.hsic_weight * hsic_loss(hidden[train_nodes], biased_hidden[train_nodes])
                    keep_rate = edge_weight[c["local_edge_idx"]].mean() if c["local_edge_idx"].numel() else edge_weight.mean()
                    loss = loss + args.edge_sparsity_weight * (keep_rate - args.mask_keep_ratio).pow(2)
                    if c["local_edge_idx"].numel():
                        edge_scores = torch.nan_to_num(
                            edge_weight[c["local_edge_idx"]],
                            nan=0.5,
                            posinf=1.0,
                            neginf=0.0,
                        ).clamp(min=1e-6, max=1.0 - 1e-6)
                        supervised_mask = c["edge_supervised_mask"]
                        if supervised_mask.any():
                            target = c["edge_supervised_target"][supervised_mask]
                            loss = loss + args.edge_supervision_weight * F.binary_cross_entropy(
                                edge_scores[supervised_mask],
                                target,
                            )
                        if flags["use_prototypes"]:
                            loss = loss + args.prototype_prior_weight * F.binary_cross_entropy(
                                edge_scores,
                                c["edge_proto_prior"].clamp(min=1e-6, max=1.0 - 1e-6),
                            )
                if flags["prox"]:
                    prox = 0.0
                    for name, param in local.named_parameters():
                        prox = prox + (param - global_state[name]).pow(2).mean()
                    loss = loss + args.prox_mu * prox
                if flags["alignment"]:
                    if flags["edge_evaluator"]:
                        keep_rate = edge_weight[c["local_edge_idx"]].mean() if c["local_edge_idx"].numel() else edge_weight.mean()
                        target = edge_weight.new_tensor(c["alignment_target"])
                        loss = loss + args.alignment_weight * (keep_rate - target).pow(2)
                    else:
                        dist = ((centers - desc[cid][None, :]) ** 2).sum(axis=1).min()
                        loss = loss + args.alignment_weight * float(dist)
                if flags["counterfactual"]:
                    if flags["edge_evaluator"]:
                        cf_edge = (0.75 * edge_weight + 0.25 * c["base_edge_weight"].to(device)).clamp(max=1.0)
                        if flags["residual_gating"]:
                            cf_logits = full_logits + args.residual_weight * local_residual(
                                data.x,
                                data.edge_index,
                                edge_weight=cf_edge,
                            )
                        else:
                            cf_logits = local(data.x, data.edge_index, edge_weight=cf_edge)
                        p = logits[train_nodes].log_softmax(dim=1)
                        q = cf_logits[train_nodes].softmax(dim=1)
                        loss = loss + args.counterfactual_weight * F.kl_div(p, q, reduction="batchmean")
                    else:
                        loss = loss + args.counterfactual_weight * logits[train_nodes].softmax(dim=1).std(dim=0).mean()
                loss.backward()
                opt.step()
                if flags["edge_evaluator"]:
                    opt_private[cid].step()
                if flags["personalized"]:
                    opt_heads[cid].step()
            local_states.append({k: v.detach().clone() for k, v in local.state_dict().items()})
            if flags["residual_gating"]:
                local_residual_states.append({k: v.detach().clone() for k, v in local_residual.state_dict().items()})
            with torch.no_grad():
                client_losses.append(float(F.cross_entropy(logits[train_nodes], data.y[train_nodes]).item()))
        prev_local_states = local_states
        prev_local_residual_states = local_residual_states
        if flags["functional_personalization"]:
            fedpub_embeddings = fedpub_functional_embeddings(data, fedpub_proxy, prev_local_states, args)
            active_fedpub_weights = fedpub_similarity_weights(fedpub_embeddings, args.fedpub_temperature)
        if flags["worst_reweight"] and args.worst_client_weight > 0:
            loss_arr = np.array(client_losses, dtype=float)
            loss_arr = loss_arr - loss_arr.min()
            agg_weights = np.exp(args.worst_client_weight * loss_arr)
            agg_weights = agg_weights / np.maximum(agg_weights.sum(), 1e-12)
            if flags["prototype_aggregation"]:
                active_proto_weights = mix_with_difficulty(proto_agg_weights, agg_weights)
        else:
            agg_weights = np.ones(len(local_states), dtype=float) / len(local_states)
            active_proto_weights = proto_agg_weights.copy()
        global_state = {}
        for key in local_states[0]:
            global_state[key] = torch.stack(
                [state[key] * float(agg_weights[i]) for i, state in enumerate(local_states)],
                dim=0,
            ).sum(dim=0)
        if flags["residual_gating"]:
            global_residual_state = {}
            for key in local_residual_states[0]:
                global_residual_state[key] = torch.stack(
                    [state[key] * float(agg_weights[i]) for i, state in enumerate(local_residual_states)],
                    dim=0,
                ).sum(dim=0)
        model.load_state_dict(global_state)
        if flags["residual_gating"]:
            residual_model.load_state_dict(global_residual_state)

    train_seconds = time.time() - train_start
    model.eval()
    rows = []
    with torch.no_grad():
        for c in clients:
            cid = c["client_id"]
            eval_model = model
            if flags["functional_personalization"] and args.rounds > 0:
                personalized_state = {}
                for key in global_state:
                    personalized_state[key] = torch.stack(
                        [
                            client_states[key] * float(active_fedpub_weights[cid, j])
                            for j, client_states in enumerate(prev_local_states)
                        ],
                        dim=0,
                    ).sum(dim=0)
                eval_model = GCN(data.num_features, args.hidden_dim, data.num_classes, args.dropout).to(device)
                eval_model.load_state_dict(personalized_state)
                eval_model.eval()
            elif flags["prototype_aggregation"] and args.rounds > 0:
                personalized_state = {}
                for key in global_state:
                    personalized_state[key] = torch.stack(
                        [
                            client_states[key] * float(active_proto_weights[cid, j])
                            for j, client_states in enumerate(prev_local_states)
                        ],
                        dim=0,
                    ).sum(dim=0)
                eval_model = GCN(data.num_features, args.hidden_dim, data.num_classes, args.dropout).to(device)
                eval_model.load_state_dict(personalized_state)
                eval_model.eval()
            if flags["edge_evaluator"]:
                use_spc = args.method != "fedath"
                edge_weight, _ = evaluator_edge_weight(c, edge_evaluators[cid], use_spc)
            else:
                edge_weight = c["static_edge_weight"].to(device)
            eval_edge_weight = c["base_edge_weight"].to(device) if flags["residual_gating"] else edge_weight
            base_logits = eval_model(data.x, data.edge_index, edge_weight=eval_edge_weight)
            if flags["residual_gating"]:
                base_logits = base_logits + args.residual_weight * residual_model(
                    data.x,
                    data.edge_index,
                    edge_weight=edge_weight,
                )
            logits = base_logits
            if flags["personalized"]:
                logits = logits + args.personalization_weight * local_heads[cid](base_logits)
            nodes = c["test_nodes"].to(device)
            pred = logits[nodes].argmax(dim=1)
            y = data.y[nodes]
            rows.append(
                {
                    **{
                        f"train_{key}": value
                        for key, value in label_coverage_stats(data, c["train_nodes"].to(device)).items()
                    },
                    **{
                        f"test_{key}": value
                        for key, value in label_coverage_stats(data, c["test_nodes"].to(device)).items()
                    },
                    "dataset": args.dataset,
                    "partition": args.partition,
                    "clients": args.clients,
                    "seed": args.seed,
                    "method": args.method,
                    "run_tag": args.run_tag,
                    "client_id": cid,
                    "accuracy": float((pred == y).float().mean().item()),
                    "macro_f1": macro_f1_torch(y, pred, data.num_classes),
                    "train_seconds": float(train_seconds),
                    "model_parameter_count": int(model_parameter_count),
                    "personal_head_parameter_count": int(head_parameter_count),
                    "descriptor_dim": int(descriptor_dim),
                    "estimated_upload_bytes": int(estimated_upload_bytes),
                    "estimated_download_bytes": int(estimated_download_bytes),
                    "estimated_model_upload_bytes": int(estimated_model_upload_bytes),
                    "estimated_model_download_bytes": int(estimated_model_download_bytes),
                    "estimated_descriptor_upload_bytes": int(estimated_descriptor_upload_bytes),
                    "estimated_candidate_download_bytes": int(estimated_candidate_download_bytes),
                    "fedpub_proxy_nodes": int(fedpub_proxy.numel()) if fedpub_proxy is not None else 0,
                    "guard_validation_ratio": float(args.guard_validation_ratio),
                    "guard_validation_drop_classes": int(args.guard_validation_drop_classes),
                    "guard_evidence_mode": str(args.guard_evidence_mode),
                    "guard_independent_ratio": float(args.guard_independent_ratio),
                    "supervised_train_nodes": int(c.get("supervised_nodes", c["train_nodes"]).numel()),
                    "gate_evidence_nodes": int(c.get("gate_evidence_nodes", c["train_nodes"]).numel()),
                    **misalignment_by_client.get(cid, {}),
                    **guard_stats_by_client.get(cid, {}),
                }
            )
    return rows


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    accs = np.array([r["accuracy"] for r in rows], dtype=float)
    f1s = np.array([r["macro_f1"] for r in rows], dtype=float)
    agg = {
        "method": rows[0]["method"],
        "dataset": rows[0]["dataset"],
        "partition": rows[0]["partition"],
        "clients": rows[0]["clients"],
        "seed": rows[0]["seed"],
        "run_tag": rows[0].get("run_tag", "base"),
        "mean_accuracy": float(accs.mean()),
        "worst_accuracy": float(accs.min()),
        "mean_macro_f1": float(f1s.mean()),
        "worst_macro_f1": float(f1s.min()),
        "train_seconds": rows[0].get("train_seconds", ""),
        "model_parameter_count": rows[0].get("model_parameter_count", ""),
        "personal_head_parameter_count": rows[0].get("personal_head_parameter_count", ""),
        "descriptor_dim": rows[0].get("descriptor_dim", ""),
        "estimated_upload_bytes": rows[0].get("estimated_upload_bytes", ""),
        "estimated_download_bytes": rows[0].get("estimated_download_bytes", ""),
        "estimated_model_upload_bytes": rows[0].get("estimated_model_upload_bytes", ""),
        "estimated_model_download_bytes": rows[0].get("estimated_model_download_bytes", ""),
        "estimated_descriptor_upload_bytes": rows[0].get("estimated_descriptor_upload_bytes", ""),
        "estimated_candidate_download_bytes": rows[0].get("estimated_candidate_download_bytes", ""),
        "fedpub_proxy_nodes": rows[0].get("fedpub_proxy_nodes", ""),
        "guard_validation_ratio": rows[0].get("guard_validation_ratio", ""),
        "guard_validation_drop_classes": rows[0].get("guard_validation_drop_classes", ""),
        "guard_evidence_mode": rows[0].get("guard_evidence_mode", ""),
        "guard_independent_ratio": rows[0].get("guard_independent_ratio", ""),
        "supervised_train_nodes": rows[0].get("supervised_train_nodes", ""),
        "gate_evidence_nodes": rows[0].get("gate_evidence_nodes", ""),
    }
    agg_path = path.with_name(path.stem + "_aggregated.csv")
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg.keys()))
        writer.writeheader()
        writer.writerow(agg)
    print(f"Wrote {path}")
    print(f"Wrote {agg_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-tag", default="base")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--prototype-count", type=int, default=3)
    parser.add_argument("--descriptor-mode", choices=["structure", "with_label"], default="structure")
    parser.add_argument("--dp-noise", type=float, default=0.0)
    parser.add_argument("--alignment-weight", type=float, default=0.02)
    parser.add_argument("--counterfactual-weight", type=float, default=0.01)
    parser.add_argument("--personalization-weight", type=float, default=0.25)
    parser.add_argument("--prox-mu", type=float, default=0.001)
    parser.add_argument("--mask-keep-ratio", type=float, default=0.55)
    parser.add_argument("--hsic-weight", type=float, default=0.02)
    parser.add_argument("--biased-entropy-weight", type=float, default=0.02)
    parser.add_argument("--edge-sparsity-weight", type=float, default=0.02)
    parser.add_argument("--edge-supervision-weight", type=float, default=0.05)
    parser.add_argument("--prototype-prior-weight", type=float, default=0.05)
    parser.add_argument("--residual-weight", type=float, default=0.35)
    parser.add_argument("--prototype-agg-temperature", type=float, default=8.0)
    parser.add_argument("--worst-client-weight", type=float, default=0.6)
    parser.add_argument("--guard-candidate-count", type=int, default=3)
    parser.add_argument("--guard-threshold", type=float, default=0.35)
    parser.add_argument("--guard-beta", type=float, default=4.0)
    parser.add_argument("--guard-uncertainty-weight", type=float, default=0.05)
    parser.add_argument("--guard-coverage-weight", type=float, default=0.35)
    parser.add_argument("--guard-distill-weight", type=float, default=0.15)
    parser.add_argument("--guard-loss-protect", type=float, default=0.35)
    parser.add_argument("--guard-loss-threshold-weight", type=float, default=0.05)
    parser.add_argument("--class-balance-strength", type=float, default=0.5)
    parser.add_argument("--fedpub-proxy-nodes", type=int, default=512)
    parser.add_argument("--fedpub-temperature", type=float, default=8.0)
    parser.add_argument("--guard-validation-ratio", type=float, default=1.0)
    parser.add_argument("--guard-validation-drop-classes", type=int, default=0)
    parser.add_argument("--guard-evidence-mode", choices=["train", "independent"], default="train")
    parser.add_argument("--guard-independent-ratio", type=float, default=0.2)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rows = train(args)
    write_rows(args.output, rows)


if __name__ == "__main__":
    main()
