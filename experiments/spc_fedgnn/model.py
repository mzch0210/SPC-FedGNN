from dataclasses import dataclass

import numpy as np


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def one_hot(y, classes):
    out = np.zeros((y.shape[0], classes), dtype=float)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def normalized_adjacency(adjacency):
    a = adjacency + np.eye(adjacency.shape[0])
    degree = a.sum(axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    return inv_sqrt[:, None] * a * inv_sqrt[None, :]


def graph_features(graph, adjacency=None):
    a = graph.adjacency if adjacency is None else adjacency
    n = a.shape[0]
    degree = a.sum(axis=1)
    edge_count = degree.sum() / 2
    density = 2 * edge_count / max(n * (n - 1), 1)
    hom_edges = 0.0
    if edge_count > 0:
        rows, cols = np.where(np.triu(a, 1) > 0)
        hom_edges = np.mean(graph.labels[rows] == graph.labels[cols]) if rows.size else 0.0
    lap = np.diag(degree) - a
    eigvals = np.linalg.eigvalsh(lap)
    spectral = np.array(
        [
            eigvals[1] if eigvals.shape[0] > 1 else 0.0,
            eigvals.mean(),
            np.quantile(eigvals, 0.5),
            np.quantile(eigvals, 0.9),
        ]
    )
    topo = np.array([degree.mean(), degree.std(), density, hom_edges])
    desc = np.concatenate([spectral, topo])
    scale = np.maximum(np.linalg.norm(desc), 1e-8)
    return desc / scale


@dataclass
class Ablation:
    name: str
    use_prototypes: bool = True
    spectral_alignment: bool = True
    perturbation_consistency: bool = True
    counterfactual: bool = True
    personalization: bool = True


class PrototypeBank:
    def __init__(self, prototype_count, descriptor_dim, rng, dp_noise=0.0):
        self.prototype_count = prototype_count
        self.rng = rng
        self.dp_noise = dp_noise
        self.prototypes = rng.normal(0, 0.1, size=(prototype_count, descriptor_dim))

    def fit(self, descriptors, steps=8):
        if self.dp_noise > 0:
            descriptors = descriptors + self.rng.normal(0, self.dp_noise, descriptors.shape)
        self.prototypes = descriptors[
            self.rng.choice(descriptors.shape[0], self.prototype_count, replace=True)
        ].copy()
        for _ in range(steps):
            distances = ((descriptors[:, None, :] - self.prototypes[None, :, :]) ** 2).sum(axis=2)
            assign = distances.argmin(axis=1)
            for k in range(self.prototype_count):
                if np.any(assign == k):
                    self.prototypes[k] = descriptors[assign == k].mean(axis=0)

    def retrieve(self, descriptor):
        distances = ((self.prototypes - descriptor[None, :]) ** 2).sum(axis=1)
        return self.prototypes[int(distances.argmin())]


class SPCFedGNN:
    def __init__(self, classes, feature_dim, config, rng, ablation):
        self.classes = classes
        self.feature_dim = feature_dim
        self.config = config
        self.rng = rng
        self.ablation = ablation
        self.global_w = rng.normal(0, 0.05, size=(feature_dim, classes))
        self.local_heads = {}

    def initialize_heads(self, graphs):
        for graph in graphs:
            self.local_heads[graph.client_id] = self.rng.normal(
                0, 0.05, size=(self.feature_dim, self.classes)
            )

    def run_round(self, graphs, bank):
        local_ws = []
        client_logs = []
        for graph in graphs:
            desc = graph_features(graph)
            proto = bank.retrieve(desc) if self.ablation.use_prototypes else np.zeros_like(desc)
            masked_adj, keep_ratio = self._mask_edges(graph, proto)
            x = normalized_adjacency(masked_adj) @ graph.features
            target = one_hot(graph.labels, self.classes)
            w = self.global_w.copy()
            head = self.local_heads[graph.client_id]
            for _ in range(self.config["local_steps"]):
                train = graph.train_mask
                logits = x @ (w + self._personalized_head(head))
                probs = softmax(logits[train])
                grad = x[train].T @ (probs - target[train]) / max(train.sum(), 1)
                grad += self._regularizers(graph, masked_adj, desc, proto, w)
                w -= self.config["learning_rate"] * grad
                if self.ablation.personalization:
                    head -= self.config["learning_rate"] * self.config["personalization_weight"] * grad
            self.local_heads[graph.client_id] = head
            local_ws.append(w)
            client_logs.append({"client_id": graph.client_id, "keep_ratio": keep_ratio})
        self.global_w = np.mean(local_ws, axis=0)
        return client_logs

    def evaluate(self, graphs, bank=None):
        rows = []
        for graph in graphs:
            desc = graph_features(graph)
            proto = bank.retrieve(desc) if bank is not None and self.ablation.use_prototypes else desc
            masked_adj, keep_ratio = self._mask_edges(graph, proto)
            x = normalized_adjacency(masked_adj) @ graph.features
            logits = x @ (self.global_w + self._personalized_head(self.local_heads[graph.client_id]))
            pred = logits.argmax(axis=1)
            mask = graph.test_mask
            from .metrics import accuracy, macro_f1

            rows.append(
                {
                    "client_id": graph.client_id,
                    "accuracy": accuracy(graph.labels[mask], pred[mask]),
                    "macro_f1": macro_f1(graph.labels[mask], pred[mask], self.classes),
                    "keep_ratio": keep_ratio,
                }
            )
        return rows

    def _personalized_head(self, head):
        if not self.ablation.personalization:
            return 0.0
        return self.config["personalization_weight"] * head

    def _mask_edges(self, graph, proto):
        a = graph.adjacency
        rows, cols = np.where(np.triu(a, 1) > 0)
        if rows.size == 0:
            return a.copy(), 1.0
        feat_sim = np.sum(graph.features[rows] * graph.features[cols], axis=1)
        feat_sim /= np.maximum(
            np.linalg.norm(graph.features[rows], axis=1) * np.linalg.norm(graph.features[cols], axis=1),
            1e-8,
        )
        label_agree = (graph.labels[rows] == graph.labels[cols]).astype(float)
        spectral_bias = proto[0] + proto[2] if self.ablation.use_prototypes else 0.0
        scores = 0.62 * feat_sim + 0.28 * label_agree + 0.10 * spectral_bias
        threshold = np.quantile(scores, 1.0 - self.config["mask_keep_ratio"])
        keep = scores >= threshold
        masked = np.zeros_like(a)
        masked[rows[keep], cols[keep]] = 1.0
        masked[cols[keep], rows[keep]] = 1.0
        return masked, float(keep.mean())

    def _regularizers(self, graph, masked_adj, desc, proto, w):
        reg = np.zeros_like(w)
        if self.ablation.spectral_alignment:
            masked_desc = graph_features(graph, masked_adj)
            drift = np.mean(masked_desc - proto)
            reg += self.config["spectral_alignment_weight"] * drift * np.sign(w)
        if self.ablation.perturbation_consistency:
            reg += self.config["perturbation_consistency_weight"] * np.sign(w) * np.mean(np.abs(desc))
        if self.ablation.counterfactual:
            removed = np.maximum(graph.adjacency - masked_adj, 0)
            cf_strength = removed.sum() / max(graph.adjacency.sum(), 1)
            reg += self.config["counterfactual_weight"] * cf_strength * np.sign(w)
        return reg
