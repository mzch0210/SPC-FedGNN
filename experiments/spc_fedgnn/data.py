from dataclasses import dataclass

import numpy as np


@dataclass
class ClientGraph:
    client_id: int
    adjacency: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    train_mask: np.ndarray
    test_mask: np.ndarray
    homophily_target: float


class SyntheticFederatedGraphDataset:
    """Federated graph generator for early SPC-FedGNN experiments.

    The generator intentionally creates clients with different homophily,
    density, feature shift, and label skew so the spectral prototype path has
    a measurable role before real benchmark loaders are connected.
    """

    def __init__(
        self,
        clients=10,
        nodes_per_client=160,
        feature_dim=16,
        classes=3,
        seed=0,
    ):
        self.clients = clients
        self.nodes_per_client = nodes_per_client
        self.feature_dim = feature_dim
        self.classes = classes
        self.rng = np.random.default_rng(seed)

    def build(self):
        centroids = self.rng.normal(0, 1.0, size=(self.classes, self.feature_dim))
        graphs = []
        for client_id in range(self.clients):
            homophily = 0.25 + 0.65 * (client_id / max(self.clients - 1, 1))
            density = 0.035 + 0.018 * ((client_id % 3) / 2)
            skew = self._client_label_prior(client_id)
            labels = self.rng.choice(self.classes, self.nodes_per_client, p=skew)
            shift = self.rng.normal(0, 0.35, size=(1, self.feature_dim))
            features = centroids[labels] + shift + self.rng.normal(
                0, 0.75, size=(self.nodes_per_client, self.feature_dim)
            )
            adjacency = self._make_graph(labels, density, homophily)
            idx = self.rng.permutation(self.nodes_per_client)
            train_mask = np.zeros(self.nodes_per_client, dtype=bool)
            test_mask = np.zeros(self.nodes_per_client, dtype=bool)
            train_mask[idx[: int(0.55 * self.nodes_per_client)]] = True
            test_mask[idx[int(0.7 * self.nodes_per_client) :]] = True
            graphs.append(
                ClientGraph(
                    client_id=client_id,
                    adjacency=adjacency,
                    features=features,
                    labels=labels,
                    train_mask=train_mask,
                    test_mask=test_mask,
                    homophily_target=homophily,
                )
            )
        return graphs

    def _client_label_prior(self, client_id):
        concentration = np.ones(self.classes) * 0.7
        concentration[client_id % self.classes] = 2.7
        return self.rng.dirichlet(concentration)

    def _make_graph(self, labels, density, homophily):
        n = labels.shape[0]
        same = labels[:, None] == labels[None, :]
        base = np.full((n, n), density * (1.0 - homophily))
        base[same] = density * (1.0 + homophily)
        samples = self.rng.random((n, n)) < base
        upper = np.triu(samples, 1)
        adjacency = upper + upper.T
        return adjacency.astype(float)
