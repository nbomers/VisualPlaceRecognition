import torch.nn as nn
import torch.nn.functional as F


class LinearAdapter(nn.Module):
    """
    Linearer Adapter mit Identitaets-Initialisierung.

    Bei zufaelliger Initialisierung zerstoert der Layer den vortrainierten
    Embedding-Raum sofort. Als Identitaet gestartet, entspricht der Adapter
    vor dem ersten Schritt exakt der Baseline -- die Differenz danach ist
    damit sauber dem Training zuzuschreiben.
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return F.normalize(self.linear(x), dim=-1)
