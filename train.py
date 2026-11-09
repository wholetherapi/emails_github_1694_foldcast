import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.data import StructureDataset, collate
from src.model import FoldCast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dataset = StructureDataset("data/structures")
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], collate_fn=collate)
    model = FoldCast(cfg["hidden_dim"], cfg["num_layers"])
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    for epoch in range(cfg["epochs"]):
        for batch in loader:
            optimiser.zero_grad()
            velocity = model(batch["coords"], batch["features"], None, None)
            loss = velocity.pow(2).mean()
            loss.backward()
            optimiser.step()
        print(f"epoch {epoch} loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
