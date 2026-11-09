import argparse
import yaml
import torch
from src.data import StructureDataset
from src.model import FoldCast
from src.flow import integrate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    model = FoldCast(cfg["hidden_dim"], cfg["num_layers"])
    model.load_state_dict(torch.load(args.checkpoint))
    dataset = StructureDataset("data/structures")
    rmsds = []
    for sample in dataset:
        pred = integrate(model, sample["coords"], sample["features"], None, None, cfg["flow_steps"])
        rmsds.append((pred - sample["coords"]).pow(2).mean().sqrt().item())
    print(f"mean RMSD: {sum(rmsds) / len(rmsds):.3f}")


if __name__ == "__main__":
    main()
