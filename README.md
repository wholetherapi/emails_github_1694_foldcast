# FoldCast

Official implementation of the paper **FoldCast: Uncertainty-Aware Protein Structure Prediction with Equivariant Flow Matching** (Nature Methods 2025).

FoldCast couples an equivariant message-passing trunk with a conditional
flow-matching head. The flow head transports a simple prior distribution to
an ensemble of backbone conformations, and a per-residue variance head turns
the ensemble into calibrated uncertainty estimates.

## Layout

- `src/model.py` — equivariant encoder and flow-matching head
- `src/data.py` — PDB-style structure dataset and featurisation
- `src/flow.py` — probability-flow ODE integrator
- `train.py` — training entry point
- `eval.py` — RMSD / uncertainty evaluation
- `configs/base.yaml` — reference hyper-parameters

## Quick start

```bash
pip install -r requirements.txt
python train.py --config configs/base.yaml
python eval.py --config configs/base.yaml --checkpoint checkpoints/foldcast.pt
```

## Citation

```
@article{rao2025foldcast,
  title={FoldCast: Uncertainty-Aware Protein Structure Prediction with Equivariant Flow Matching},
  author={Rao, Anika and Kaczmarek, Tomasz and Pham, Linh},
  journal={Nature Methods},
  year={2025}
}
```
