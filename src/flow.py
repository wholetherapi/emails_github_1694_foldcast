import torch


def integrate(model, coords, features, edge_index, edge_attr, steps: int = 32):
    dt = 1.0 / steps
    x = coords.clone()
    for step in range(steps):
        t = torch.full((x.size(0), 1), step * dt, device=x.device)
        velocity = model(x, features, edge_index, edge_attr)
        x = x + dt * velocity
    return x
