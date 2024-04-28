import numpy as np
from data import get_batches
def evaluate_loss(model, dataset, config):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):
            xb, yb = get_batches(dataset, config['batch_size'], config['context_window'])
            _, loss, _ = model(xb, yb)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    model.train()