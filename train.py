import time
from data import get_batches
from eval import evaluate_loss
import pandas as pd
from matplotlib import pyplot as plt

import torch
def train(model, optimizer, dataset, output_path, scheduler=None, config=None, print_logs=False, l2_lambda=0.0001, tolerance=2):
    losses = []
    best_val_loss = float('inf')
    best_model = None
    consec_increases = 0  # Count of consecutive increases in validation loss
    start_time = time.time()
    

    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])
        _, loss, _ = model(xs, targets=ys)

        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)
        loss += l2_lambda * l2_reg
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()

        if epoch % config['log_interval'] == 0:
            batch_time = time.time() - start_time
            x = evaluate_loss(model, dataset, config)
            losses.append(x)

            if x['val'] < best_val_loss:
                best_val_loss = x['val']
                best_model = model.state_dict()  # Save the best model state
                consec_increases = 0  # Reset consecutive increases
            else:
                consec_increases += 1  # Increment consecutive increases
            
            if consec_increases > tolerance:
                print(f"Early stopping at epoch {epoch} due to increase in validation loss.")
                break  # Early stopping

            if print_logs:
                lr = optimizer.param_groups[0]['lr']
                print_logs_function(lr, epoch, x, batch_time, config)

    # Save the best model outside the loop
    if best_model is not None:
        torch.save(best_model, output_path)
        print(f"Best model saved with validation loss: {best_val_loss}")

    print_final_results(losses)

def print_logs_function(lr, epoch, x, batch_time, config):
    # Function to print logs, can be customized as needed
    print(f"Epoch {epoch} | Train Loss: {x['train']:.3f} | Val Loss: {x['val']:.3f} | LR: {lr:.6f} | Time: {batch_time:.3f} | Num_Head: {config['n_heads']}| batch_size: {config['batch_size']}| context_window: {config['context_window']} |  ETA: {batch_time * (config['epochs'] - epoch) / config['log_interval']:.3f}")

def print_final_results(losses):
    # Function to print final results, can be customized as needed
    df_losses = pd.DataFrame(losses)
    plt.figure(figsize=(10, 6))
    plt.plot(df_losses['train'], label='Train Loss')
    plt.plot(df_losses['val'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig("losses.png")
