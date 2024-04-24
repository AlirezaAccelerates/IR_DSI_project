import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split


# Train model
def train_model(dataset, model, max_epochs, seed=42, batch_size=1024, split_ratio=0.8, **dataloader_kwargs):
    # Set the random seed for reproducibility
    torch.manual_seed(42)
    
    # Calculate split sizes
    calculate_split_sizes = lambda dataset_size, split_ratio: (int(split_ratio * dataset_size), dataset_size - int(split_ratio * dataset_size))

    # Splitting the dataset
    train_size, eval_size = calculate_split_sizes(len(dataset), split_ratio)
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # Creating dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs)
    val_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, **dataloader_kwargs)

    # Training the model
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, train_dataloader, val_dataloader)

    # Return datasets
    return train_dataset, eval_dataset


def learn_docids(dataset, model, max_epochs, batch_size=1024, **dataloader_kwargs):
    # Creating dataloaders
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **dataloader_kwargs)

    # Training the model
    trainer = pl.Trainer(max_epochs=max_epochs)
    trainer.fit(model, dataloader)


def plot_pl_logs(log_dir, metric, save_dir):
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Capitalize metric name
    metric_str = ' '.join(word.capitalize() for word in metric.split('_'))

    # Find the event file in the log directory
    event_file = next(file for file in os.listdir(log_dir) if 'events.out.tfevents' in file)
    full_path = os.path.join(log_dir, event_file)

    # Load the TensorBoard event file
    ea = event_accumulator.EventAccumulator(full_path)
    ea.Reload()

    # Extracting the scalar 'loss'
    if metric in ea.scalars.Keys():
        metric_df = pd.DataFrame(ea.scalars.Items(metric))

        # Plotting
        plt.figure(figsize=(6, 4))
        plt.plot(metric_df['step'], metric_df['value'], label=f'{metric_str}')
        plt.xlabel('Steps')
        plt.ylabel(f'{metric_str}')
        plt.title(f'{metric_str} Over Time')
        plt.savefig(os.path.join(save_dir, f'{metric}.pdf'), bbox_inches='tight')
    else:
        print(f"{metric} data not found in logs")



