import click
import torch
import torch.nn as nn

from util import get_dataloader, save_model, load_model, print_graph, meta_train, meta_test
from model import MetaModel
from tqdm import tqdm


@click.command()
@click.option('--dataset_folder', default='dataset', help='Dataset Folder Name')
@click.option('--download', default=True, help='Download or Not')
@click.option('--num_shots', default=5, help='Number of Shots')
@click.option('--num_ways', default=5, help='Number of Ways')
@click.option('--output_folder', default='saved_model', help='Output Folder Name')
@click.option('--task_batch_size', default=32, help='Task Batch Size')
@click.option('--num_task_batch_train', default=600, help='Number of Train Task Batch')
@click.option('--num_task_batch_test', default=200, help='Number of Test Task Batch')
@click.option('--model_name', default='maml_classification', help='Model Name to save')
@click.option('--device', default='cuda', help='Device Type')
def run(dataset_folder, download, num_shots, num_ways, output_folder, task_batch_size,
        num_task_batch_train, num_task_batch_test, model_name, device):

    config = {
        'dataset_folder': dataset_folder,
        'download': download,
        'num_shots': num_shots,
        'num_ways': num_ways,
        'output_folder': output_folder,
        'task_batch_size': task_batch_size,
        'num_task_batch_train': num_task_batch_train,
        'num_task_batch_test': num_task_batch_test,
        'model_name': model_name,
        'device': device
    }

    # Get DataLoaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(config)

    # Get the MAML Model
    model = MetaModel(in_channels=1, out_features=config['num_ways']).to(
        device=config['device']
    )

    # Set Options
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the MAML Model
    with tqdm(zip(train_dataloader, val_dataloader), total=config['num_task_batch_train']) as pbar:
        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []

        for task_batch_idx, (train_batch, val_batch) in enumerate(pbar):
            if task_batch_idx >= config['num_task_batch_train']:
                break

            train_accuracy, train_loss = meta_train(
                device=config['device'],
                task_batch_size=config['task_batch_size'],
                task_batch=train_batch,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
            )
            val_accuracy, val_loss = meta_test(
                device=config['device'],
                task_batch_size=config['task_batch_size'],
                task_batch=val_batch,
                model=model,
                criterion=criterion,
            )

            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            pbar.set_postfix(
                train_accuracy='{0:.4f}'.format(train_accuracy),
                val_accuracy='{0:.4f}'.format(val_accuracy),
                train_loss='{0:.4f}'.format(train_loss),
                val_loss='{0:.4f}'.format(val_loss),
            )

    # Save the MAML Model
    save_model(output_folder=config['output_folder'], model=model, title=config['model_name'] + '.th')

    # Show Accuracy and Loss
    print_graph(train_accuracies, val_accuracies, train_losses, val_losses)

    # Load the MAML Model
    load_model(output_folder=config['output_folder'], model=model, title=config['model_name'] + '.th')

    # Test the MAML Model
    with tqdm(test_dataloader, total=config['num_task_batch_test']) as pbar:
        sum_test_accuracies = 0.0
        sum_test_losses = 0.0

        for task_batch_idx, test_batch in enumerate(pbar):
            if task_batch_idx >= config['num_task_batch_test']:
                break

            test_accuracy, test_loss = meta_test(
                device=config['device'],
                task_batch_size=config['task_batch_size'],
                task_batch=test_batch,
                model=model,
                criterion=criterion,
            )

            sum_test_accuracies += test_accuracy
            sum_test_losses += test_loss
            pbar.set_postfix(
                test_accuracy='{0:.4f}'.format(sum_test_accuracies / (task_batch_idx + 1)),
                test_loss='{0:.4f}'.format(sum_test_losses / (task_batch_idx + 1)),
            )


if __name__ == '__main__':
    run()