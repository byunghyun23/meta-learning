import os
import torch
import matplotlib.pyplot as plt
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters


def get_dataloader(config):
    train_dataset = omniglot(
        folder=config['dataset_folder'],
        shots=config['num_shots'],
        # test_shots=1, # default = shots
        ways=config['num_ways'],
        shuffle=True,
        meta_train=True,
        download=config['download'],
    )
    train_dataloader = BatchMetaDataLoader(
        train_dataset, batch_size=config['task_batch_size'], shuffle=True
    )

    val_dataset = omniglot(
        folder=config['dataset_folder'],
        shots=config['num_shots'],
        # test_shots=1, # default = shots
        ways=config['num_ways'],
        shuffle=True,
        meta_val=True,
        download=config['download'],
    )
    val_dataloader = BatchMetaDataLoader(
        val_dataset, batch_size=config['task_batch_size'], shuffle=True
    )

    test_dataset = omniglot(
        folder=config['dataset_folder'],
        shots=config['num_shots'],
        # test_shots=1, # default = shots
        ways=config['num_ways'],
        shuffle=True,
        meta_test=True,
        download=config['download'],
    )
    test_dataloader = BatchMetaDataLoader(
        test_dataset, batch_size=config['task_batch_size'], shuffle=True
    )

    return train_dataloader, val_dataloader, test_dataloader


def meta_train(device, task_batch_size, task_batch, model, criterion, optimizer):
    model.train()

    support_xs = task_batch['train'][0].to(device=device)
    support_ys = task_batch['train'][1].to(device=device)
    query_xs = task_batch['test'][0].to(device=device)
    query_ys = task_batch['test'][1].to(device=device)

    outer_loss = torch.tensor(0.0, device=device)
    accuracy = torch.tensor(0.0, device=device)

    for support_x, support_y, query_x, query_y in zip(support_xs, support_ys, query_xs, query_ys):
        support_prob = model(support_x)
        inner_loss = criterion(support_prob, support_y)

        params = gradient_update_parameters(
            model, inner_loss, step_size=0.4, first_order=True
        )

        query_prob = model(query_x, params=params)
        outer_loss += criterion(query_prob, query_y)

        with torch.no_grad():
            _, query_pred = torch.max(query_prob, dim=-1)
            accuracy += torch.mean(query_pred.eq(query_y).float())

    outer_loss.div_(task_batch_size)

    model.zero_grad()
    outer_loss.backward()
    optimizer.step()

    accuracy.div_(task_batch_size)
    return accuracy.item(), outer_loss.item()


def meta_test(device, task_batch_size, task_batch, model, criterion):
    model.eval()

    support_xs = task_batch['train'][0].to(device=device)
    support_ys = task_batch['train'][1].to(device=device)
    query_xs = task_batch['test'][0].to(device=device)
    query_ys = task_batch['test'][1].to(device=device)

    outer_loss = torch.tensor(0.0, device=device)
    accuracy = torch.tensor(0.0, device=device)

    for support_x, support_y, query_x, query_y in zip(support_xs, support_ys, query_xs, query_ys):
        support_prob = model(support_x)
        inner_loss = criterion(support_prob, support_y)

        params = gradient_update_parameters(
            model, inner_loss, step_size=0.4, first_order=True
        )

        query_prob = model(query_x, params=params)
        outer_loss += criterion(query_prob, query_y)

        with torch.no_grad():
            _, query_pred = torch.max(query_prob, dim=-1)
            accuracy += torch.mean(query_pred.eq(query_y).float())

    outer_loss.div_(task_batch_size)
    accuracy.div_(task_batch_size)
    return accuracy.item(), outer_loss.item()


def save_model(output_folder, model, title):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    filename = os.path.join(output_folder, title)

    with open(filename, 'wb') as f:
        state_dict = model.state_dict()
        torch.save(state_dict, f)
    print('Model is saved in', filename)


def load_model(output_folder, model, title):
    filename = os.path.join(output_folder, title)
    model.load_state_dict(torch.load(filename))
    print('Model is loaded')


def print_graph(train_accuracies, val_accuracies, train_losses, val_losses):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].plot(train_accuracies, label='train_acc')
    axs[0].plot(val_accuracies, label='test_acc')
    axs[0].set_title('Accuracy')
    axs[0].legend()

    axs[1].plot(train_losses, label='train_loss')
    axs[1].plot(val_losses, label='test_loss')
    axs[1].set_title('Loss')
    axs[1].legend()

    plt.show()