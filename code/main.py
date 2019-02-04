import copy
import os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# directory for the data used between multiple runs
from model import DNN, init_he_normal
from utils import numpy_to_dataloaders, data_to_numpy

STATE_DIR = './state'
MNIST_DATA_DIR = os.path.join(STATE_DIR, 'mnist_data/')
MODEL_04_PATH = os.path.join(STATE_DIR, 'mnist_04_best.pt')
MODEL_59_PATH = os.path.join(STATE_DIR, 'mnist_59_best.pt')

INPUT_DIM = 28 * 28
HIDDEN_DIM = 100
OUTPUT_DIM = 5


def train_model(model, device, X_train, y_train, criterion, optimizer,
                batch_size=128, X_valid=None, y_valid=None,
                scheduler=None, n_epochs=25, early_stopping=5, verbose=True):
    since = time.time()

    best_model_wts = None
    best_acc = 0.0
    n_epochs_no_impr = 0

    losses = {'train': [], 'valid': []}
    accs = {'train': [], 'valid': []}

    # building data loaders
    dataloaders, dataset_sizes = numpy_to_dataloaders(X_train, y_train, batch_size, X_valid, y_valid)

    phases = dataloaders.keys()
    for epoch in range(1, n_epochs + 1):
        if verbose:
            print('Epoch {}/{}'.format(epoch, n_epochs))

        # each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for inputs, labels in dataloaders[phase]:
                # inputs = inputs.view(-1, 28*28)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            losses[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)

            if verbose:
                print(' - {:5s} loss: {:.4f} acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    n_epochs_no_impr = 0
                else:
                    n_epochs_no_impr += 1

        if n_epochs_no_impr >= early_stopping:
            # stop early
            if verbose:
                print('Early stopping')
            break

    time_elapsed = time.time() - since
    if verbose:
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:.4f}'.format(best_acc))

    # load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    return model, losses, accs


def main():
    print('> Starting execution...')

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--fit', action='store_true',
                       help='fit the tuned model on digits 0-4')
    group.add_argument('--transfer', action='store_true',
                       help='train a pretrained model on digits 5-9')

    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=50, metavar='E',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='L',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--early-stopping', type=int, default=7, metavar='E',
                        help='early stopping (default: 7 epochs)')
    parser.add_argument('--size', type=int, default=100, metavar='S',
                        help='size of the training data for transfer learning (default: 100)')

    parser.add_argument('--seed', type=int, default=23, metavar='S',
                        help='random seed (default: 23)')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()  # use cuda if available
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)  # random seed

    print('> Loading MNIST data')
    train_set = datasets.MNIST(MNIST_DATA_DIR, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))

    test_set = datasets.MNIST(MNIST_DATA_DIR, train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))

    train_digits_04 = np.where(train_set.train_labels < 5)[0]
    train_digits_59 = np.where(train_set.train_labels > 4)[0]

    test_digits_04 = np.where(test_set.test_labels < 5)[0]
    test_digits_59 = np.where(test_set.test_labels > 4)[0]

    if args.fit:
        # Training the tuned model on digits 0-4
        print('> Training a new model on MNIST digits 0-4')

        X_train_04, y_train_04, X_valid_04, y_valid_04 = data_to_numpy(
            train_set, test_set, INPUT_DIM, train_digits_04, test_digits_04
        )

        torch.manual_seed(args.seed)

        print('> Initializing the model')

        model = DNN(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, batch_norm=True)
        model.apply(init_he_normal)  # He initialization

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        print('> Training the model')
        model, _, _ = train_model(
            model, device, X_train_04, y_train_04,
            criterion, optimizer,
            X_valid=X_valid_04, y_valid=y_valid_04, batch_size=args.batch_size,
            n_epochs=args.epochs, early_stopping=args.early_stopping
        )

        print(f'> Saving the model state at {MODEL_04_PATH}')
        torch.save(model.state_dict(), MODEL_04_PATH)
    elif args.transfer:
        # Transfer learning
        print('> Training a model on MNIST digits 5-9 from a pretrained model for digits 0-4')

        if os.path.isfile(MODEL_04_PATH):
            print('> Loading the pretrained model')

            model = DNN(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, batch_norm=True).to(device)
            model.load_state_dict(torch.load(MODEL_04_PATH))

            for param in model.parameters():
                param.requires_grad = False

            # Parameters of newly constructed modules have requires_grad=True by default
            model.fc4 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            model.fc5 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            model.out = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

            print('> Using saved model state')
        else:
            print('> Model state file is not found, fit a model before the transfer learning')
            print('> Stopping execution')
            return

        X_train_59, y_train_59, X_valid_59, y_valid_59 = data_to_numpy(
            train_set, test_set, INPUT_DIM, train_digits_59[:args.size], test_digits_59
        )

        # fixing the issues with labels
        y_train_59 = y_train_59 - 5
        y_valid_59 = y_valid_59 - 5

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        print('> Training the model')
        model, _, _ = train_model(
            model, device, X_train_59, y_train_59,
            criterion, optimizer,
            X_valid=X_valid_59, y_valid=y_valid_59, batch_size=args.batch_size,
            n_epochs=args.epochs, early_stopping=args.early_stopping
        )

        print(f'> Saving the model state at {MODEL_59_PATH}')
        torch.save(model.state_dict(), MODEL_59_PATH)
    else:
        print('> Incorrect mode, try either `--fit` or `--transfer`')
        print('> Stopping execution')


if __name__ == '__main__':
    main()
