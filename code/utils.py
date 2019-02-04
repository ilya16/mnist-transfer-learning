import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset


def build_dataloaders(train_set, test_set, batch_size=256, train_indices=None, test_indices=None):
    """ Returns data loaders and dataset_sizes """

    if train_indices is None:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices)
        )

    if test_indices is None:
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=True
        )
    else:
        test_loader = DataLoader(
            test_set, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices)
        )

    dataloaders = {
        'train': train_loader,
        'valid': test_loader
    }
    dataset_sizes = {
        'train': len(train_indices),
        'valid': len(test_indices)
    }

    return dataloaders, dataset_sizes


def data_to_numpy(train_set, test_set, input_dim, train_indices=None, test_indices=None):
    """ Returns datasets as NumPy arrays """
    batch_size = max(len(train_indices), len(test_indices))
    dataloaders, dataset_sizes = build_dataloaders(train_set, test_set, batch_size,
                                                   train_indices, test_indices)

    X_train, y_train, X_valid, y_valid = None, None, None, None

    for inputs, labels in dataloaders['train']:
        X_train = inputs.view(-1, input_dim).data.numpy()
        y_train = labels.data.numpy()

    for inputs, labels in dataloaders['valid']:
        X_valid = inputs.view(-1, input_dim).data.numpy()
        y_valid = labels.data.numpy()

    return X_train, y_train, X_valid, y_valid


def numpy_to_dataloaders(X_train, y_train, batch_size, X_valid=None, y_valid=None):
    """ Transforms NumPy arrays into dataloaders """
    dataset_sizes = {'train': X_train.shape[0]}
    X_train = torch.Tensor(X_train)
    y_train = torch.LongTensor(y_train)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True,
    )
    dataloaders = {'train': train_loader}

    if X_valid is not None and y_valid is not None:
        dataset_sizes['valid'] = X_valid.shape[0]
        X_valid = torch.Tensor(X_valid)
        y_valid = torch.LongTensor(y_valid)
        val_loader = DataLoader(
            TensorDataset(X_valid, y_valid),
            batch_size=batch_size, shuffle=True,
        )
        dataloaders['valid'] = val_loader

    return dataloaders, dataset_sizes
