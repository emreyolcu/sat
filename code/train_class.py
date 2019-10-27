import logging
import pdb
from os.path import join

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim

import util
from data_class import create_batches, load_dir
from gnn import GraphClassifier

logger = logging.getLogger(__name__)


def log(epoch, batch_count, avg_loss, avg_acc):
    logger.info(
        'Epoch: {:4d},  Iter: {:8d},  Loss: {:.4f},  Acc: {:.4f}'.format(
            epoch, batch_count, avg_loss, avg_acc
        )
    )


def pred_correct(z, y):
    return torch.sum((z > 0.5) == y.byte()).item()


def eval(model, criterion, batches):
    with torch.no_grad():
        loss = 0
        correct = 0
        sample_count = 0
        for data in batches:
            z = model(data)
            sample_count += data.y.shape[0]
            loss += criterion(z, data.y).item() * data.y.shape[0]
            correct += pred_correct(z, data.y)
    return loss / sample_count, correct / sample_count


def load_data(path, no_dev=False):
    train_data = load_dir(join(path, 'train'))
    dev_data = load_dir(join(path, 'dev')) if not no_dev else None
    return (train_data, dev_data)


def main():
    config, device = util.setup()
    logger.setLevel(getattr(logging, config['log_level'].upper()))

    train_data, dev_data = load_data(config['data_path'], config['no_dev'])
    if dev_data:
        dev_batches = list(create_batches(dev_data, config['eval_batch_size'], device))

    gnn = GraphClassifier(2, config['gnn_hidden_size'], config['readout_hidden_size']).to(device)
    optimizer = optim.Adam(gnn.parameters(), lr=config['lr'])
    criterion = nn.BCELoss()

    batch_count = 0
    best_dev_acc = 0

    for epoch in range(1, config['epochs'] + 1):
        train_loss = train_correct = sample_count = 0

        for data in create_batches(train_data, config['batch_size'], device):
            gnn.train()
            z = gnn(data)
            loss = criterion(z, data.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_count += 1
            sample_count += data.y.shape[0]
            train_loss += loss.item() * data.y.shape[0]
            train_correct += pred_correct(z, data.y)

            if batch_count % config['report_interval'] == 0:
                log(epoch, batch_count, train_loss / sample_count, train_correct / sample_count)
                train_loss = train_correct = sample_count = 0

            if not config['no_dev'] and batch_count % config['eval_interval'] == 0:
                gnn.eval()
                dev_loss, dev_acc = eval(gnn, criterion, dev_batches)
                logger.info('(Dev)  Loss: {:.4f},  Acc: {:.4f}'.format(dev_loss, dev_acc))
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    torch.save(gnn, join(config['dir'], 'model.pth'))

    torch.save(gnn, join(config['dir'], 'model_final.pth'))


if __name__ == '__main__':
    main()
