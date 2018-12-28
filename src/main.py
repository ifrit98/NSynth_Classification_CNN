"""
NSynth classification using PyTorch

Authors: Japheth Adhavan, Jason St. George
Reference: Sasank Chilamkurthy <https://chsasank.github.io>
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import utils
import models
import data
import visualize
import time
import os
import argparse

import logging
logging.basicConfig(level=logging.INFO)

gpu_idx = utils.get_free_gpu()
device = torch.device("cuda:{}".format(gpu_idx))
logging.info("Using device cuda:{}".format(gpu_idx))


def train_model(model, dataloaders, criterion, optimizer, scheduler, network_type, num_epochs=10):
    """

    :param model:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :return:
    """
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ["train", "val"]}
    model_loss = {x: [0 for _ in range(num_epochs)] for x in ["train", "val"]}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # iterate over data
            for batch_idx, (samples, labels, targets) in enumerate(dataloaders[phase]):
                inputs = samples.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # aggregate statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'train' and batch_idx % 50 == 0:
                    logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        (epoch + 1),
                        (batch_idx + 1) * len(samples),
                        dataset_sizes[phase],
                        100. * (batch_idx + 1) / len(dataloaders[phase]),
                        loss.item()))


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            model_loss[phase][epoch] = epoch_loss

            logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase.capitalize(), epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, "./models/{}Network.pt".format(network_type))

        print()

    time_elapsed = time.time() - since
    logging.info('Training completed in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best overall val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, model_loss


# def test(model, dataloader):
#     spreads = [[0 for y in range(10)] for x in range(10)]
#     test_c = [0 for x in range(10)]
#     test_t = [0 for x in range(10)]
#     c_done = [False for x in range(10)]
#     i_done = [False for x in range(10)]
#     c_samples, i_samples = 0, 0
#     y_test, y_pred = [], []
#
#     correct = (preds == labels).squeeze()
#     np_predicted = preds.cpu().numpy()  # Get vector of int class output labels
#     y_pred.extend(np_predicted)
#     y_test.extend(if_label.cpu().numpy())
#
#     if i_samples < 10 and c_samples < 10:
#         for i in range(len(outputs)):
#             label = str(labels[i])  # e.g. 'tensor(0)'
#             label = int(label[7])  # 0
#             test_c[label] += correct[i].item()
#             test_t[label] += 1
#
#             if np_predicted[i] != label:
#                 spreads[label][np_predicted[i]] += 1
#                 if i_samples < 10:
#                     i_samples += visualize.save_samples(inputs[i],
#                                                         np_predicted[i], label,
#                                                         i_done, False, CLASS_NAMES)
#             else:
#                 if c_samples < 10:
#                     c_samples += visualize.save_samples(inputs[i], None, label,
#                                                         c_done, True, CLASS_NAMES)


def test(model, test_loader, criterion, classes):
    model.eval()
    test_loss = 0
    correct = 0
    no_of_classes = len(classes)
    spread = [([0] * no_of_classes) for _ in range(no_of_classes)]
    examples = [{} for _ in range(no_of_classes)]
    y_test, y_pred = [], []

    with torch.no_grad():
        for data, labels, target in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            test_loss += criterion(output, labels).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            actual = labels.view_as(pred)
            is_correct = pred.equal(actual)
            label_actual = int(labels) if int(labels) < 9 else 9
            label_pred = int(pred) if int(pred) < 9 else 9
            spread[label_actual][label_pred] += 1
            correct += 1 if is_correct else 0
            examples[label_actual][is_correct] = (data, label_pred)
            y_pred.append(label_pred)
            y_test.append(label_actual)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return y_test, y_pred, spread, examples


def main(args):
    np.warnings.filterwarnings('ignore')

    os.makedirs("./graphs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    model = {
        "Simple": models.SimpleNetwork,
        "Epic" : models.EpicNetwork,
        "Bonus" : models.BonusNetwork
    }[args.network]().to(device)

    classes = ['bass', 'brass', 'flute', 'guitar', 'keyboard',
                   'mallet', 'organ', 'reed', 'string', 'vocal']

    if args.network == "Bonus":
        classes = ['acoustic', 'electronic', 'synthetic']


    model.double()
    criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=args.step, gamma=args.gamma)

    if not args.test:
        train_loader = data.get_data_loader("train", batch_size=args.batch_size,
                                            shuffle=True, num_workers=4, network=args.network)
        valid_loader = data.get_data_loader("valid", network=args.network)

        dataloaders = {
            "train": train_loader,
            "val": valid_loader
        }

        logging.info('Training...')
        model, model_loss = train_model(model, dataloaders,
                                        criterion, optimizer_conv,
                                        exp_lr_scheduler,
                                        args.network,
                                        num_epochs=args.epochs)

        visualize.plot_loss(model_loss, "{}Network".format(args.network))
    else:
        logging.info('Testing...')
        model.load_state_dict(torch.load("./models/{}Network.pt".format(args.network)))
        test_loader = data.get_data_loader("test", network=args.network)
        y_test, y_pred, spreads, examples = test(model, test_loader, criterion, classes)

        visualize.plot_histograms(classes, spreads, type=args.network)
        visualize.plot_confusion_matrix(y_test, y_pred, classes, type=args.network)
        visualize.save_samples(examples, classes)

    logging.info('Completed Successfully!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NSynth classifier')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test', action='store_true', default=False,
                        help='disables training, loads model')
    parser.add_argument('--network', default='Epic', const='Epic', nargs="?", choices=['Simple', 'Epic', 'Bonus'],
                        help='Choose the type of network from Simple, Epic and Bonus (default: Epic)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--step', type=int, default=3, metavar='N',
                        help='number of epochs to decrease learn-rate (default: 3)')
    parser.add_argument('--gamma', type=float, default=0.1, metavar='N',
                        help='factor to decrease learn-rate (default: 0.1)')
    main(parser.parse_args())