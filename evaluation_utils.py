import os

import torch
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.utils import make_grid

from data.dataset import MaskedDataset
from models.models_classifiers import Classifier

def STL10_eval(input_path, output_path, batch_size, learning_rate, model, device, generator, finetuning=False, writer=None, verbose=True):
    print('Starting {}...'.format('finetuning' if finetuning else 'readout'))

    data_main = datasets.STL10(root=input_path, split='train')
    
    data_main, data_test = random_split(data_main, [4000, 1000], generator=generator)
    
    train_dataset = MaskedDataset(data_main)
    test_dataset = MaskedDataset(data_test)
    
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        generator = generator,
        num_workers = 32,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False,
        generator = generator,
        num_workers = 32,
        pin_memory=True
    )

    if not finetuning:
        model.eval();
    else:
        model.train();

    classifier = Classifier(encoded_dim=128*12*12, num_classes=10)
    classifier = classifier.to(device)
    if verbose:
        print(classifier)

    loss_function = torch.nn.CrossEntropyLoss()
    MSE_loss = torch.nn.MSELoss()

    # Adam Optimizer
    optimizer = torch.optim.Adam(
        classifier.parameters() if not finetuning else list(model.parameters()) + list(classifier.parameters()),
        lr = learning_rate,
        weight_decay = 1e-8
    )

    lowest_test_loss = (9999, 0)
    improvement_window = 0

    for epoch in range(1000):
        loss = 0
        for batch_features, _, labels, _ in train_loader:
            batch_features, labels = batch_features.to(device), labels.to(device)

            # reset the gradients
            optimizer.zero_grad()

            # retrieve representations
            if not finetuning:
                with torch.no_grad():
                    _, representations = model(batch_features)
            else:
                _, representations = model(batch_features)

            # compute class predictions
            classifications = classifier(torch.flatten(representations, start_dim=1))

            # compute classification loss
            train_loss = loss_function(classifications, labels)

            # compute gradients
            train_loss.backward()

            # perform weight updates based on current gradients
            optimizer.step()

            loss += train_loss.item()

        # compute epoch training loss
        loss = loss / len(train_loader)

        if writer:
            writer.add_scalar('{}/loss/train'.format('finetuning' if finetuning else 'readout'), loss, epoch)

        if verbose:
            print("Epoch : {}, train-loss = {:.8f}".format(epoch + 1, loss))
        
        test_loss = 0
        for batch_features, _, labels, _ in test_loader:
            batch_features, labels = batch_features.to(device), labels.to(device)
            
            # retrieve representations
            with torch.no_grad():
                _, representations = model(batch_features)
            
                # compute class predictions
                classifications = classifier(torch.flatten(representations, start_dim=1))

                # compute classification loss
                test_loss += loss_function(classifications, labels).item()
        
        # compute epoch training loss
        test_loss = test_loss / len(test_loader)

        if writer:
            writer.add_scalar('{}/loss/test'.format('finetuning' if finetuning else 'readout'), test_loss, epoch)

        if verbose:
            print("Epoch : {}, test-loss = {:.8f}".format(epoch + 1, test_loss))

        if test_loss < lowest_test_loss[0]:
            if abs(test_loss - lowest_test_loss[0]) < 1e-4:
                if verbose:
                    print('Very small improvement')
                improvement_window += 1
            else:
                improvement_window = 0

            lowest_test_loss = (test_loss, epoch + 1)

            if finetuning:
                torch.save(model.state_dict(), output_path + '/finetuning_model.pt')
            torch.save(classifier.state_dict(), output_path + '/{}_classifier.pt'.format('finetuning' if finetuning else 'readout'))
        else:
            if verbose:
                print('No improvement in epoch {}'.format(epoch + 1))
            improvement_window += 1

        if improvement_window >= 10:
            print('Ending classifier training')
            print('Lowest validation-loss of {} at epoch {}'.format(lowest_test_loss[0], lowest_test_loss[1]))

            break

    if finetuning:
        if os.path.exists(output_path + '/finetuning_model.pt'):
            model.load_state_dict(torch.load(output_path + '/finetuning_model.pt', map_location=device))
            model.eval();
        else:
            raise Exception('Model save-file not found')

    if os.path.exists(output_path + '/{}_classifier.pt'.format('finetuning' if finetuning else 'readout')):
        classifier.load_state_dict(torch.load(output_path + '/{}_classifier.pt'.format('finetuning' if finetuning else 'readout'), map_location=device))
    else:
        raise Exception('Classifier save-file not found')
    classifier.eval();

    # Training accuracy
    data_main = datasets.STL10(root=input_path, split='train')
    train_dataset = MaskedDataset(data_main)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
        num_workers=32,
        pin_memory=True
    )

    correct = 0
    total = 0

    trained_loss = 0

    with torch.no_grad():
        for batch_features, full_res, labels, _ in train_loader:
            batch_features, full_res, labels = batch_features.to(device), full_res.to(device), labels.to(device)

            reconstructions, representations = model(batch_features)

            classifications = classifier(torch.flatten(representations, start_dim=1))
            
            _, predicted = torch.max(classifications.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            trained_loss += MSE_loss(reconstructions, full_res).item()

    print('{} out of {} images correct: {} %'.format(correct, total, round((100 * correct / total), 2)))

    if writer:
        writer.add_scalar('{}/accuracy/train'.format('finetuning' if finetuning else 'readout'), round((100 * correct / total), 2), 0)

    trained_loss = trained_loss / len(train_loader)

    print('Loss of reconstructions {}'.format(trained_loss))

    if writer:
        writer.add_scalar('{}/MSE/train'.format('finetuning' if finetuning else 'readout'), trained_loss, 0)

    # Testing accuracy
    data_test = datasets.STL10(root=input_path, split='test')
    test_dataset = MaskedDataset(data_test)

    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False,
        generator = generator,
        num_workers = 32,
        pin_memory=True
    )

    correct = 0
    total = 0

    trained_loss = 0

    with torch.no_grad():
        for batch_features, full_res, labels, _ in test_loader:
            batch_features, full_res, labels = batch_features.to(device), full_res.to(device), labels.to(device)

            reconstructions, representations = model(batch_features)

            classifications = classifier(torch.flatten(representations, start_dim=1))
            
            _, predicted = torch.max(classifications.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            trained_loss += MSE_loss(reconstructions, full_res).item()

    print('{} out of {} images correct: {} %'.format(correct, total, round((100 * correct / total), 2)))

    if writer:
        writer.add_scalar('{}/accuracy/test'.format('finetuning' if finetuning else 'readout'), round((100 * correct / total), 2), 0)

    trained_loss = trained_loss / len(test_loader)

    if verbose:
        print('Loss of reconstructions {}'.format(trained_loss))

    if writer:
        writer.add_scalar('{}/MSE/test'.format('finetuning' if finetuning else 'readout'), trained_loss, 0)

    # Reconstructions
    grid_input = make_grid(batch_features[:32])
    grid_target = make_grid(full_res[:32])
    grid_reconstructions = make_grid(reconstructions[:32])

    if writer:
        writer.add_image('Images/{}/Inputs'.format('finetuning' if finetuning else 'readout'), grid_input)
        writer.add_image('Images/{}/Targets'.format('finetuning' if finetuning else 'readout'), grid_target)
        writer.add_image('Images/{}/Reconstructions'.format('finetuning' if finetuning else 'readout'), grid_reconstructions)

    return
