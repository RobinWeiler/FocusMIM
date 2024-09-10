import click
import signal

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.utils import make_grid
from torchinfo import summary

from data.dataset import MaskedDataset
from models.models_MAE import CNN_MAE
from evaluation_utils import STL10_eval

kill_signal = False

def signal_handler(sig, frame):
    global kill_signal

    print('Kill signal caught: setting global flag for main loop to terminate')
    kill_signal = True

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Running on {}'.format(device))


@click.command()
@click.option('--input-path')
@click.option('--output-path')
@click.option('-mm', '--masking-mode')
@click.option('-mr', '--masking-ratio')
@click.option('-b', '--blur')
@click.option('-rc', '--random-crop')
@click.option('-lr', '--learning-rate')
@click.option('-e', '--epochs')
@click.option('-sd', '--seed')
@click.option('-sp', '--segment-path')
@click.option('-remms', '--remove-missing-segments')
def main(input_path, output_path, masking_mode, masking_ratio, blur, random_crop, learning_rate, epochs, seed, segment_path, remove_missing_segments):

    print('Masking mode: {}'.format(masking_mode))
    print('Masking ratio: {}'.format(masking_ratio))
    masking_ratio = float(masking_ratio) if masking_ratio else 0.5

    print('Using blurred masks: {}'.format(blur))
    if blur == 'True':
        blur = True
    elif blur == 'False':
        blur = False
    else:
        raise Exception('{} not a valid argument for blur. Use "True" or "False".')

    print('Using random crop: {}'.format(random_crop))
    if random_crop == 'True':
        random_crop = True
    elif random_crop == 'False':
        random_crop = False
    else:
        raise Exception('{} not a valid argument for random-crop. Use "True" or "False".')

    seed = int(seed) if seed else 0
    print('Seed: {}'.format(seed))

    torch.manual_seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    batch_size = 512
    learning_rate = float(learning_rate) if learning_rate else 1e-3
    print('Learning rate: {}'.format(learning_rate))

    epochs = int(epochs) if epochs else 500
    print('Num epochs: {}'.format(epochs))

    writer = SummaryWriter(output_path + '/tensorboard')


    # Load data
    print('Loading data...')
    print(input_path)

    data_main = datasets.STL10(root=input_path, split='unlabeled')

    if segment_path:
        print('Removing missing segmentation masks: {}'.format(remove_missing_segments))
        if remove_missing_segments == 'True':
            remove_missing_segments = True
        elif remove_missing_segments == 'False':
            remove_missing_segments = False
        else:
            raise Exception('{} not a valid argument for remove-missing-segments. Use "True" or "False".')
        train_dataset = MaskedDataset(data_main, masking_mode=masking_mode, masking_ratio=masking_ratio, blur=blur, random_crop=random_crop, segment=True, segment_path=segment_path, remove_missing_segments=remove_missing_segments, seed=seed)
    else:
        train_dataset = MaskedDataset(data_main, masking_mode=masking_mode, masking_ratio=masking_ratio, blur=blur, random_crop=random_crop, seed=seed)
    print('{} samples'.format(len(train_dataset)))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=32,
        pin_memory=True
    )


    # Model
    print('Creating model...')

    model = CNN_MAE()
    model = model.to(device)

    summary(model, (batch_size, 3, 96, 96), device=device)


    # Loss functions
    class MaskedMSELoss(torch.nn.Module):
        def __init__(self):
            super(MaskedMSELoss, self).__init__()

        def forward(self, input, target, mask):
            error = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
            mse = torch.sum(error) / torch.sum(mask)

            return mse

    reconstruction_loss_function = MaskedMSELoss() if masking_mode != None else torch.nn.MSELoss()

    def off_diagonal(x):
        n, m = x.shape
        assert n == m

        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = learning_rate,
        weight_decay = 1e-8
    )


    print('Start training...')
    lowest_loss = 99999
    for epoch in range(epochs):
        total_loss = 0
        total_reconstruction_loss = 0
        total_cov_loss = 0
        
        for batch_features, full_res, _, mask in train_loader:
            if kill_signal:
                print('Job was killed! Stop training...')

                print ('Saving final checkpoint for epoch {}'.format(epoch))
                torch.save(model.state_dict(), output_path+'/model.pt')

                break

            batch_features, full_res, mask = batch_features.to(device), full_res.to(device), mask.to(device)

            # reset the gradients
            optimizer.zero_grad()
            
            # compute reconstructions
            reconstructions, reps = model(batch_features)
            
            # compute loss
            rc_loss = reconstruction_loss_function(reconstructions, full_res, mask)
            total_reconstruction_loss += rc_loss.item()

            reps_flat = reps.flatten(start_dim=1)
            representation_dim = reps_flat.shape[1]
            mean = torch.mean(reps_flat, dim=0)
            stddev = torch.std(reps_flat, dim=0)
            reps_normalized = (reps_flat - mean)/ (stddev + 1e-5)
            cov = (reps_normalized.T @ reps_normalized) / (batch_size - 1)
            cov_loss = off_diagonal(cov).pow_(2).sum().div(representation_dim)
            total_cov_loss += cov_loss.item()
            
            # compute gradients
            train_loss = rc_loss
            train_loss.backward()
            
            # perform weight updates based on gradients
            optimizer.step()
            
            total_loss += train_loss.item()
        
        # compute epoch training loss
        total_loss = total_loss / len(train_loader)
        total_reconstruction_loss = total_reconstruction_loss / len(train_loader)

        writer.add_scalar('Loss/train', total_loss, epoch)
        writer.add_scalar('Loss/train/reconstruction_loss', total_reconstruction_loss, epoch)
        writer.add_scalar('Loss/train/cov_loss', total_cov_loss, epoch)
        
        print("Epoch : {}/{}, training-loss = {:.8f}".format(epoch + 1, epochs, total_loss))
    
        torch.save(model.state_dict(), output_path + '/model.pt')

        if epoch == 0:
            torch.save(model.state_dict(), output_path + '/model_0e.pt')
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), output_path + '/model_{}e.pt'.format(epoch))

        if total_loss < lowest_loss:
            torch.save(model.state_dict(), output_path + '/model_best.pt')

            lowest_loss = total_loss
        else:
            print('No improvement in epoch {}'.format(epoch + 1))

    print('Finished training!')


    # Visualize reconstructions
    test_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=False
    )

    with torch.no_grad():
        for batch_features, full_res, _, mask in test_loader:
            batch_features, full_res, mask = batch_features.to(device), full_res.to(device), mask.to(device)

            print(torch.count_nonzero(mask[0, 0]))

            reconstructions, _ = model(batch_features)

            break

        grid_input = make_grid(batch_features)
        grid_target = make_grid(full_res)
        grid_reconstructions = make_grid(reconstructions)
        grid_mask = make_grid(mask[:, :3])
        
        writer.add_image('Images/Inputs', grid_input)
        writer.add_image('Images/Targets', grid_target)
        writer.add_image('Images/Reconstructions', grid_reconstructions)
        writer.add_image('Images/mask', grid_mask)


    # Linear readout

    # use commented out lines to evaluate in 50-epoch increments
    # models = np.arange(0, 550, 50)

    # for model_index in models:
    #     print(' ----------------------------------- {} ----------------------------------- '.format(model_index))

    #     model.load_state_dict(torch.load(output_path + '/model_{}e.pt'.format(model_index), map_location=device))

    model.load_state_dict(torch.load(output_path + '/model_best.pt', map_location=device))

    STL10_eval(input_path, output_path, batch_size=batch_size, learning_rate=1e-4, model=model, device=device, generator=generator, writer=writer, verbose=True)


    # Linear finetuning
    STL10_eval(input_path, output_path, batch_size=batch_size, learning_rate=1e-5, model=model, device=device, generator=generator, finetuning=True, writer=writer, verbose=True)

    writer.close()



if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_handler)

    main()
