import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def train(n_epochs, net, train_loader, test_loader, use_GPU, patience, path_save):

    # define loss function and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
   
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize for saving model with best validation loss
    valid_loss_best = np.Inf
    # early stopping initialisation
    epochs_no_improve = 0
    
    # move model to either GPU or CPU
    if use_GPU:
        print("Using GPU for training...")
        net.to("cuda")
    else:
        print("Using CPU only for training...")
        net.to("cpu")

    for epoch in range(n_epochs):  
        
        ###################
        # train the model #
        ###################

        net.train() # prep model for training
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']
            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)
            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)
            
            if use_GPU:
                images = images.to('cuda')
                key_pts = key_pts.to('cuda')
            else:
                images = images.to('cpu')
                key_pts = key_pts.to('cpu')
            
            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            # forward pass to get outputs
            output_pts = net(images)
            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)
            # backward pass to calculate the weight gradients
            loss.backward()
            # update the weights
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################    
        # validate the model #
        ######################

        net.eval() # prep model for eval (no dropouts/batch norm)        
        with torch.no_grad():
            for batch_i, data in enumerate(test_loader):
                images = data['image']
                key_pts = data['keypoints']
                
                # flatten pts
                key_pts = key_pts.view(key_pts.size(0), -1)

                # convert variables to floats for regression loss
                key_pts = key_pts.type(torch.FloatTensor)
                images = images.type(torch.FloatTensor)

                if use_GPU:
                    images = images.to('cuda')
                    key_pts = key_pts.to('cuda')
                else:
                    images = images.to('cpu')
                    key_pts = key_pts.to('cpu')
                    
                output_pts = net(images)
                loss = criterion(output_pts, key_pts)
                # record validation loss
                valid_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        print('Epoch: {}/{},'.format(epoch+1, n_epochs),
            'Train_loss: {:.3f}'.format(train_loss),
            'Valid_loss: {:.3f}'.format(valid_loss))

        # if validation loss improves, save this model
        if valid_loss < valid_loss_best:
            torch.save(net.state_dict(), path_save)
            epochs_no_improve = 0
            valid_loss_best = valid_loss
            best_epoch = epoch+1

        else:
            epochs_no_improve +=1
            if epochs_no_improve >= patience:
                print("Early stopping!")
                break
                
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
    
    print("Best Epoch: {},".format(best_epoch),
                "Best Valid_loss: {:.3f}".format(valid_loss_best))

    print('Finished Training')

    # load the last checkpoint with the best model
    net.load_state_dict(torch.load(path_save))
    
    return net, avg_train_losses, avg_valid_losses

