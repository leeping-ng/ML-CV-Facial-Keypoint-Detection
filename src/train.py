import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train(n_epochs, net, train_loader, test_loader, use_GPU):

    # define loss function and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    # prepare the net for training
    net.train()
    
    train_loss_over_time = []
    valid_loss_over_time = []
    
    if use_GPU:
        # move model to GPU
        print("Using GPU for training...")
        net.to("cuda")
    else:
        print("Using CPU only for training...")
        net.to("cpu")

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
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

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 27 == 26:    # print every 27 batches
                
                # set model to eval mode to avoid applying dropout
                net.eval()
                
                with torch.no_grad():
                    valid_loss = validate_net(net, test_loader, criterion, use_GPU)
                
                training_loss = running_loss/27
                
                print('Epoch: {}/{},'.format(epoch+1, n_epochs),
                      'Batch: {},'.format(batch_i+1), 
                      'Training Loss: {:.3f}'.format(training_loss),
                      'Validation Loss: {:.3f}'.format(valid_loss/len(test_loader)))
                
                train_loss_over_time.append(training_loss)
                valid_loss_over_time.append(valid_loss/len(test_loader))
                
                running_loss = 0.0
                
                # set model back to training mode
                net.train()

    print('Finished Training')
    
    return net, train_loss_over_time, valid_loss_over_time


def validate_net(net, test_loader, criterion, use_GPU):

    if use_GPU:
        # move model to GPU
        net.to("cuda")
    else:
        net.to("cpu")
    
    test_loss = 0
    
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
        
        test_loss += loss.item()
        
    return test_loss