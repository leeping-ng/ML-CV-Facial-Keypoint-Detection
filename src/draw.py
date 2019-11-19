import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

def show_keypoints(image, face_vertices, net, use_GPU, fig_width, pad):
    
    # make a copy of the original image to plot on
    image_copy = image.copy()

    for i, (x,y,w,h) in enumerate(face_vertices):
        # Select the region of interest that is the face in the image
        roi = image_copy[y-pad : y+h+pad, x-pad : x+w+pad]

        # Convert the face region from RGB to grayscale
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
        roi = roi/255.0

        # Rescale detected face to expected square size (224, 224)
        roi = cv2.resize(roi, (224, 224))

        # Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
        # if image has no grayscale color channel, add one
        if(len(roi.shape) == 2):
            # add that third color dim
            roi = roi.reshape(roi.shape[0], roi.shape[1], 1, 1)

        roi_torch = roi.transpose((3, 2, 0, 1))
        roi_torch = torch.from_numpy(roi_torch)

        # convert images to FloatTensors
        roi_torch = roi_torch.type(torch.FloatTensor)

        if use_GPU:
            # move model to GPU
            net.to("cuda")
            roi_torch = roi_torch.to("cuda")
        else:
            net.to("cpu")
            roi_torch = roi_torch.to("cpu")

        # make prediction
        output_pts = net(roi_torch)

        if use_GPU:
            # move to CPU for subsequent steps
            output_pts=output_pts.to("cpu")

        output_pts = output_pts.data.numpy()

        #reshape to 68 x 2 pts
        output_pts = output_pts[0].reshape((68, 2))

        #undo normalization
        output_pts = output_pts*50.0+100

        #correct for scaling and padding to original image format
        output_pts[:, 0] = output_pts[:, 0]*(w+2*pad)/224 + x-pad
        output_pts[:, 1] = output_pts[:, 1]*(h+2*pad)/224 + y-pad

        plt.scatter(output_pts[:, 0], output_pts[:, 1], s=20, marker='.', c='m')

    fig = plt.figure(figsize=(fig_width, fig_width/1.5))
    plt.imshow(image)
    
    plt.axis("off")
    plt.show()

