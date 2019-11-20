import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

def show_keypoints(image, face_vertices, net, use_GPU, fig_shape, pad):
    """
    Plot keypoints on faces and return x and y of keypoints
    """
    
    # make a copy of the original image to plot on
    image_copy = image.copy()
    keypoints_x = []
    keypoints_y = []
    keypoints = []

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

        keypoints_x.append(output_pts[:, 0])
        keypoints_y.append(output_pts[:, 1])
        keypoints.append(output_pts)

    fig = plt.figure(figsize=tuple(fig_shape))
    plt.scatter(keypoints_x, keypoints_y, s=20, marker='.', c='m')
    plt.imshow(image_copy)
    
    plt.axis("off")
    plt.show()

    return keypoints


def show_shades(image, face_vertices, shades, keypoints, fig_shape):
    """
    Draw shades on each face of the image
    """

    # make a copy of the original image to plot on
    image_copy = image.copy()
    
    for i in range(len(face_vertices)):

        keypoints_i = keypoints[i]

        # add shades in this section
        shades_x = int(keypoints_i[17, 0])
        shades_y = int(keypoints_i[17, 1])
        shades_h = int(abs(keypoints_i[27,1] - keypoints_i[34,1]))
        shades_w = int(abs(keypoints_i[17,0] - keypoints_i[26,0]))
        
        new_shades = cv2.resize(shades, (shades_w, shades_h), interpolation = cv2.INTER_CUBIC)

        # get region of interest on the face to change
        roi_color = image_copy[shades_y:shades_y+shades_h,shades_x:shades_x+shades_w]
        
        ind = np.argwhere(new_shades[:,:,3] > 0)
        
        for i in range(3):
            roi_color[ind[:,0],ind[:,1],i] = new_shades[ind[:,0],ind[:,1],i]
        
        image_copy[shades_y:shades_y+shades_h,shades_x:shades_x+shades_w] = roi_color

    fig = plt.figure(figsize=tuple(fig_shape))
    plt.imshow(image_copy)

    plt.axis("off")
    plt.show()

