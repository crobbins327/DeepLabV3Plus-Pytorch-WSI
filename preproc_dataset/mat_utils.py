import cv2
import numpy as np
# import scipy.io as sio
import matplotlib.pyplot as plt


def nuclei_dict_from_inst(labels, mask, metadata=None):

    nuclei_id = []
    classes = []
    bboxs = []
    centroids = []

    # Iterate over each unique label identified, skipping label 0 (background)
    for label_num in np.unique(labels)[1:]:
        # Create a mask for the current nucleus
        nucleus_mask = (labels == label_num)

        # Find the class based on original image's values, ignoring borders (outline might affect mode)
        nucleus_class = np.bincount(mask[nucleus_mask].flat).argmax()

        # Find bounding box coordinates
        y, x = np.where(nucleus_mask)
        # For each row in the array, the ordering of coordinates is (y1, y2, x1, x2). 
        bbox = [min(y), max(y), min(x), max(x)]
        
        # Find centroid
        centroid = [int(np.mean(x)), int(np.mean(y))]
        
        # Store the results
        nuclei_id.append(label_num)
        classes.append(nucleus_class)
        bboxs.append(bbox)
        centroids.append(centroid)

    # Convert lists to numpy arrays for consistency with .mat format expectations
    nuclei_id = np.array(nuclei_id)[:, np.newaxis]
    classes = np.array(classes)[:, np.newaxis]
    bboxs = np.array(bboxs)
    centroids = np.array(centroids)

    class_map = remap_inst_to_class(labels, nuclei_id, classes)

    # plt.figure(2)
    # plt.imshow(class_map)
    # plt.figure(3)
    # plt.imshow(mask)

    data_dict = {
        'inst_map': labels,
        'class_map': class_map,
        'id': nuclei_id,
        'class': classes,
        'bbox': bboxs,
        'centroid': centroids
    }
    
    if metadata is not None:
        for key in metadata:
            if key not in data_dict:
                data_dict[key] = metadata[key]
            else:
                print(f"Metadata key {key} already exists in data_dict, skipping.")
    
    # debugging code
    plt.figure(1)
    # plt.imshow(labels)
    plt.imshow(class_map)
    # plot each centroid and bbox
    for i in range(len(centroids)):
        bbox = bboxs[i]
        centroid = centroids[i]
        plt.plot(centroid[0], centroid[1], 'ro')
        plt.plot([bbox[2], bbox[2], bbox[3], bbox[3], bbox[2]], [bbox[0], bbox[1], bbox[1], bbox[0], bbox[0]], 'r-')

    return data_dict

def nuclei_dict_from_mask(mask, metadata=None, border_val=255):
    """
    Convert a mask to a nuclei mat file.
    """
    mask[mask == border_val] = 0
    ret, labels = cv2.connectedComponents(mask)
    nuclei_id = []
    classes = []
    bboxs = []
    centroids = []

    # Iterate over each unique label identified, skipping label 0 (background)
    for label_num in np.unique(labels)[1:]:
        # Create a mask for the current nucleus
        nucleus_mask = (labels == label_num)

        # Find the class based on original image's values, ignoring borders (outline might affect mode)
        nucleus_class = np.bincount(mask[nucleus_mask].flat).argmax()

        # Find bounding box coordinates
        y, x = np.where(nucleus_mask)
        # For each row in the array, the ordering of coordinates is (y1, y2, x1, x2). 
        bbox = [min(y), max(y), min(x), max(x)]
        
        # Find centroid
        centroid = [int(np.mean(x)), int(np.mean(y))]
        
        # Store the results
        nuclei_id.append(label_num)
        classes.append(nucleus_class)
        bboxs.append(bbox)
        centroids.append(centroid)

    # Convert lists to numpy arrays for consistency with .mat format expectations
    nuclei_id = np.array(nuclei_id)[:, np.newaxis]
    classes = np.array(classes)[:, np.newaxis]
    bboxs = np.array(bboxs)
    centroids = np.array(centroids)

    class_map = remap_inst_to_class(labels, nuclei_id, classes)

    # plt.figure(2)
    # plt.imshow(class_map)
    # plt.figure(3)
    # plt.imshow(mask)

    data_dict = {
        'inst_map': labels,
        'class_map': class_map,
        'id': nuclei_id,
        'class': classes,
        'bbox': bboxs,
        'centroid': centroids
    }
    
    if metadata is not None:
        for key in metadata:
            if key not in data_dict:
                data_dict[key] = metadata[key]
            else:
                print(f"Metadata key {key} already exists in data_dict, skipping.")
    
    # debugging code
    # plt.figure(1)
    # plt.imshow(labels)
    # # plot each centroid and bbox
    # for i in range(len(centroids)):
    #     bbox = bboxs[i]
    #     centroid = centroids[i]
    #     plt.plot(centroid[0], centroid[1], 'ro')
    #     plt.plot([bbox[2], bbox[2], bbox[3], bbox[3], bbox[2]], [bbox[0], bbox[1], bbox[1], bbox[0], bbox[0]], 'r-')

    return data_dict



def remap_inst_to_class(inst_map, nuclei_id, classes):
    """
    Remap the instance map to a class map.
    """

    class_map = np.zeros_like(inst_map)
    for i in range(len(nuclei_id)):
        class_map[inst_map == nuclei_id[i]] = classes[i]

    return class_map