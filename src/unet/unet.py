import xml.etree.ElementTree as ET
import os 

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

from model import get_unet, train, predict
from vagabond_dataset import VagabondDatasetPanel

def extract_polygon_from_xml(label_path: 'String') -> list: 
    polygon_list = []

    tree = ET.parse(label_path)
    root = tree.getroot()
    for polygon in root.iter('polygon'): 
        coordinate_list = []

        for coordinate in polygon: 
            coordinate_list.append(int(np.round(float(coordinate.text))))
            
        polygon_list.append(coordinate_list)

    return polygon_list
        

def draw_one_polygon_to_mask(mask, polygon, color) -> np.array:   
    isClosed = True
    thickness = 2
    
    mask = cv2.polylines(mask, [polygon], isClosed, color, thickness)
    cv2.fillPoly(mask, [polygon], color)
    return mask


def create_mask(shape: tuple, label_path) -> np.array:
    mask = np.full(shape[0:2], 0, dtype=np.int32)
    polygon_list = extract_polygon_from_xml(label_path)

    for index, polygon in enumerate(polygon_list): 
        polygon = np.array(polygon, np.int32)
        polygon = np.reshape(polygon, (-1, 2))
        mask = draw_one_polygon_to_mask(mask, polygon, color=index + 1)

    return mask


def get_shape(label_path: 'String') -> tuple: 
    tree = ET.parse(label_path)
    root = tree.getroot()

    for width in root.iter('width'):
        width = int(width.text)

    for height in root.iter('height'):
        height = int(height.text)

    for channel in root.iter('depth'):
        channel = int(channel.text)
    
    return (height, width, channel)


def save_mask(label_dir, mask_dir) -> None: 
    label_list = os.listdir(label_dir)

    for label_name in label_list: 
        label_path = os.path.join(label_dir, label_name)
        mask_shape = get_shape(label_path)
        mask = create_mask(mask_shape, label_path)
        plt.imsave(mask_dir + label_name[:-3] + "jpg", mask)        


def main() -> None: 
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(f'Using {device}')

    img_dir = '/home/dangnh/b3/group_project/data/vagabond/images'
    label_dir = '/home/dangnh/b3/group_project/data/vagabond/label_panel_xml/'
    mask_dir = '/home/dangnh/b3/group_project/data/vagabond/mask_panel/'

    save_mask(label_dir, mask_dir)

    # transform = v2.Compose([
    #     v2.Resize([640, 640])
    # ])
    #
    #
    # dataset = VagabondDatasetPanel(img_dir, mask_dir,
    #                                transform=transform,
    #                                mask_transform=transform)
    #
    # generator = torch.Generator().manual_seed(42)
    # train_dataset, test_dataset =  random_split(dataset, [0.8, 0.2])
    #
    # batch_size = 2
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #
    # model = get_unet(in_channels=3, out_channels=1).to(device)
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)

    # img, mask = dataset[10]
    # print(f'Img shape: {img.size()}, Mask shape: {mask.size()}')
    # plt.imshow(img.permute(1,2,0))
    # plt.show()
    # plt.imshow(mask.permute(1,2,0), 'gray')
    # plt.show()


    # print(model)
    # model = train(model, train_dataloader, train_dataloader, 
                  # loss_fn, optimizer, epochs=1, device=device)

    # res = predict(model, img.unsqueeze(0), device=device)
    # print(res, res.size())


if __name__ == '__main__':
    main()
