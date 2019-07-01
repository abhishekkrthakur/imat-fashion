import collections
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageFile
import pandas as pd
from tqdm import tqdm
from numba import jit

ImageFile.LOAD_TRUNCATED_IMAGES = True


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array
    shape: (height,width) of array to return
    Returns numpy array according to the shape, 1 - mask, 0 - background
    '''
    shape = (shape[1], shape[0])
    s = mask_rle.split()
    # gets starts & lengths 1d arrays
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    # gets ends 1d array
    ends = starts + lengths
    # creates blank mask image 1d array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    # sets mark pixles
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # reshape as a 2d mask image
    return img.reshape(shape).T  # Needed to align to RLE direction



class FashionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, df_path, height, width, folds=[], transforms=None):
        self.transforms = transforms
        self.image_dir = image_dir
        self.df = pd.read_csv(df_path)
        self.height = height
        self.width = width
        self.folds = folds
        self.image_info = collections.defaultdict(dict)
        if len(self.folds) > 0:
            self.df['CategoryId'] = self.df.ClassId.apply(lambda x: str(x).split("_")[0])
            self.df = self.df[self.df.kfold.isin(folds)].reset_index(drop=True)
            temp_df = self.df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x)).reset_index()
            size_df = self.df.groupby('ImageId')['Height', 'Width'].mean().reset_index()
            temp_df = temp_df.merge(size_df, on='ImageId', how='left')
            for index, row in tqdm(temp_df.iterrows(), total=len(temp_df)):
                image_id = row['ImageId']
                image_path = os.path.join(self.image_dir, image_id)
                self.image_info[index]["image_id"] = image_id
                self.image_info[index]["image_path"] = image_path
                self.image_info[index]["width"] = self.width
                self.image_info[index]["height"] = self.height
                self.image_info[index]["labels"] = row["CategoryId"]
                self.image_info[index]["orig_height"] = row["Height"]
                self.image_info[index]["orig_width"] = row["Width"]
                self.image_info[index]["annotations"] = row["EncodedPixels"]
        else:
            for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
                image_id = row['ImageId']
                image_path = os.path.join(self.image_dir, image_id)
                self.image_info[index]["image_id"] = image_id
                self.image_info[index]["image_path"] = image_path
                self.image_info[index]["width"] = self.width
                self.image_info[index]["height"] = self.height

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        if len(self.folds) == 0:
            return img

        info = self.image_info[idx]
        mask = np.zeros((len(info['annotations']), self.width, self.height), dtype=np.uint8)
        labels = []
        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):
            #sub_mask = np.full(info['orig_height'] * info['orig_width'], 0, dtype=np.uint8)
            #annotation = np.array([int(x) for x in annotation.split(' ')])

            #for i, start_pixel in enumerate(annotation[::2]):
            #    sub_mask[start_pixel: start_pixel + annotation[2 * i + 1]] = 1

            #sub_mask = sub_mask.reshape((info['orig_height'], info['orig_width']), order='F')
            sub_mask = rle_decode(annotation, (info['orig_height'], info['orig_width']))
            sub_mask = Image.fromarray(sub_mask)
            sub_mask = sub_mask.resize((self.width, self.height), resample=Image.BILINEAR)
            mask[m, :, :] = sub_mask
            labels.append(int(label) + 1)

        # get bounding box coordinates for each mask
        num_objs = len(labels)
        boxes = []
        new_labels = []
        new_masks = []

        for i in range(num_objs):
            try:
                pos = np.where(mask[i, :, :])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if abs(xmax - xmin) >= 20 and abs(ymax - ymin) >= 20:
                    boxes.append([xmin, ymin, xmax, ymax])
                    new_labels.append(labels[i])
                    new_masks.append(mask[i, :, :])
            except ValueError:
                continue

        if len(new_labels) == 0:
            boxes.append([0, 0, 20, 20])
            new_labels.append(0)
            new_masks.append(mask[0, :, :])

        nmx = np.zeros((len(new_masks), self.width, self.height), dtype=np.uint8)
        for i, n in enumerate(new_masks):
            nmx[i, :, :] = n

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(new_labels, dtype=torch.int64)
        masks = torch.as_tensor(nmx, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_info)
