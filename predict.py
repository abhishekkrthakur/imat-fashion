from tqdm import tqdm

import numpy as np

import torch

import pandas as pd
from dataset import FashionDataset
from model import get_instance_segmentation_model
from torchvision import transforms
from PIL import Image
import itertools
from numba import jit
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
device = torch.device('cuda:0')

# Convert data to run-length encoding
def to_rle(bits):
   rle = []
   pos = 0
   for bit, group in itertools.groupby(bits):
       group_list = list(group)
       if bit:
           rle.extend([pos, sum(group_list)])
       pos += len(group_list)
   return rle

def refine_masks(masks, labels):
   # Compute the areas of each mask
   areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
   # Masks are ordered from smallest to largest
   mask_index = np.argsort(areas)
   # One reference mask is created to be incrementally populated
   union_mask = {k:np.zeros(masks.shape[:-1], dtype=bool) for k in np.unique(labels)}
   # Iterate from the smallest, so smallest ones are preserved
   for m in mask_index:
       label = labels[m]
       masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask[label]))
       union_mask[label] = np.logical_or(masks[:, :, m], union_mask[label])
   # Reorder masks
   refined = list()
   for m in range(masks.shape[-1]):
       mask = masks[:, :, m].ravel(order='F')
       rle = to_rle(mask)
       label = labels[m] - 1
       refined.append([masks[:, :, m], rle, label])
   return refined


@jit
def _to_rle(bits):
    rle = []
    pos = 0
    for bit, group in itertools.groupby(bits):
        group_list = list(group)
        if bit:
            rle.extend([pos, sum(group_list)])
        pos += len(group_list)
    return rle


@jit
def _refine_masks(masks):
    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)
    mask_index = np.argsort(areas)
    union_mask = np.zeros(masks.shape[:-1], dtype=bool)
    for m in mask_index:
        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))
        union_mask = np.logical_or(masks[:, :, m], union_mask)
    return masks


def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return rle.flatten()


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(mask, scores, labels):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] #s+ 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append((rle, labels[o]))
    return lines



num_classes = 46 + 1


dataset_test = FashionDataset("../input/test/", "../input/sample_submission.csv", 1024, 1024,
                               folds=[], transforms=None)

sample_df = pd.read_csv("../input/sample_submission.csv")


model_ft = get_instance_segmentation_model(num_classes)
model_ft.load_state_dict(torch.load("model.bin"))
model_ft = model_ft.to(device)

for param in model_ft.parameters():
    param.requires_grad = False

model_ft.eval()


sub_list = []
missing_count = 0
submission = []
ctr = 0

tk0 = tqdm(range(3200))
tt = transforms.ToTensor()
for i in tk0:
    #if i > 10:
    #    break
    #if i < 100:
    img = dataset_test[i]
    img = tt(img)
    result = model_ft([img.to(device)])[0]
    # Encode image to RLE. Returns a string of multiple lines
    masks = np.zeros((512, 512, len(result["masks"])))
    for j, m in enumerate(result["masks"]):
        #res = result["masks"][j].permute(1, 2, 0).cpu().numpy()#.astype("uint8")
        res = transforms.ToPILImage()(result["masks"][j].permute(1, 2, 0).cpu().numpy())
        #res = Image.fromarray(res)
        res = np.asarray(res.resize((512, 512), resample=Image.BILINEAR))
        #print(res.shape)
        masks[:, :, j] = (res[:, :] * 255. > 127).astype(np.uint8)

    lbls = result['labels'].cpu().numpy()
    scores = result['scores'].cpu().numpy()

    best_idx = 0
    for scr in scores:
      if scr > 0.8:
        best_idx += 1

    if best_idx == 0:
      sub_list.append([sample_df.loc[i, 'ImageId'], '1 1', 23])
      missing_count += 1
      continue

    if masks.shape[-1] > 0:
        #lll = mask_to_rle(masks[:, :, :4], scores[:4], lbls[:4])
        masks = refine_masks(masks[:, :, :best_idx], lbls[:best_idx])
        for m, rle, label in masks:
            sub_list.append([sample_df.loc[i, 'ImageId'], ' '.join(list(map(str, list(rle)))), label])
        #else:
        #    sub_list.append([sample_df.loc[i, 'ImageId'], '1 1', 23])
        #    missing_count += 1

    else:
        # The system does not allow missing ids, this is an easy way to fill them
        sub_list.append([sample_df.loc[i, 'ImageId'], '1 1', 23])
        missing_count += 1
    #else:
    #    sub_list.append([sample_df.loc[i, 'ImageId'], '1 1', 23])
    #    missing_count += 1

submission_df = pd.DataFrame(sub_list, columns=sample_df.columns.values)
print("Total image results: ", submission_df['ImageId'].nunique())
print("Missing Images: ", missing_count)
submission_df = submission_df[submission_df.EncodedPixels.notnull()]
for row in range(len(submission_df)):
   line = submission_df.iloc[row,:]
   submission_df.iloc[row, 1] = line['EncodedPixels'].replace('.0','')
submission_df.head()
submission_df.to_csv("submission.csv", index=False)
