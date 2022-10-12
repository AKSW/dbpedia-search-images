import os
import argparse
import numpy as np
from tqdm import tqdm

from generate_features import ImgFeatures

parser = argparse.ArgumentParser()
parser.add_argument(
    '-p',
    required=True,
    help='Path to the image to load',
)
parser.add_argument(
    '-m',
    default='resnet50',
    help='Name of the model to load for extracting the features',
)
args = parser.parse_args()

img_feats = ImgFeatures(model_name=args.m)

if os.path.isdir(args.p):
    images_path = os.listdir(args.p)
    embeds_dict = dict()
    print(f'Processind directory with {len(images_path)} images')
    for path in tqdm(images_path, desc="Generating Embeddings"):
        image_path = os.path.join(args.p, path)
        if '.json' not in image_path:
            # Ignoring the JSON files saved.
            feats = img_feats.get_features(image_path)
            if feats!=None:
                embeds_dict[image_path] = feats.detach()
    embeddings_path = os.path.join(args.p, 'embeddings.npy')
    np.save(embeddings_path, embeds_dict)
    print(f'Saved embeddings to {embeddings_path}...')
else:
    feats = img_feats.get_features(args.p)
    print(f'Features of shape {feats.shape} generated for {os.path.basename(args.p)}!')
