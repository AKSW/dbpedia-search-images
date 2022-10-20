"""
Code to lead the embeddings saved, get the query image, and query on the dataset
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from pprint import pprint

from .query_embds import QueryEmbds
from img_models.generate_features import ImgFeatures


class RankedList():
    def __init__(self):
        self.imgemb = ImgFeatures()
        self.embeddings, self.img_paths, self.uris_list = self.get_lists()

    def load_dict(self, dict_):
        output = list()
        img_path = list()
        for key, value in dict_.items():
            output.append(np.expand_dims(value.cpu().detach().numpy(), axis=0))
            img_path.append(key)
        return output, img_path

    def get_lists(
        self,
        embed_path='/scratch/sid/dataset_embeddings',
        images_path='/scratch/sid/dataset',
        ):
        print('Loading the embeddings...')
        birds = np.load(
            os.path.join(embed_path, 'birds_embeddings.npy'),
            allow_pickle=True,
        ).item()
        weapons = np.load(
            os.path.join(embed_path, 'weapons_embeddings.npy'),
            allow_pickle=True,
        ).item()
        historic_places = np.load(
            os.path.join(embed_path, 'historic_places_embeddings.npy'),
            allow_pickle=True,
        ).item()
        politician = np.load(
            os.path.join(embed_path, 'politician_embeddings.npy'),
            allow_pickle=True,
        ).item()
        reptile = np.load(
            os.path.join(embed_path, 'reptile_embeddings.npy'),
            allow_pickle=True,
        ).item()

        print('Concatenating the embeddings...')
        embds = list()
        embds.extend(self.load_dict(birds)[0])
        embds.extend(self.load_dict(weapons)[0])
        embds.extend(self.load_dict(historic_places)[0])
        embds.extend(self.load_dict(politician)[0])
        embds.extend(self.load_dict(reptile)[0])

        print('Loading the URIs...')
        birds_uri = json.load(
            open(os.path.join(images_path, 'birds/info_dict.json'), 'r')
        )
        weapons_uri = json.load(
            open(os.path.join(images_path, 'weapons/info_dict.json'), 'r')
        )
        historic_places_uri = json.load(
            open(
                os.path.join(images_path, 'historic_places/info_dict.json'),
                'r',
            )
        )
        politician_uri = json.load(
            open(os.path.join(images_path, 'politician/info_dict.json'), 'r')
        )
        reptile_uri = json.load(
            open(os.path.join(images_path, 'reptile/info_dict.json'), 'r')
        )

        img_paths = list()
        uris_list = list()

        birds_list = self.load_dict(birds)[1]
        img_paths.extend(birds_list)
        uris_list.extend(self.load_uri(birds_uri, birds_list))

        weapons_list = self.load_dict(weapons)[1]
        img_paths.extend(weapons_list)
        uris_list.extend(self.load_uri(weapons_uri, weapons_list))

        historic_places_list = self.load_dict(historic_places)[1]
        img_paths.extend(historic_places_list)
        uris_list.extend(
            self.load_uri(historic_places_uri, historic_places_list)
        )

        politician_list = self.load_dict(politician)[1]
        img_paths.extend(politician_list)
        uris_list.extend(self.load_uri(politician_uri, politician_list))

        reptile_list = self.load_dict(reptile)[1]
        img_paths.extend(reptile_list)
        uris_list.extend(self.load_uri(reptile_uri, reptile_list))

        embeddings = np.concatenate(embds, axis=0)
        return embeddings, img_paths, uris_list

    def load_uri(self, uri_dict, images_path):
        uri_list = list()
        for query_uri in tqdm(images_path):
            for key, value in uri_dict.items():
                img_name = key.split('/')[-1]
                img_path_name = query_uri.split('/')[-1]
                if img_name == img_path_name:
                    uri_list.append(value['URI'])
                    break
        return uri_list

    def generate_results(self, query_index, num_img=10):
        print('Querying the embeddings...')
        queryembd = QueryEmbds(self.embeddings)
        _, ind = queryembd.query(self.embeddings[query_index])
        print('Generating the ranked list...')
        images_ranked_list = list()
        uris_ranked_list = list()
        for pred in ind[0]:
            images_ranked_list.append(self.img_paths[pred])
            uris_ranked_list.append(self.uris_list[pred])
        return images_ranked_list[:num_img], uris_ranked_list[:num_img]

    def generate_img_results(self, image_path, num_img=10):
        print('Querying the embeddings from the image loaded...')
        queryembd = QueryEmbds(self.embeddings)
        print('Generating embeddings for the query image...')
        query_img_embd = self.imgemb.get_features(image_path)
        _, ind = queryembd.query(query_img_embd.detach().cpu())
        print('Generating the ranked list...')
        images_ranked_list = list()
        uris_ranked_list = list()
        for pred in ind[0]:
            images_ranked_list.append(self.img_paths[pred])
            uris_ranked_list.append(self.uris_list[pred])
        return images_ranked_list[:num_img], uris_ranked_list[:num_img]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        default='/scratch/sid/dataset_embeddings',
        help='Path to the directory containing the embeddings',
    )
    parser.add_argument(
        '-n',
        default=10,
        type=int,
        help='Number of results to show',
    )
    parser.add_argument(
        '-p',
        required=True,
        help='Path to the image for quering the DBpedia articles',
    )
    args = parser.parse_args()
    ranked_list = RankedList()
    images_ranked_list, uris_ranked_list = ranked_list.generate_img_results(
        image_path=args.p,
        num_img=args.n,
    )
    print('List of DBpedia articles...')
    print(f'Rank\tURI')
    for i, uri in enumerate(uris_ranked_list):
        print(f'{i+1}\t{uri}')
