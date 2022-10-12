
import os
import json
import argparse
from tqdm import tqdm

from utils import save_image
from rdflib import sparqlEndpoint


def save_batch(output, dir_path):
    count = 0
    save_json = dict()
    error_json = dict()

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for input_dict in tqdm(output):
        saved = True
        count += 1
        image_path = os.path.join(dir_path, f'{count}.jpg')
        print (image_path)
        try:
            save_image(input_dict, image_path)
        except:
            os.system(f'rm {image_path}')
            saved = False
        if saved:
            # save json
            save_json[image_path] = {
                'URI': input_dict['e']['value'],
                'Image-URI': input_dict['image']['value'],
            }
        else:
            error_json[image_path] = {
                'URI': input_dict['e']['value'],
                'Image-URI': input_dict['image']['value'],
            }
    json.dump(save_json, open(os.path.join(dir_path, 'info_dict.json'), 'w'))
    json.dump(error_json, open(os.path.join(dir_path, 'error_dict.json'), 'w'))
    print(f'{len(save_json)} images saved to {dir_path}...')
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--endpoint',
        '-e',
        default="http://dbpedia.org/sparql",
        help='Link to the endpoint',
    )
    parser.add_argument(
        '--graph',
        '-g',
        default="http://dbpedia.org",
        help='Link to the graph',
    )
    parser.add_argument(
        '--format',
        '-f',
        default='JSON',
        help='Format to get the output in',
    )
    parser.add_argument(
        '--subjs_cxml',
        '-s',
        default='121',
        help='TODO',
    )
    parser.add_argument(
        '--hrefs_cxml',
        '-c',
        default='',
        help="TODO",
    )
    parser.add_argument(
        '--timeout',
        '-t',
        default='600',
        help='Amount of timeout seconds',
    )
    parser.add_argument(
        '--debug',
        '-d',
        default='on',
        help='Debug?',
    )
    parser.add_argument(
        '--query',
        '-q',
        default='''select ?e ?image where { ?e rdf:type dbo:Bird  . ?e dbo:thumbnail ?image .}''',
        help='query string',
    )
    parser.add_argument(
        '--save_dir',
        '-sd',
        default='''img_dataset/''',
        help='Path to the directory to save the images into',
    )
    args=parser.parse_args()
    print("Generating image's links...")

    output = sparqlEndpoint(
        query=args.query,
        endpoint=args.endpoint,
        graph=args.graph,
        format=args.format,
        cxml_subjs=args.subjs_cxml,
        cxml_hrefs=args.hrefs_cxml,
        timeout=args.timeout,
        debug=args.debug,
    )
    print("Saving images...")
    save_batch(output, args.save_dir)
