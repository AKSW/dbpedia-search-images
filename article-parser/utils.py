
import argparse
from pprint import pprint
from rdflib import sparqlEndpoint
import requests

def save_image(input_dict, filename):
    """
    Method to save an image given the image link
    """
    image_link = input_dict['image']['value']
    file = open(filename, 'wb')
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:71.0) Gecko/20100101 Firefox/71.0'}
    file.write(requests.get(image_link, headers=headers).content)
    file.close()
    return None
