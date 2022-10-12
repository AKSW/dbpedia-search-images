"""
This file contains code for generating features for an input image
"""

import os
import torch
from PIL import Image
from torchvision import transforms, models


class ImgFeatures():
    def __init__(self, model_name='resnet50'):
        # assert os.path.isfile(img_path)
        # self.img_path = img_path
        self.model_name = model_name
        self.data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using {self.device}!')
        self.get_model()

    def load_img(self, image_path):
        """
        Given image's path, load it
        """
        image = Image.open(image_path).convert('RGB')
        return image

    def preprocess_img(self, image_path):
        """
        Given the image, convert it to torch tensor, and make it model's
        compatible
        """
        image_pil = self.load_img(image_path)
        image = self.data_transforms(image_pil)
        return image.unsqueeze(dim=0)

    def get_model(self):
        """
        Given the model name, load the pretrained model
        """
        if self.model_name == 'resnet50':
            print('Loading pre-trained ResNet-50')
            model = models.resnet50(pretrained=True)
        elif self.model_name == 'resnet18':
            print('Loading pre-trained ResNet-18')
            model = models.resnet18(pretrained=True)
        elif self.model_name == 'resnet101':
            print('Loading pre-trained ResNet-101')
            model = models.resnet101(pretrained=True)
        elif self.model_name == 'resnet152':
            print('Loading pre-trained ResNet-152')
            model = models.resnet152(pretrained=True)
        else:
            raise NotImplementedError()
        self.model = torch.nn.Sequential(*list(model.children())[:-1])
        self.model.eval()
        return self.model.to(self.device)

    def get_features(self, image_path):
        """
        Given the image tensor, pass it through a pre-trained model and get the
        features
        """
        if os.path.isfile(image_path) and self.load_img(image_path):
            img_tensor = self.preprocess_img(image_path).to(self.device)
            return self.model(img_tensor).squeeze()
        return None


if __name__ == '__main__':
    input_img_path = "/google-images-download/images/beer_labels/6.jockeyclub_0001.jpg"
    feats_class = ImgFeatures()
    feats = feats_class.get_features(image_path=input_img_path)
