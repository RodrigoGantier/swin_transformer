import os
import matplotlib.pyplot as plt
import numpy as np
import PIL
import cv2
import argparse

import torchvision.transforms as T
from models.swin_transfoemer_ import swin_tiny_patch4_window7_224


def load_model():
   
    model = swin_tiny_patch4_window7_224(pretrained=True)
    model.eval()
    model.to('cuda:0')

    return model


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def main(args):

    imagenet_labels = dict(enumerate(open(args.labels)))
    imgs_list = [args.data_path + '/' + im.name for im in os.scandir(args.data_path) if im.name.endswith('jpg')] 
    imgs_list.sort()

    IMG_SIZE = (224, 224)
    NORMALIZE_MEAN = (0.485, 0.456, 0.406)
    NORMALIZE_STD = (0.229, 0.224, 0.225)
    transforms = [
                T.Resize(IMG_SIZE),
                T.ToTensor(),
                T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
                ]
    transforms = T.Compose(transforms)

    model = load_model()

    for img_name in imgs_list:

        img = PIL.Image.open(img_name).convert('RGB')
        img_tensor = transforms(img).unsqueeze(0).to(args.device)

        output = model(img_tensor)

        _, pred = output.topk(5, 1, True, True)

        print("-"*20)
        print("Inference Result: \n")
        for i in range(5):
            print(imagenet_labels[int(pred[0, i])])
    print('finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gradient prediction')
    parser.add_argument('--model_n', default=9, help='9 == swin_tiny')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--labels', default='./ilsvrc2012_wordnet_lemmas.txt')
    parser.add_argument('--ckpt', default='./swin_tiny_patch4_window7_224.pth')
    parser.add_argument('--data_path', default='./images')

    args = parser.parse_args()
    main(args)

    os.getcwd()
