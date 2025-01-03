import os
import argparse

import torch
from torch.utils.data import DataLoader, SequentialSampler

from utils.datasets.flickr30k_captions import Flickr30kCaptions
from utils.datasets.mscoco_captions import MSCOCOCaptions
from utils.datasets.conceptual_captions import ConceptualCaptions

from utils.loss import compute_clip_loss

from models.custom_vista import CustomVISTA


def create_model(name, local_path=None):
    if name == 'zero_shot_VISTA':
        return CustomVISTA()
    elif name == 'VISTA':
        return CustomVISTA(text_encoder_from_local=True, image_encoder_from_local=True, local_pretrained_weights_path=local_path)
    else:
        raise ValueError('The selected model is not implemented yet')
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help='name of the model', dest='MODEL')
    parser.add_argument("--local-weights", type=str, required=False,
                        help='the local path in which the pretrained weights are saved', dest='PATH')
    parser.add_argument("--dataset", type=str, required=True, help='name of the dataset', dest='DATASET')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size', dest='BATCH_SIZE')

    args = parser.parse_args()

    return args


args = parse_args()
MODEL = args.MODEL
PATH = args.PATH
DATASET = args.DATASET
BATCH_SIZE = args.BATCH_SIZE
NUM_WORKERS = 2

# for debugging
# os.environ['TORCH_HOME']='/nfs/home/morovatian/.cache/torch'
# MODEL = 'VISTA'
# PATH = 'weights/CLIP/ViT32_LiUi_clip_loss_mscoco.pth'
# LOSS = 'clip'
# DATASET = 'mscoco'
# BATCH_SIZE = 2
# CPI = 5 # captions per image
# SAVE_EMBDS = False
# NUM_WORKERS = 2

assert MODEL in ['zero_shot_VISTA', 'VISTA']
    
assert DATASET in ['mscoco', 'flickr30k', 'conceptualCaptions']

model = create_model(MODEL, PATH)

if DATASET == 'mscoco':
    test_dataset = MSCOCOCaptions(
        root='../modality-invariance-VLMs/data/images/mscoco/val2017/',
        annotations_file='../modality-invariance-VLMs/data/annotations/mscoco/val2017_captions.json',
        image_transform=model.transform,
        caption_transform=model.text_tokenizer,
        classified_ann_file='../modality-invariance-VLMs/data/classified/mscoco_val2017.csv')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
elif DATASET == 'flickr30k':
    test_dataset = Flickr30kCaptions(
        root='../modality-invariance-VLMs/data/images/flickr30k/',
        annotations_file='../modality-invariance-VLMs/data/annotations/flickr30k/test.token',
        image_transform=model.transform,
        caption_transform=model.text_tokenizer)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

elif DATASET == 'conceptualCaptions':
    test_dataset = ConceptualCaptions(
        root='../modality-invariance-VLMs/data/images/conceptualCaptions/',
        annotations_file='../modality-invariance-VLMs/data/annotations/conceptualCaptions/test.csv',
        image_transform=model.transform,
        caption_transform=model.text_tokenizer)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

else:
    raise ValueError('The selected dataset is not supported')

loss_function = compute_clip_loss

retrieval_inputs = model.encode_for_retrieval(test_dataloader, loss_function)

result_dir = f'../modality-invariance-VLMs/results/embeddings/visualization/{MODEL}/{DATASET}/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

torch.save(retrieval_inputs['image_embeddings'], result_dir+'image.pt')
torch.save(retrieval_inputs['text_embeddings'], result_dir+'text.pt')