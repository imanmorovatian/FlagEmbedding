import os
import argparse
import csv

from torch.utils.data import DataLoader, SequentialSampler

from utils.datasets.flickr30k_captions import Flickr30kCaptions
from utils.datasets.mscoco_captions import MSCOCOCaptions
from utils.datasets.conceptual_captions import ConceptualCaptions

from utils.loss import compute_clip_loss, compute_CUA_loss, compute_CUAXU_loss

from models.custom_vista import CustomVISTA

from utils.metrics.retrieval import CrossModalRetrieval
from utils.metrics.metrics import CMD, CD


def create_model(name, local_path=None):
    if name == 'zero_shot_VISTA':
        return CustomVISTA()
    elif name == 'VISTA':
        return CustomVISTA(text_encoder_from_local=True, mage_encoder_from_local=True, local_pretrained_weights_path=local_path)
    else:
        raise ValueError('The selected model is not implemented yet')
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help='name of the model', dest='MODEL')
    parser.add_argument("--local-weights", type=str, required=False,
                        help='the local path in which the pretrained weights are saved', dest='PATH')
    parser.add_argument("--loss", type=str, required=True, help='name of the loss function', dest='LOSS')
    parser.add_argument("--dataset", type=str, required=True, help='name of the dataset', dest='DATASET')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size', dest='BATCH_SIZE')
    parser.add_argument('--captions_per_image', type=int, required=True, help='number of captions per image', dest='CPI')

    args = parser.parse_args()

    return args


args = parse_args()
MODEL = args.MODEL
PATH = args.PATH
LOSS = args.LOSS
DATASET = args.DATASET
BATCH_SIZE = args.BATCH_SIZE
CPI = args.CPI # captions per image
NUM_WORKERS = 2

#  for debugging
# MODEL = 'zero_shot_VISTA'
# PATH = 'weights/CLIP/ViT32_LiUi_clip_loss_mscoco.pth'
# LOSS = 'clip'
# DATASET = 'mscoco'
# BATCH_SIZE = 128
# CPI = 5 # captions per image
# NUM_WORKERS = 2

assert MODEL in ['zero_shot_VISTA', 'VISTA']

if MODEL == 'VISTA':
    if PATH is None:
        raise ValueError('''if you are not going to use the pre-trained weights from the Internet
                         ,then you must provide local pretrained weights''')
    
assert DATASET in ['mscoco', 'flickr30k', 'conceptualCaptions']

assert LOSS in ['clip', 'cua', 'cuaxu']

model = create_model(MODEL, PATH)

if DATASET == 'mscoco':
    test_dataset = MSCOCOCaptions(root='../modality-invariance-VLMs/data/images/mscoco/val2017/',
						annotations_file='../modality-invariance-VLMs/data/annotations/mscoco/val2017_captions.json',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer,
                        no_cap_per_img=CPI)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
elif DATASET == 'flickr30k':
    test_dataset = Flickr30kCaptions(root='../modality-invariance-VLMs/data/images/flickr30k/',
                        annotations_file='../modality-invariance-VLMs/data/annotations/flickr30k/test.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer,
                        no_cap_per_img=CPI)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

elif DATASET == 'conceptualCaptions':
    test_dataset = ConceptualCaptions(root='../modality-invariance-VLMs/data/images/conceptualCaptions/',
                        annotations_file='../modality-invariance-VLMs/data/annotations/conceptualCaptions/test.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer,
                        no_cap_per_img=CPI)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

else:
    raise ValueError('The selected dataset is not supported')


if LOSS == 'clip':
    loss_function = compute_clip_loss
elif LOSS == 'cua':
    loss_function = compute_CUA_loss
elif LOSS == 'cuaxu':
    loss_function = compute_CUAXU_loss
else:
    raise ValueError('The selected loss is not supported')

retrieval_inputs = model.encode_for_retrieval(test_dataloader, loss_function)

metrics = {}

retrieval_obj = CrossModalRetrieval(image_encodings=retrieval_inputs['image_embeddings'],
                                    text_encodings=retrieval_inputs['text_embeddings'],
                                    text_to_image_map=retrieval_inputs['text_to_image_mapping'],
                                    image_to_text_map=retrieval_inputs['image_to_text_mapping'],
                                    cpi=CPI,
                                    search_space='unimodal',
                                    k_vals=[1,5,10])
metrics['retrieval_unimodal'] = retrieval_obj.compute()


retrieval_obj = CrossModalRetrieval(image_encodings=retrieval_inputs['image_embeddings'],
                                    text_encodings=retrieval_inputs['text_embeddings'],
                                    text_to_image_map=retrieval_inputs['text_to_image_mapping'],
                                    image_to_text_map=retrieval_inputs['image_to_text_mapping'],
                                    cpi=CPI,
                                    search_space='multimodal',
                                    k_vals=[1,5,10])
metrics['retrieval_multimodal'] = retrieval_obj.compute()

cmd = CMD()
metrics['cmd_img_txt'] = round(
    cmd(retrieval_inputs['image_embeddings'], retrieval_inputs['text_embeddings'][::5]).item(),
    2)

cd = CD()
metrics['cd_img_txt'] = round(
    cd(retrieval_inputs['image_embeddings'], retrieval_inputs['text_embeddings'][::5]).item(),
    2)

result_dir = '../modality-invariance-VLMs/results/metrics'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# file_name = os.path.join(result_dir, 'finetune.csv')
file_name = os.path.join(result_dir, 'pretrained.csv')
# writing column names
if not os.path.exists(file_name):
    with open(file_name, 'w', encoding='UTF8') as f:
        rows = ['dataset', 'model', 'path', 'loss', 'cmd_img_txt', 'cd_img_txt']

        for k in metrics['retrieval_unimodal'][0]:
            rows.append(f'retrieval_unimodal_t2i_k={k}')
            rows.append(f'retrieval_unimodal_i2t_k={k}')

        for k in metrics['retrieval_multimodal'][0]:
            rows.append(f'retrieval_multimodal_t2i_k={k}')
            rows.append(f'retrieval_multimodal_i2t_k={k}')

        writer = csv.writer(f)
        writer.writerow(rows)

# writing column values
with open(file_name, 'a', encoding='UTF8') as f:
    rows = [DATASET,
            MODEL,
            PATH if PATH is not None else 'None',
            retrieval_inputs['loss'],
            metrics['cmd_img_txt'],
            metrics['cd_img_txt']]

    for i in range(len(metrics['retrieval_unimodal'][0])):
        rows.append(metrics['retrieval_unimodal'][1][i])
        rows.append(metrics['retrieval_unimodal'][2][i])

    for i in range(len(metrics['retrieval_multimodal'][0])):
        rows.append(metrics['retrieval_multimodal'][1][i])
        rows.append(metrics['retrieval_multimodal'][2][i])

    writer = csv.writer(f)
    writer.writerow(rows)