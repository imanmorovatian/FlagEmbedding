import os
import csv

import torch
from torch.utils.data import DataLoader, SequentialSampler

from utils.datasets.mscoco_captions import MSCOCOCaptions
from utils.metrics.retrieval import CrossModalRetrieval
from utils.metrics.metrics import CMD, CD
from utils.contrastive_loss import compute_contrastive_loss

from FlagEmbedding.visual.modeling import Visualized_BGE


device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
CPI = 5
NUM_WORKERS = 2

model = Visualized_BGE(model_name_bge='BAAI/bge-base-en-v1.5', model_weight='Visualized_base_en_v1.5.pth')
model = model.to(device)

test_dataset = MSCOCOCaptions(
    root='../modality-invariance-VLMs/data/images/mscoco/train2017/',
    annotations_file='../modality-invariance-VLMs/data/annotations/mscoco/test2017_captions.json',
    image_transform=None,
    caption_transform=None,
    no_cap_per_img=CPI)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


image_to_text_map = []
text_to_image_map = []
text_index = 0
image_index = 0
total_loss = 0
total_batches = 0
image_features = []
text_features = []
total_loss = 0.0
total_batches = 0
temperature = 1 / 0.07
model.eval()

with torch.no_grad():
    for batch in test_dataloader:
        images, text = batch
        
        batch_size = len(images)
        captions_per_image = len(text)

        for ـ in range(batch_size):
            # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
            text_indices = list(range(text_index, text_index + captions_per_image))
            image_to_text_map.append(text_indices)
            text_index += captions_per_image

            # Each of the next captions_per_image text captions correspond to the same image
            text_to_image_map += [image_index] * captions_per_image
            image_index += 1

        batch_image_embeds = []
        batch_text_embeds = []
        for img_id in range(batch_size):
            batch_image_embeds.append( model.encode(image=images[img_id]) )

            temp = []
            for cap_id in range(captions_per_image):
                temp.append( model.encode(text=text[cap_id][img_id]) )
            
            temp = torch.vstack(temp)
            batch_text_embeds.append(temp)

        batch_image_embeds = torch.vstack(batch_image_embeds)
        batch_image_embeds = batch_image_embeds.to(device)

        batch_text_embeds = torch.vstack(batch_text_embeds)
        batch_text_embeds = batch_text_embeds.to(device)

        loss = compute_contrastive_loss(batch_image_embeds, batch_text_embeds[::captions_per_image], temperature)
        total_loss += loss.item()
        total_batches += 1

        image_features.append(batch_image_embeds)
        text_features.append(batch_text_embeds)

    text_to_image_map = torch.Tensor(text_to_image_map).to(device)
    image_to_text_map = torch.Tensor(image_to_text_map).to(device)

    image_features = torch.vstack(image_features)
    image_features = image_features.to(device)
    image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)

    text_features = torch.vstack(text_features)
    text_features = text_features.to(device)
    text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            
retrieval_inputs = {
    'text_embeddings': text_features.squeeze(),
    'image_embeddings': image_features.squeeze(),
    'text_to_image_mapping': text_to_image_map.squeeze(),
    'image_to_text_mapping': image_to_text_map.squeeze(),
    'loss': total_loss / total_batches
}

metrics = {}

retrieval_obj = CrossModalRetrieval(image_encodings=retrieval_inputs['image_embeddings'],
                                    text_encodings=retrieval_inputs['text_embeddings'],
                                    text_to_image_map=retrieval_inputs['text_to_image_mapping'],
                                    image_to_text_map=retrieval_inputs['image_to_text_mapping'],
                                    cpi=captions_per_image,
                                    search_space='unimodal',
                                    k_vals=[1,5,10])
metrics['retrieval_unimodal'] = retrieval_obj.compute()

retrieval_obj = CrossModalRetrieval(image_encodings=retrieval_inputs['image_embeddings'],
                                    text_encodings=retrieval_inputs['text_embeddings'],
                                    text_to_image_map=retrieval_inputs['text_to_image_mapping'],
                                    image_to_text_map=retrieval_inputs['image_to_text_mapping'],
                                    cpi=captions_per_image,
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

if not os.path.exists(os.path.join(result_dir, 'metrics.csv')):
    with open(os.path.join(result_dir, 'metrics.csv'), 'w', encoding='UTF8') as f:
        rows = ['dataset', 'model', 'loss', 'cmd_img_txt', 'cd_img_txt']

        for k in metrics['retrieval_unimodal'][0]:
            rows.append(f'retrieval_unimodal_t2i_k={k}')
            rows.append(f'retrieval_unimodal_i2t_k={k}')

        for k in metrics['retrieval_multimodal'][0]:
            rows.append(f'retrieval_multimodal_t2i_k={k}')
            rows.append(f'retrieval_multimodal_i2t_k={k}')

        writer = csv.writer(f)
        writer.writerow(rows)

with open(os.path.join(result_dir, 'metrics.csv'), 'a', encoding='UTF8') as f:
    rows = ['mscoco', 'VISTA', retrieval_inputs['loss'], metrics['cmd_img_txt'], metrics['cd_img_txt']]

    for i in range(len(metrics['retrieval_unimodal'][0])):
        rows.append(metrics['retrieval_unimodal'][1][i])
        rows.append(metrics['retrieval_unimodal'][2][i])

    for i in range(len(metrics['retrieval_multimodal'][0])):
        rows.append(metrics['retrieval_multimodal'][1][i])
        rows.append(metrics['retrieval_multimodal'][2][i])

    writer = csv.writer(f)
    writer.writerow(rows)