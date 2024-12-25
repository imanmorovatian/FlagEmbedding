import argparse
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.datasets.flickr30k_captions import Flickr30kCaptions
from utils.datasets.mscoco_captions import MSCOCOCaptions
from utils.datasets.conceptual_captions import ConceptualCaptions

from utils.loss import compute_clip_loss, compute_CUA_loss, compute_CUAXU_loss

from models.custom_vista import CustomVISTA


def create_model(name, local_weights=None): 
    if name == 'VISTA_LU':
        # Freeze the image encoder, while fine tune the text encoder
        return CustomVISTA(frozen_text_encoder=False, frozen_image_encoder=True)
    
    elif name == 'VISTA_UL':
        # Freeze the text encoder, while fine tune the image encoder
        return CustomVISTA(frozen_text_encoder=True, frozen_image_encoder=False)
    
    elif name == 'VISTA_UU':
        # Fine tune the whole model
        return CustomVISTA(frozen_text_encoder=False, frozen_image_encoder=False)
    
    elif name == 'VISTA_UL+LU':
        # Firtly, finetune the image encoder (using the local weights), and then finetune the text encoder
        return CustomVISTA(frozen_text_encoder=False,
                          text_encoder_from_local=False,
                          frozen_image_encoder=True,
                          image_encoder_from_local=True,
                          local_pretrained_weights_path=local_weights)
    
    elif name == 'VISTA_LU+UL':
        # Firtly, finetune the text encoder (using the local weights), and then finetune the image encoder
        return CustomVISTA(frozen_text_encoder=True,
                          text_encoder_from_local=True,
                          frozen_image_encoder=False,
                          image_encoder_from_local=False,
                          local_pretrained_weights_path=local_weights)
    
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
    parser.add_argument('--no_epochs', type=int, required=True, help='number of epochs', dest='NO_EPOCHS')

    args = parser.parse_args()

    return args


args = parse_args()
MODEL = args.MODEL
PATH = args.PATH
LOSS = args.LOSS
DATASET = args.DATASET
BATCH_SIZE = args.BATCH_SIZE
NO_EPOCHS = args.NO_EPOCHS
NUM_WORKERS = 2

# for debug
# MODEL = 'VISTA_LU+UL'
# PATH = 'weights/VISTA/LiUi_clip_loss_mscoco.pth'
# LOSS = 'clip'
# DATASET = 'mscoco'
# BATCH_SIZE = 64
# NO_EPOCHS = 10
# NUM_WORKERS = 2

assert MODEL in ['VISTA_LU', 'VISTA_UL', 'VISTA_UU', 'VISTA_UL+LU', 'VISTA_LU+UL']

assert DATASET in ['mscoco', 'flickr30k', 'conceptualCaptions']

assert LOSS in ['clip', 'cua', 'cuaxu']

model = create_model(MODEL, PATH)
    
if DATASET == 'mscoco':
    train_dataset = MSCOCOCaptions(root='../modality-invariance-VLMs/data/images/mscoco/train2017/',
						annotations_file='../modality-invariance-VLMs/data/annotations/mscoco/train2017full_captions.json',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    val_dataset = MSCOCOCaptions(root='../modality-invariance-VLMs/data/images/mscoco/val2017/',
						annotations_file='../modality-invariance-VLMs/data/annotations/mscoco/val2017_captions.json',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # test split is sampled from train split
    # test_dataset = MSCOCOCaptions(root='../modality-invariance-VLMs/data/images/mscoco/train2017/',
	# 					annotations_file='../modality-invariance-VLMs/data/annotations/mscoco/test2017_captions.json',
    #                     image_transform=model.transform,
    #                     caption_transform=model.text_tokenizer)
    # test_sampler = SequentialSampler(test_dataset)
    # test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    test_dataloader = val_dataloader
    
elif DATASET == 'flickr30k':
    train_dataset = Flickr30kCaptions(root='../modality-invariance-VLMs/data/images/flickr30k/',
                        annotations_file='../modality-invariance-VLMs/data/annotations/flickr30k/train.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    val_dataset = Flickr30kCaptions(root='../modality-invariance-VLMs/data/images/flickr30k/',
                        annotations_file='../modality-invariance-VLMs/data/annotations/flickr30k/val.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    test_dataset = Flickr30kCaptions(root='../modality-invariance-VLMs/data/images/flickr30k/',
                        annotations_file='../modality-invariance-VLMs/data/annotations/flickr30k/test.token',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

elif DATASET == 'conceptualCaptions':
    train_dataset = ConceptualCaptions(root='../modality-invariance-VLMs/data/images/conceptualCaptions/',
                        annotations_file='../modality-invariance-VLMs/data/annotations/conceptualCaptions/train.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    val_dataset = ConceptualCaptions(root='../modality-invariance-VLMs/data/images/conceptualCaptions/',
                        annotations_file='../modality-invariance-VLMs/data/annotations/conceptualCaptions/val.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    test_dataset = ConceptualCaptions(root='../modality-invariance-VLMs/data/images/conceptualCaptions/',
                        annotations_file='../modality-invariance-VLMs/data/annotations/conceptualCaptions/test.csv',
                        image_transform=model.transform,
                        caption_transform=model.text_tokenizer)
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

model.orchestrate_training(DATASET, train_dataloader, val_dataloader, test_dataloader,
                            loss_function, BATCH_SIZE, NO_EPOCHS)