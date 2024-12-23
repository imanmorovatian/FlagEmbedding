import os
import json
from collections import OrderedDict, defaultdict
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Callable, Optional

from torch.utils.data import Dataset


class MSCOCOCaptions(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annotations_file (string): Path to annotation file.
        image_transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        caption_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        max_length_tokenizer (int): The maximum length required by some text tokenizers,
        no_cap_per_img (int): The number of captions for an image. Could be between 1 and 5 
    """

    def __init__(
        self,
        root: str,
        annotations_file: str,
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 77,
        no_cap_per_img = 1
    ) -> None:
        super(MSCOCOCaptions, self).__init__()
        self.name = 'MSCOCO'
        self.root = root
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer
        self.cpi = no_cap_per_img

        f_name = Path(annotations_file)
        with f_name.open('rt') as handle:
            annotations = json.load(handle, object_hook=OrderedDict)

        self.img_id_to_file_name = {}
        for img_info in annotations['images']:
            img_id = img_info['id']
            file_name = img_info['file_name']
            self.img_id_to_file_name[img_id] = file_name
    
        self.img_id_to_captions = defaultdict(list)
        for caption_info in annotations['annotations']:
            img_id = caption_info['image_id']
            self.img_id_to_captions[img_id].append(caption_info['caption'])

        self.img_ids = list(self.img_id_to_file_name.keys())

    def __getitem__(self, index: int):
        """
        Args:
            index (int): index in [0, self.__len__())

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """

        img_id = self.img_ids[index]

        # Image
        # filename = os.path.join(self.root, self.img_id_to_file_name[img_id])
        # img = Image.open(filename).convert("RGB")
        img = os.path.join(self.root, self.img_id_to_file_name[img_id])
        if self.image_transform is not None:
            img = self.image_transform(img)

        # Captions
        captions = list(
            map(str, np.random.choice(self.img_id_to_captions[img_id], size=self.cpi))
        )

        # wanna limit the size of target here but error happened when search relevant
        # target = self.__remove_punctuation(target)
        # target = self.__limit_length(target)
        
        if self.caption_transform is not None:
            captions = self.caption_transform(
                captions,
                padding='max_length',
                max_length=self.max_length_tokenizer,
                truncation=True,
                return_tensors='pt')

        return img, captions

    def __len__(self) -> int:
        return len(self.img_ids)
    

    # def __remove_punctuation(self,texts):
    #     punctuation = string.punctuation
    #     translator = str.maketrans('', '', punctuation)
    #     for i, text in enumerate(texts):
    #         texts[i] = text.translate(translator)
    #     return texts
    
    # def __limit_length(self, captions, max_length=50):
    #     # limit the length of caption
    #     return [' '.join(caption.split()[:max_length]) for caption in captions]