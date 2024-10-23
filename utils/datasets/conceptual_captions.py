import os
import pandas as pd
from PIL import Image
from typing import Callable, Optional

from torch.utils.data import Dataset


class ConceptualCaptions(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to annotation file.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        image_transform: Optional[Callable] = None,
        caption_transform: Optional[Callable] = None,
        max_length_tokenizer: int = 64,
        no_cap_per_img = 1
    ) -> None:
        super(ConceptualCaptions, self).__init__()
        self.name = 'ConceptualCaptions'
        self.root = root
        self.image_transform = image_transform
        self.caption_transform = caption_transform
        self.max_length_tokenizer = max_length_tokenizer
        self.annFile = os.path.expanduser(annFile)
        ann = pd.read_csv(self.annFile)
        self.annotations = ann.set_index('id')['caption'].to_dict()
        self.ids = list(self.annotations.keys())

    def __getitem__(self, index: int):
        """
        Args:
            index (int): index in [0, self.__len__())

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """

        img_id = self.ids[index]

        # Image
        filename = os.path.join(self.root, img_id)
        img = Image.open(filename).convert("RGB")
        if self.image_transform is not None:
            img = self.image_transform(img)

        # Captions
        captions = self.annotations[img_id]
        
        # wanna limit the size of target here but error happened when search relevant
        # target = self.__remove_punctuation(target)
        # target = self.__limit_length(target)
        
        if self.caption_transform is not None:
            captions = self.caption_transform(captions,
                                              padding='max_length',
                                              max_length=self.max_length_tokenizer,
                                              truncation=True,
                                              return_tensors='pt')

        return img, captions

    def __len__(self) -> int:
        return len(self.ids)
    
    # def __remove_punctuation(self,texts):
    #     punctuation = string.punctuation
    #     translator = str.maketrans('', '', punctuation)
    #     for i, text in enumerate(texts):
    #         texts[i] = text.translate(translator)
    #     return texts
    
    # def __limit_length(self, captions, max_length=50):
    #     # limit the length of caption
    #     return [' '.join(caption.split()[:max_length]) for caption in captions]