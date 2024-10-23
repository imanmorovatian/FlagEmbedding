import torch
from typing import List
import numpy as np

"""
This script, retrieval.py, is designed to compute retrieval on two datasets:
COCOCaptions and Flickr30kCaptions. Both datasets associate 5 captions with each image,
providing rich textual context for the images.

Here's a brief overview of the datasets:
1. COCOCaptions: This dataset contains a large collection of images along with 5 captions
describing each image. The captions in COCOCaptions are typically short, succinct descriptions
of the content of the image.

2. Flickr30kCaptions: Similarly, this dataset also provides 5 captions for each image.
However, compared to COCOCaptions, the captions in Flickr30kCaptions tend to be more
diverse in style and content. This diversity can be useful for testing the robustness
of retrieval algorithms.

By leveraging the textual information from these datasets, the retrieval.py script aims to implement
algorithms that can find semantically similar images based on their associated captions.
"""

class CrossModalRetrieval:
    def __init__(self,
                 image_encodings,
                 text_encodings,
                 text_to_image_map,
                 image_to_text_map,
                 cpi, # the number of captions per image
                 search_space='unimodal',
                 k_vals: List[int]=[1, 5, 10]):
        
        assert search_space in ['unimodal', 'multimodal'], "search_space must be either 'unimodal' or 'multimodal'"
        
        self.image_encodings = image_encodings
        self.text_encodings = text_encodings
        self.text_to_image_map = text_to_image_map
        self.image_to_text_map = image_to_text_map
        self.cpi = cpi
        self.device = self.image_encodings.device
        
        self.search_space = search_space
        self.k_vals = k_vals
        self.t2i, self.i2t = None, None

    def __text2image_mask(self, text_text_dist):
        n = text_text_dist.shape[0]
        t = np.ones((n, n))

        for i in range(n):
            start_col = self.cpi * (i // self.cpi)
            t[i, start_col:start_col+self.cpi] = 0

        mask = torch.tensor(t, dtype=torch.bool)
        return torch.masked_select(text_text_dist, mask).reshape(n, n-self.cpi)
    
    def __recall_at_k(self):

        num_text = self.text_encodings.shape[0]
        num_im = self.image_encodings.shape[0]

        captions_per_image = self.image_to_text_map.shape[1]

        # TEXT-TO-IMAGE
        text_to_image_recall = []
        if self.search_space == 'unimodal':
            dist_matrix = self.text_encodings @ self.image_encodings.T  # dist_matrix[i] gives logits for ith text
        else:
            uni         = self.text_encodings @ self.text_encodings.T  # dist_matrix[i] gives logits for ith text
            # uni         = uni.flatten()[1:].view(num_text-1, num_text+1)[:,:-1].reshape(num_text, num_text-1)
            uni         = uni.cpu()
            uni         = self.__text2image_mask(uni)
            cross       = self.text_encodings @ self.image_encodings.T
            cross       = cross.cpu()
            dist_matrix = torch.cat([cross,uni], dim=1)

        # Note: this matrix is pretty big (5000 x 25000 with dtype float16 = 250MB)
        #  torch.argsort runs out of memory for me (6GB VRAM) so I move to CPU for sorting
        dist_matrix = dist_matrix.cpu()

        # Sort in descending order; first is the biggest logit
        inds = torch.argsort(dist_matrix, dim=1, descending=True)
        # inds = inds.to(self.device)

        for k in self.k_vals:
            # Extract top k indices only
            topk = inds[:, :k].to(self.device)

            # Correct iff one of the top_k values equals the correct image (as given by text_to_image_map)
            correct = torch.eq(topk, self.text_to_image_map.unsqueeze(-1)).any(dim=1)
            num_correct = correct.sum().item()
            text_to_image_recall.append(num_correct / num_text)

        # IMAGE-TO-TEXT
        image_to_text_recall = []
        if self.search_space == 'unimodal':
            dist_matrix = dist_matrix.T
        else:
            uni         = self.image_encodings @ self.image_encodings.T  # dist_matrix[i] gives logits for ith text
            uni         = uni.flatten()[1:].view(num_im-1, num_im+1)[:,:-1].reshape(num_im, num_im-1)
            uni         = uni.cpu()
            cross       = cross.T
            dist_matrix = torch.cat([cross,uni], dim=1)
            
        dist_matrix = dist_matrix.cpu()

        # Sort in descending order; first is the biggest logit
        inds = torch.argsort(dist_matrix, dim=1, descending=True)
        # inds = inds.to(self.device)

        for k in self.k_vals:
            # Extract top k indices only
            topk = inds[:, :k].to(self.device)

            correct = torch.zeros((num_im,), dtype=torch.bool).cuda()

            #  For each image, check whether one of the 5 relevant captions was retrieved
            # Check if image matches its ith caption (for i=0..4)
            for i in range(captions_per_image):
                contains_index = torch.eq(topk, self.image_to_text_map[:, i].unsqueeze(-1)).any(dim=1)
                correct = torch.logical_or(correct, contains_index)

            num_correct = correct.sum().item()
            image_to_text_recall.append(num_correct / num_im)#

        return text_to_image_recall, image_to_text_recall

    def compute(self):
        t2i, i2t = self.__recall_at_k()
        self.t2i, self.i2t = t2i, i2t
        return self.k_vals, t2i, i2t
    