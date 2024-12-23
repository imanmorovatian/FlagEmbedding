import os
import wandb

import torch
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from utils.custom_schedulers import get_cosine_schedule_with_warmup

from FlagEmbedding.visual.modeling import Visualized_BGE


class CustomVISTA():
    def __init__(self,
                 sentence_pooling_method: str = 'mean',
                 frozen_text_encoder: bool = True,
                 text_encoder_from_local: bool = False,
                 frozen_image_encoder: bool = True,
                 image_encoder_from_local: bool = False,
                 local_pretrained_weights_path: str = None):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = Visualized_BGE(
            model_name_bge='BAAI/bge-base-en-v1.5',
            model_weight='weights/Visualized_base_en_v1.5.pth',
            sentence_pooling_method=sentence_pooling_method
            )

        self.transform = self.model.preprocess_val
        self.text_tokenizer = self.model.tokenizer

        # By default, weights are loaded from the Internet. If you want to load some specific weights from
        # the local, you must specify using the *_from_local arguments
        if local_pretrained_weights_path is not None:
            local_state_dict = torch.load(local_pretrained_weights_path, map_location=torch.device('cpu'))
            hybrid_state_dict = {}

            if text_encoder_from_local:
                for name, param in local_state_dict.items():
                    if name.startswith('bge'):
                        hybrid_state_dict[name] = param

            if image_encoder_from_local:
                for name, param in local_state_dict.items():
                    # don't worry, visual linear layer params start with visual_proj
                    if name.startswith('model_visual'):
                        hybrid_state_dict[name] = param

            filled_params = set(hybrid_state_dict.keys())
            for name, param in self.model.named_parameters():
                if name not in filled_params:
                    hybrid_state_dict[name] = param

            self.model.load_state_dict(hybrid_state_dict)

        
        if frozen_text_encoder:
            for name, param in self.model.named_parameters():
                if name.startswith('bge'):
                    param.requires_grad = False

        if frozen_image_encoder:
            for name, param in self.model.named_parameters():
                # don't worry, visual linear layer params start with visual_proj
                if name.startswith('model_visual'):
                    param.requires_grad = False

        self.model = self.model.to(self.device)

        self.name = 'VISTA'

        if frozen_image_encoder:
            self.name += '_L' # (L)ocked 
            if image_encoder_from_local:
                self.name += 'l' # loaded from (l)ocal
            else:
                self.name += 'i' # loaded from the (I)nternet
        else:
            self.name += '_U' # (U)nlocked
            if image_encoder_from_local:
                self.name += 'l' # loaded from (l)ocal
            else:
                self.name += 'i' # loaded from the (I)nternet

        if frozen_text_encoder:
            self.name += 'L' # (L)ocked 
            if text_encoder_from_local:
                self.name += 'l' # loaded from (l)ocal
            else:
                self.name += 'i' # loaded from the (I)nternet
        else:
            self.name += 'U' # (U)nlocked
            if text_encoder_from_local:
                self.name += 'l' # loaded from (l)ocal
            else:
                self.name += 'i' # loaded from the (I)nternet
            
    def train(self, dataloader, loss_function, optimizer, scheduler, grad_scaler):
        self.model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            images, text = batch
            text = text.to(self.device)
            images = images.to(self.device)
            
            optimizer.zero_grad()

            with autocast('cuda'):
                img_embeds = self.model.encode(image=images)
                text_embeds = self.model.encode(text=text)               
                temperature = self.model.model_visual.logit_scale.exp()
                
                loss = loss_function(img_embeds, text_embeds, temperature)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_loss += loss.item()

            self.model.model_visual.logit_scale.data = torch.clamp(self.model.model_visual.logit_scale.data, 0, 4.6052)
            scheduler.step() 
            
        epoch_loss = epoch_loss / len(dataloader)

        return epoch_loss      

    def evaluation(self, dataloader, loss_function):
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                images, text = batch
                text = text.to(self.device)
                images = images.to(self.device)

                with autocast('cuda'):
                    img_embeds = self.model.encode(image=images)
                    text_embeds = self.model.encode(text=text)               
                    temperature = self.model.model_visual.logit_scale.exp()
                    
                    loss = loss_function(img_embeds, text_embeds, temperature)
                
                epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(dataloader)

        return epoch_loss

    def orchestrate_training(self, dataset_name, train_dataloader, val_dataloader, test_dataloader,
                             loss_function, batch_size, no_epochs):
                
        optimizer = AdamW(self.model.parameters(), lr=5e-5, eps=1.0e-08, weight_decay=0.1)
        grad_scaler = GradScaler()
        gradient_accumulation_steps = 1
        t_total = len(train_dataloader) // gradient_accumulation_steps * no_epochs
        # num_warmup_steps = int(0.20 * t_total)
        num_warmup_steps = 0
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
            )

        wandb_config = {
            'number_of_parameters': sum(p.numel() for p in self.model.parameters()),
            'batch_size': batch_size,
            'number_of_epochs': no_epochs,
        }

        wandb.init(
            config=wandb_config,
            entity='iman_morovatian',
            project='Thesis',
            name=f'Training {self.name} {dataset_name} {loss_function.__name__}'
            )

        for epoch in range(no_epochs):
            train_loss = self.train(train_dataloader, loss_function, optimizer, scheduler, grad_scaler)
            val_loss = self.evaluation(val_dataloader, loss_function)
            
            print(f'Epoch: {epoch+1} --> train loss = {train_loss}, validation loss = {val_loss}')
            wandb.log({
                'epoch': epoch+1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })

        test_loss = self.evaluation(test_dataloader, loss_function)

        wandb.log({'test_loss': test_loss})
        wandb.finish()

        model_name, ext_name = self.name.split('_')
        loss_name = loss_function.__name__.split('compute_')[-1]
        folder = f'weights/{model_name}'
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(self.model.state_dict(), f'{folder}/{ext_name}_{loss_name}_{dataset_name}.pth')
        print(f'Saved model in {folder}')

    def encode_for_retrieval(self, dataloader, loss_function):
        self.model.eval()
        
        image_to_text_map = []
        text_to_image_map = []
        text_index = 0
        image_index = 0
        total_loss = 0
        total_batches = 0
        image_features = []
        text_features = []

        with torch.no_grad():
            for batch in dataloader:
                images, text = batch
                
                batch_size, captions_per_image, _ = text['input_ids'].size()
                for ـ in range(batch_size):
                    # the next image corresponds to text captions [text_index ... text_index + captions_per_image - 1]
                    text_indices = list(range(text_index, text_index + captions_per_image))
                    image_to_text_map.append(text_indices)
                    text_index += captions_per_image

                    # Each of the next captions_per_image text captions correspond to the same image
                    text_to_image_map += [image_index] * captions_per_image
                    image_index += 1


                text = text.to(self.device)                
                images = images.to(self.device)

                img_embeds = self.model.encode(image=images)
                text_embeds = self.model.encode(text=text) 

                image_features.append(img_embeds)
                text_features.append(text_embeds)
                
                temperature = self.model.model_visual.logit_scale.exp()
                
                loss = loss_function(img_embeds, text_embeds[::captions_per_image], temperature)
                total_loss += loss.item()
                total_batches += 1


            text_to_image_map = torch.Tensor(text_to_image_map).to(self.device)
            image_to_text_map = torch.Tensor(image_to_text_map).to(self.device)

            text_features = torch.vstack(text_features)
            text_features = torch.nn.functional.normalize(text_features, p=2.0, dim=1)

            image_features = torch.vstack(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2.0, dim=1)
            
            avg_loss = total_loss / total_batches


        return {
            'text_embeddings': text_features.squeeze(),
            'image_embeddings': image_features.squeeze(),
            'text_to_image_mapping': text_to_image_map.squeeze(),
            'image_to_text_mapping': image_to_text_map.squeeze(),
            'loss': avg_loss
        }