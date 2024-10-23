import torch


def compute_contrastive_loss(image_features, text_features, temperature):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = temperature * image_features @ text_features.t()
    logits_per_text = temperature * text_features @ image_features.t()

    labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

    image_loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
    text_loss  = torch.nn.functional.cross_entropy(logits_per_text, labels)

    return (image_loss + text_loss) / 2