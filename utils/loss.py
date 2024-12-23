import torch


def compute_clip_loss(image_features, text_features, temperature):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = temperature * image_features @ text_features.t()
    logits_per_text = temperature * text_features @ image_features.t()

    labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

    image_loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
    text_loss  = torch.nn.functional.cross_entropy(logits_per_text, labels)

    return (image_loss + text_loss) / 2


def compute_align_loss(image_features, text_features, alpha=2):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return (image_features - text_features).norm(p=2, dim=1).pow(alpha).mean()


def compute_unimodal_uniform_loss(x, t=2):
    x = x / x.norm(dim=-1, keepdim=True)

    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def compute_multimodal_uniform_loss(image_features, text_features, t=2):
    img_uniform = compute_unimodal_uniform_loss(image_features, t)
    txt_uniform = compute_unimodal_uniform_loss(text_features, t)
    
    return (img_uniform + txt_uniform) / 2


def compute_crossmodal_uniform_loss(image_features, text_features, t=2):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    pairwise_distances = torch.cdist(image_features, text_features, p=2)
    mask = ~torch.eye(image_features.size(0), dtype=torch.bool)
    result = pairwise_distances[mask].view(image_features.size(0), -1)
    result = result.pow(2).mul(-t).exp().mean().log()

    return result


def compute_CUA_loss(image_features, text_features, temperature, t=2, alpha=2):
    clip_loss = compute_clip_loss(image_features, text_features, temperature)
    uniform_loss = compute_multimodal_uniform_loss(image_features, text_features, t)
    align_loss = compute_align_loss(image_features, text_features, alpha)

    return clip_loss + uniform_loss + align_loss


def compute_CUAXU_loss(image_features, text_features, temperature, t=2, alpha=2):
    CUA_loss = compute_CUA_loss(image_features, text_features, temperature, t, alpha)
    crossmodal_uniform_loss = compute_crossmodal_uniform_loss(image_features, text_features, t)
    
    return  CUA_loss + crossmodal_uniform_loss