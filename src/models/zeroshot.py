import torch
from tqdm import tqdm

import clip.clip as clip

import src.templates as templates
import src.datasets as datasets

from src.models.modeling import ClassificationHead


def get_train_classnames(args):
    assert args.train_dataset is not None
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        None,
        location=args.data_location,
        batch_size=args.batch_size,
        classnames=args.classnames,
        num_workers=args.num_workers,
    )
    return dataset.classnames


def get_zeroshot_classifier(args, clip_model, classnames):
    assert args.template is not None
    template = getattr(templates, args.template)
    logit_scale = clip_model.logit_scale
    device = args.device
    clip_model.eval()
    clip_model.to(device)

    print('Getting zeroshot weights.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = clip.tokenize(texts).to(device) # tokenize
            embeddings = clip_model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head
