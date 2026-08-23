import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor


class CLIPEmbedder:
    def __init__(self, model_id, device, revision = None):
        self.device = device

        self.model = (
            CLIPModel.from_pretrained(
                model_id,
                revision=revision,
            )
            .to(device)
            .eval()
        )

        self.processor = CLIPProcessor.from_pretrained(
            model_id,
            revision=revision,
        )

        self.embedding_dim = self.model.config.projection_dim

    def embed_images(self, image_paths, batch_size=32):

        emb = []
        for start in tqdm(range(0, len(image_paths), batch_size), desc="CLIP embeddings"):
            batch_paths = image_paths[start : start + batch_size]

            images = []

            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))

            inputs = self.processor(images=images, return_tensors="pt")

            inputs = {key: value.to(self.device) for key, value in inputs.items()}


            with torch.inference_mode():
                batch_embeddings = self.model.get_image_features(**inputs).pooler_output


                batch_embeddings = torch.nn.functional.normalize(
                    batch_embeddings, dim=-1
                )

            emb.append(batch_embeddings.cpu().numpy())

        return np.concatenate(emb, axis=0)




