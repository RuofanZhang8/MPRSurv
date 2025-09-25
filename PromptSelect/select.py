import os
import json
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModel

class SentenceTemplateSelector:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device

    def get_sentence_embeddings(self, data):
        print(data.shape)
        all_embeddings = []
        for i in range(data.shape[0]):
            class_embeddings = []
            for j in range(data.shape[1]):
                text = data[i, j]
                tokenized = self.model.text_encoder.tokenizer([text]).to(self.device)
                embedding = self.model.encode_text(tokenized, normalize=True)
                class_embeddings.append(embedding.squeeze(0))
            sentence_embeddings = torch.stack(class_embeddings, dim=0)
            all_embeddings.append(sentence_embeddings)
        return torch.stack(all_embeddings, dim=0)

    def uniformity_loss(self, text_embed, t=2):
        return torch.pdist(text_embed, p=2).pow(2.0).mul(-t).exp().mean()

    def ensemble(self, TE_list):
        return torch.stack(TE_list, dim=0).mean(dim=0)

    def select_best_sentence(self, data):
        TE_all = self.get_sentence_embeddings(data)
        best_idx = -1
        best_loss = float('inf')
        for i in range(TE_all.shape[0]):
            loss = self.uniformity_loss(TE_all[i])
            print(f"[Check] Sentence {i} uniformity_loss: {loss.item():.6f}")
            if loss < best_loss:
                best_loss = loss
                best_idx = i
        print(f"[Result] Best sentence index: {best_idx}, loss: {best_loss:.6f}")
        return best_idx, data[best_idx]

    def load_json_and_generate_sentences(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

model = AutoModel.from_pretrained('<YOUR_MODEL_PATH>', trust_remote_code=True)
print("Model loaded successfully")

selector = SentenceTemplateSelector(model, device='cuda')

data = selector.load_json_and_generate_sentences('<YOUR_JSON_PATH>/raw_matrix.json')
data = np.array(data)

result = []
for i in range(data.shape[2]):
    data_this = data[:, :, i]
    best_idx, best_sentence = selector.select_best_sentence(data_this)
    print(f"✅ Best sentence template: {best_sentence}")
    result.append(best_sentence.tolist())

with open("<YOUR_JSON_PATH>/2Select_sentence.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4)