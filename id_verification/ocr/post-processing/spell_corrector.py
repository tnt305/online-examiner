from spellchecker import SpellChecker
from spello.model import SpellCorrectionModel
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np


class SpellCorrector:
    def __init__(self, spell_correction_model_path, spell_checker_language='en', sentence_transformer_model='sentence-transformers/all-MiniLM-L6-v2'):
        self.cker1 = SpellCorrectionModel(language='en')
        self.cker1.load(spell_correction_model_path)
        self.cker2 = SpellChecker(language=spell_checker_language)
        self.tokenizer = AutoTokenizer.from_pretrained(sentence_transformer_model)
        self.model = AutoModel.from_pretrained(sentence_transformer_model)

    def cosine_similarity(self, vector_a, vector_b):
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        similarity = dot_product / (norm_a * norm_b)
        return similarity

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def spell_correction(self, input_text):
        output1 = self.cker1.spell_correct(input_text)['spell_corrected_text']
        input_text = input_text.split(' ')
        text1 = []
        text2 = []
        for i in range(len(input_text)):
            if input_text[i] not in self.cker2.unknow(input_text):
                text1.append(item)
            else:
                item = list(self.cker2.candidates(input_text[i]))
                if len(item) > 1:
                    text1.append(item[0])
                    text2.append(item[-1])
                else:
                    text1.append(item[0])
                    text2.append(item[0])
        text1 = " ".join(text1)
        text2 = " ".join(text2)
        chosen_text = [text1, text2, output1]

        embedded = []
        for text in chosen_text:
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            embedded.append(sentence_embeddings)

        return chosen_text[self.finest_representation(embedded)]

    def finest_representation(self, embeddings):
        cosine1 = self.cosine_similarity(embeddings[0], embeddings[2])
        cosine2 = self.cosine_similarity(embeddings[0], embeddings[1])
        cosine3 = self.cosine_similarity(embeddings[1], embeddings[2])

        max_cosine = max(cosine1, cosine2, cosine3)

        if max_cosine == cosine3:
            second_max_cosine = max(cosine1, cosine2)
            if second_max_cosine == cosine1:
                return 0
            elif second_max_cosine == cosine2:
                return 1
        elif max_cosine == cosine2:
            second_max_cosine = max(cosine1, cosine3)
            if second_max_cosine == cosine1:
                return 0
            elif second_max_cosine == cosine3:
                return 2
        elif max_cosine == cosine1:
            second_max_cosine = max(cosine2, cosine3)
            if second_max_cosine == cosine2:
                return 1
            elif second_max_cosine == cosine3:
                return 2
