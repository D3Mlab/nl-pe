from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractmethod
import logging
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer

class BaseEmbedder(ABC):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

    @abstractmethod
    def embed(self, texts_csv_path = '', pckl_path = ''):
        #inputs: path to csv with headers where first col is doc id and second is doc text
        #output, pickled dict of {<d_id>: <torch_embedding>}
        raise NotImplementedError("This method must be implemented by a subclass.")
    
class HuggingFaceEmbedderSentenceTransformers(BaseEmbedder):

    def __init__(self, model_name='', matryoshka_dim = None):
        super().__init__()

        #e.g. model_name = Qwen/Qwen3-Embedding-8B
        self.model = SentenceTransformer(
             model_name,
             model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
             tokenizer_kwargs={"padding_side": "left"},
             truncate_dim=matryoshka_dim, 
         )
        self.logger.info("Primary model device: %s", next(self.model[0].auto_model.parameters()).device)
        self.logger.info("Matryoshka dimension set to: %s", matryoshka_dim if matryoshka_dim else "full")

    def embed(self, texts_csv_path, pckl_path):
        """
        Read CSV with first column doc_id and second column doc_text.
        Encode documents into torch tensors and pickle as {doc_id: tensor}.
        """
        df = pd.read_csv(texts_csv_path, header=0)
        doc_ids = df.iloc[:, 0].tolist()
        texts = df.iloc[:, 1].tolist()

        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=True
        )

        embedding_dict = {doc_id: emb for doc_id, emb in zip(doc_ids, embeddings)}

        with open(pckl_path, "wb") as f:
            pickle.dump(embedding_dict, f)
        self.logger.info("Saved embeddings to %s", pckl_path)
        
    @staticmethod
    def _last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def _get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'
