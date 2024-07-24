import pathlib
import pickle

import torch
from torch.nn import Identity
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

from embed.embedding_esm import EsmEmbedding


class Esm3Embedding(EsmEmbedding):
    def __init__(self, pooling="mean"):

        model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda") # or "cpu"

        self.model = model
        self.alphabet = EsmSequenceTokenizer()
        # 只需要用其 获取 embedding
        self.model.eval()
        self.embed_dim = 1536  # TODO: model.embed_dim
        self.pooling = pooling

        for param in self.model.parameters():
            param.requires_grad = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_embedding(self, batch):

        tokens = self.alphabet.batch_encode_plus(batch["sequence"], padding=True)[
            "input_ids"
        ]  # encode
        batch_tokens = torch.tensor(tokens, dtype=torch.int64).cuda()  # To GPU

        with torch.no_grad():
            esm_result = self.model(sequence_tokens=batch_tokens)

        return self._pooling(
            self.pooling,
            esm_result.embeddings,
            batch_tokens,
            self.alphabet.pad_token_id,
        )
    
    def _pooling(self, strategy, tensors, batch_tokens, pad_token_id):
        """Perform pooling on [batch_size, seq_len, emb_dim] tensor

        Args:
            strategy: One of the values ["mean", "max", "cls"]
        """
        if strategy == "cls":
            seq_repr = tensors[:, 0, :]
        elif strategy == "mean":
            seq_repr = []
            batch_lens = (batch_tokens != pad_token_id).sum(1)

            for i, tokens_len in enumerate(batch_lens):
                seq_repr.append(tensors[i, 1 : tokens_len - 1].mean(0))

            seq_repr = torch.vstack(seq_repr)
        else:
            raise NotImplementedError("This type of pooling is not supported")
        return seq_repr
    
    def get_embedding_dim(self):
        return self.embed_dim

    def store_embeddings(self, batch, out_dir):
        # 存储但蛋白质 embedding 按照蛋白质的 id 存储在 out_dir 目录下
        # 格式为 pkl 文件，每个文件名为 chunk_01.pkl chunk_02.pkl
        # 每个 chunk文件包含 10240 条蛋白质序列信息
        # 其中存储为 字典形式 key值为 protein_id, value 为 value_dict格式
        # value_dict 中包含 key emb:[mean, max, cls] seq: 蛋白质序列 lable: 蛋白质标签 probality: 预测概率

        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        embeddings = self.get_embedding(batch)
        embeddings = embeddings.detach().cpu()
        embeddings_dict = {}
        for protein_id, emb in zip(batch["protein_id"], embeddings):
      
            # 构建单个蛋白质的嵌入信息字典
            protein_info = {
                "emb": emb.numpy(),
                "seq": batch["sequence"][protein_id],  # 假设sequence是一个字典，使用protein_id作为键
                "label": batch["label"][protein_id],    # 同上
                "probability": 0
            }
            # 将蛋白质ID作为键，protein_info作为值添加到embeddings_dict字典中
            embeddings_dict[protein_id] = protein_info


        return embeddings_dict, out_dir
        # for protein_id, emb in zip(batch["protein_id"], embeddings):
        #     embeddings_dict = {
        #         self.pooling: emb.numpy(),
        #     }

        #     with open(f"{out_dir}/{protein_id}.pkl", "wb") as file:
        #         pickle.dump(embeddings_dict, file)
        '''
        # 初始化一个列表来存储每50个embeddings_dict
        chunk_size = 1280
        embeddings_chunks = []

        for protein_id, emb in zip(batch["protein_id"], embeddings):
            # 创建embeddings_dict            
            embeddings_dict = {
                self.pooling: emb.numpy()
                
            }
            
            # 将当前的embeddings_dict添加到列表中
            embeddings_chunks.append(embeddings_dict)
            
            # 检查列表长度是否达到50
            if len(embeddings_chunks) == chunk_size:
                # 保存当前列表中的所有embeddings_dict到一个文件
                chunk_file_name = f"{out_dir}/chunk_{len(embeddings_chunks)//chunk_size}.pkl"
                with open(chunk_file_name, "wb") as file:
                    pickle.dump([emb_dict for emb_dict in embeddings_chunks], file)
                
                # 清空列表以便收集下一批
                embeddings_chunks.clear()

        # 检查是否有剩余的embeddings_dict需要保存
        if embeddings_chunks:
            chunk_file_name = f"{out_dir}/chunk_{len(embeddings_chunks)//chunk_size + 1}.pkl"
            with open(chunk_file_name, "wb") as file:
                pickle.dump([emb_dict for emb_dict in embeddings_chunks], file)


        
        '''
