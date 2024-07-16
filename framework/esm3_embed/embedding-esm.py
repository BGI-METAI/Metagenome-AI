
import torch

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


import os
# 设置环境变量
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# 切换到指定目录
os.chdir("/home/share/huadjyin/home/wangshengfu/06_esm/")

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein

from splitdata import process_sequence_data


# 已经有了一个预训练的蛋白质模型
model_name = 'esm3_sm_open_v1'

model_esm: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda") # or "cpu"


# 获取单个序列的 embedding
def get_embedding(input):
    # 将序列转换为嵌入
    protein = ESMProtein(sequence=input)
    tok = model_esm.encode(protein)
    x_reshaped = tok.sequence.reshape(1, len(tok.sequence))
    with torch.no_grad():

        output = model_esm.forward(sequence_tokens=x_reshaped,)

        # 返回最后一层的嵌入
        return output.embeddings
    
# 获取整个文件的 embedding
def get_embedding_from_file(file_path):
    df = process_sequence_data(file_path)

    # 将Sequence列转换为列表
    sequence_list = df['Sequence'].tolist()
    # 将lable列转换为列表
    label_list = df['lable'].tolist()

    embeddings = []

    # for seq in sequence_list:
    #     embeddings.append((get_embedding(seq)))

    # 使用 ThreadPoolExecutor 来创建一个线程池
    with ThreadPoolExecutor(max_workers=2) as executor:  # 你可以根据你的CPU核心数来设置 max_workers
        # 使用 executor.map 来异步执行 get_embedding 函数，并将结果收集到 embeddings 列表中
        embeddings = list(executor.map(get_embedding, sequence_list))
    
    # 将列表转为张量
    label_tensor = torch.tensor(label_list)


    file_path = 'embed-data/data-seq.pt'
    if os.path.exists(file_path):
        print("文件存在")
    else:
        print("文件不存在")
        torch.save(embeddings, 'embed-data/data-seq.pt')
        torch.save(label_tensor, 'embed-data/data-lab.pt')


    torch.save(embeddings, 'embed-data/data-seq.pt')
    torch.save(label_tensor, 'embed-data/data-lab.pt')



def main(file_path):
    get_embedding_from_file(file_path)
  

if __name__ == "__main__":
    file_path = "data_AMP/test.fa"
    main(file_path)