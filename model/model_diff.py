import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class EmbeddingDiffusion(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_steps=10):
        super(EmbeddingDiffusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.denoiser = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def add_noise(self, x, noise_level):
        noise = torch.randn_like(x) * noise_level
        return x + noise

    def remove_noise(self, noisy_x, noise_level):
        return self.denoiser(noisy_x / (1 + noise_level))

    def forward(self, x, noise_levels):
        noisy_x = x
        for noise_level in noise_levels:
            noisy_x = self.add_noise(noisy_x, noise_level)
        clean_x = noisy_x
        for noise_level in reversed(noise_levels):
            clean_x = self.remove_noise(clean_x, noise_level)
        return clean_x



def train(model, dataloader, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (x, lab) in enumerate(dataloader):
            noise_levels = torch.linspace(0.1, 1, steps=model.num_steps)
            noisy_x = model.add_noise(x, noise_levels[-1])  # 只添加最高级别的噪声

            # 计算去噪后的输出
            output = model(noisy_x, noise_levels)

            # 计算损失（例如均方误差）
            loss = nn.MSELoss()(output, x)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {epoch_loss / len(dataloader):.4f}')

# 超参数
embedding_dim = 1536
hidden_dim = 5120
num_steps = 10
num_epochs = 10
batch_size = 32

# 创建模型
model = EmbeddingDiffusion(embedding_dim, hidden_dim, num_steps)

model = model.to('cuda')

# 创建数据集
original_embeddings = torch.randn(1000, embedding_dim) # 1000个样本
dataset = TensorDataset(original_embeddings, torch.zeros(1000, 1))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

data_seq = torch.load("/home/share/huadjyin/home/wangshengfu/06_esm/data-seq-1w-mean.pt")
data_lable = torch.load("/home/share/huadjyin/home/wangshengfu/06_esm/data-lab-1w.pt")
data_lab = data_lable.view(-1, 1)
# 将张量列表转换为一个多维张量
tensor_stacked = torch.stack(data_seq)
squeezed_tensor_permute = torch.squeeze(tensor_stacked, dim=1)


# 创建 TensorDataset
embedding_dataset = TensorDataset(squeezed_tensor_permute, data_lab)
# 创建 DataLoader
batch_size = 4
dataloader = DataLoader(embedding_dataset, batch_size=batch_size, shuffle=True)


# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
train(model, dataloader, optimizer, num_epochs)