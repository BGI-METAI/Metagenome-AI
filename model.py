import torch.nn as nn
import torch.nn.functional as F
class ComplexBinaryClassifier(nn.Module):
    def __init__(self):
        super(ComplexBinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(1536*72, 1024)  # 第一个线性层
        self.relu1 = nn.ReLU()            # 第一个ReLU激活层
        self.dropout1 = nn.Dropout(0.1)   # 第一个Dropout层

        self.fc2 = nn.Linear(1024, 512)   # 第二个线性层
        self.relu2 = nn.ReLU()            # 第二个ReLU激活层
        self.dropout2 = nn.Dropout(0.1)   # 第二个Dropout层

        self.fc3 = nn.Linear(512, 128)   # 第三个线性层
        self.relu3 = nn.ReLU()            # 第三个ReLU激活层

        self.fc4 = nn.Linear(128, 1)     # 最后的线性层，输出一个值

    def forward(self, x):
        # 假设x的形状是[batch_size, L, 1536]
        x = x.view(x.size(0), -1)  # 展平到[batch_size, sequence_length * feature_size]
        '''
        # 直接删除第二个维度 也就是序列维度
        x = torch.squeeze(x, dim=1)  # x的形状变为[batch_size, 52, 1536]
        # 然后沿着52维度进行聚合操作，这里以求和为例
        x = torch.sum(x, dim=1)  # x的形状变为[batch_size, 1536] 
        '''

        x = self.dropout1(self.relu1(self.fc1(x)))  # 经过第一个线性层、ReLU和Dropout
        x = self.dropout2(self.relu2(self.fc2(x)))  # 经过第二个线性层、ReLU和Dropout
        x = self.relu3(self.fc3(x))                # 经过第三个线性层和ReLU
        x = x.view(-1, 128)                      # 再次展平以匹配最后一个线性层的输入维度
        logits = self.fc4(x)                       # 经过最后的线性层
        # probabilities = torch.sigmoid(logits)
        return logits

    def loss(self, outputs, targets):
        # 计算二元交叉熵损失
        return F.binary_cross_entropy_with_logits(outputs, targets)