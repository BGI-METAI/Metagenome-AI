import torch
import torch.nn as nn

class Classifier(nn.Module):
    """A classification head used to output protein family (or other targets) probabilities

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model, num_classes, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            # Single fully connected layer for classification head
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Sequential(nn.Linear(d_model, num_classes))
        # Multiple hidden layers followed by a linear layer for classification head
        else:
            layers = []
            prev_size = d_model
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, num_classes))
            self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)
