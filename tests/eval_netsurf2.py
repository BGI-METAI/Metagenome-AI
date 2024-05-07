import os

import numpy as np
import torch
from torch import nn, optim
from torch.nn import BatchNorm1d
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NSPData(Dataset):
    def __init__(self, dataset, indices=False):
        """ Constructor
        Args:
            X (np.array): The array that contains the training data
            y (np.array): The array that contains the test data
        """

        self.data = torch.tensor(dataset['data'][:, :, :50]).float()
        self.targets = torch.tensor(dataset['data'][:, :, 50:68]).float()
        self.lengths = torch.tensor([sum(target[:, 0] == 1) for target in self.targets])

        self.unknown_nucleotide_mask()

    def unknown_nucleotide_mask(self):
        """ Augments the target with a unknown nucleotide mask
            by finding entries that don't have any residue
        """

        # creates a mask based on the one hot encoding
        unknown_nucleotides = torch.max(self.data[:, :, :20], dim=2)
        unknown_nucleotides = unknown_nucleotides[0].unsqueeze(2)

        # merge the mask to first position of the targets
        self.targets = torch.cat([self.targets, unknown_nucleotides], dim=2)

    def __getitem__(self, index):
        """ Returns train and test data at an index
        Args:
            index (int): Index at the array
        """
        X = self.data[index]
        y = self.targets[index]
        lengths = self.lengths[index]

        return X, y, lengths

    def __len__(self):
        """Returns the length of the data"""
        return len(self.data)


def prep_data(data_dir: str):
    """
    each data is compressed as a numpy zip and the dimensions of the dataset: [sequence, position, label]
    [0:20] Amino Acids (sparse encoding) (Unknown residues are stored as an all-zero vector)
    [20:50] hmm profile
    [50] Seq mask (1 = seq, 0 = empty)
    [51] Disordered mask (0 = disordered, 1 = ordered)
    [52] Evaluation mask (For CB513 dataset, 1 = eval, 0 = ignore)
    [53] ASA (isolated)
    [54] ASA (complexed)
    [55] RSA (isolated)
    [56] RSA (complexed)
    [57:65] Q8 GHIBESTC (Q8 -> Q3: HHHEECCC)
    [65:67] Phi+Psi
    :param data_dir: root path of the data
    :return:
    """
    pass


class NSP_Network(nn.Module):
    def __init__(self, init_channels, n_hidden):
        """ Initializes the model with the required layers
        Args:
            init_channels: The size of the incoming feature vector
            n_hidden: The amount of hidden neurons in the bidirectional lstm
        """
        super(NSP_Network, self).__init__()

        # CNN block
        self.conv1 = nn.Sequential(*[
            nn.Dropout(p=0.5),
            nn.Conv1d(in_channels=init_channels, out_channels=32, kernel_size=129, padding=64),
            nn.ReLU(),
        ])

        self.conv2 = nn.Sequential(*[
            nn.Dropout(p=0.5),
            nn.Conv1d(in_channels=init_channels, out_channels=32, kernel_size=257, padding=128),
            nn.ReLU(),
        ])

        self.batch_norm = BatchNorm1d(init_channels + 64)

        # LSTM block
        self.lstm = nn.LSTM(input_size=init_channels + 64, hidden_size=n_hidden, batch_first=True, \
                            num_layers=2, bidirectional=True, dropout=0.5)

        # add dropout to last layer
        self.lstm_dropout = nn.Dropout(p=0.5)

        # output block
        self.ss8 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=8),
            # nn.Softmax(),
        ])
        self.ss3 = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=3),
            # nn.Softmax(),
        ])
        self.disorder = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
            # nn.Softmax(),
        ])
        self.rsa = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=1),
            nn.Sigmoid()
        ])
        self.phi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
            nn.Tanh()
        ])
        self.psi = nn.Sequential(*[
            nn.Linear(in_features=n_hidden * 2, out_features=2),
            nn.Tanh()
        ])

    def forward(self, x, lengths, mask):
        """ Forwards the input through each layer in the model
        Args:
            x: input data containing sequences
            lengths: list containing each sequence length
            mask: data containing the sequence padding mask
        """

        _, length, _ = x.size()

        # calculate the residuals
        x = x.permute(0, 2, 1)

        r1 = self.conv1(x)
        r2 = self.conv2(x)

        # concatenate channels from residuals and input
        x = torch.cat([x, r1, r2], dim=1)

        # normalize
        # x = self.batch_norm(x, mask.unsqueeze(1))

        # calculate double layer bidirectional lstm
        x = x.permute(0, 2, 1)

        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, total_length=length, batch_first=True)

        x = self.lstm_dropout(x)

        # hidden neurons to classes
        ss8 = self.ss8(x)
        ss3 = self.ss3(x)
        disorder = self.disorder(x)
        rsa = self.rsa(x)
        phi = self.phi(x)
        psi = self.psi(x)

        return [ss8, ss3, disorder, rsa, phi, psi]


class MultiTaskLoss(nn.Module):
    """ Weighs multiple loss functions by considering the
        homoscedastic uncertainty of each task """

    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros((6)))

    def mse(self, outputs, labels, mask):
        loss = torch.square(outputs - labels) * mask
        return torch.sum(loss) / torch.sum(mask)

    def cross_entropy(self, outputs, labels, mask):
        labels = labels.clone()
        labels[mask == 0] = -999

        return nn.CrossEntropyLoss(ignore_index=-999)(outputs, labels.long())

    def ss8(self, outputs, labels, mask):
        labels = torch.argmax(labels[:, :, 7:15], dim=2)
        outputs = outputs[0].permute(0, 2, 1)

        return self.cross_entropy(outputs, labels, mask)

    def ss3(self, outputs, labels, mask):
        structure_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]).to(device)

        labels = torch.max(labels[:, :, 7:15] * structure_mask, dim=2)[0].long()
        outputs = outputs[1].permute(0, 2, 1)

        return self.cross_entropy(outputs, labels, mask)

    def disorder(self, outputs, labels, mask):
        # apply the disorder loss
        labels = labels[:, :, 1].unsqueeze(2)
        labels = torch.argmax(torch.cat([labels, 1 - labels], dim=2), dim=2)

        outputs = outputs[2].permute(0, 2, 1)

        return self.cross_entropy(outputs, labels, mask)

    def rsa(self, outputs, labels, mask):
        labels = labels[:, :, 5].unsqueeze(2)
        outputs = outputs[3]

        mask = mask.unsqueeze(2)

        return self.mse(outputs, labels, mask)

    def phi(self, outputs, labels, mask):
        labels = labels[:, :, 15].unsqueeze(2)
        outputs = outputs[4]

        mask = mask * (labels != 360).squeeze(2).int()
        mask = torch.cat(2 * [mask.unsqueeze(2)], dim=2)

        loss = self.mse(outputs.squeeze(2),
                        torch.cat((torch.sin(dihedral_to_radians(labels)), torch.cos(dihedral_to_radians(labels))),
                                  dim=2).squeeze(2), mask)
        return loss

    def psi(self, outputs, labels, mask):
        labels = labels[:, :, 16].unsqueeze(2)
        outputs = outputs[5]

        mask = mask * (labels != 360).squeeze(2).int()
        mask = torch.cat(2 * [mask.unsqueeze(2)], dim=2)

        loss = self.mse(outputs.squeeze(2),
                        torch.cat((torch.sin(dihedral_to_radians(labels)), torch.cos(dihedral_to_radians(labels))),
                                  dim=2).squeeze(2), mask)
        return loss

    def forward(self, outputs, labels, weighted=True):
        """ Forwarding of the multitaskloss input
        Args:
            outputs (torch.tensor): output data from model
            labels (torch.tensor): corresponding labels for the output
        """

        # filters
        zero_mask = labels[:, :, 0]
        disorder_mask = labels[:, :, 1]
        unknown_mask = labels[:, :, -1]

        # weighted losses
        ss8 = self.ss8(outputs, labels, zero_mask) * 1
        ss3 = self.ss3(outputs, labels, zero_mask) * 5
        dis = self.disorder(outputs, labels, zero_mask) * 5
        rsa = self.rsa(outputs, labels, zero_mask * disorder_mask * unknown_mask) * 100
        phi = self.phi(outputs, labels, zero_mask * disorder_mask * unknown_mask) * 5
        psi = self.psi(outputs, labels, zero_mask * disorder_mask * unknown_mask) * 5

        loss = torch.stack([ss8, ss3, dis, rsa, phi, psi])

        return loss.sum()


def dihedral_to_radians(angle):
    """ Converts angles to radians
    Args:
        angles (1D Tensor): vector with angle values
    """
    return angle * np.pi / 180


def init_model(initial_channels, hidden_neurons, learning_rate):
    """ Initializes a model, criterion and optimizer
    Args:
        initial_channels (int): amount of initial inputs for the model
        hidden_neurons (int): amount of hidden neurons in the model
    """
    nsp_net = NSP_Network(initial_channels, hidden_neurons)
    criterion = MultiTaskLoss()

    # enable cuda on model and criterion if possible
    if device.type != "cpu":
        nsp_net.cuda(device)
        criterion.cuda(device)

    # optimizer for model and criterion
    optimizer = optim.Adam([{"params": criterion.parameters()}, {"params": nsp_net.parameters()}], lr=learning_rate)

    return nsp_net, criterion, optimizer


def split_dataset(batch_size, dataset):
    """ Splits the dataset into train and validation
    Args:
        batch_size (int): size of each batch
        dataset (np.array): dataset containing training data
        validation_fraction (float): the size of the validation set as a fraction
    """

    num_train = len(dataset['data'])
    train_indices = np.array(range(num_train))
    validation_indices = np.random.choice(train_indices, int(num_train * 0.05), replace=False)

    train_indices = np.delete(train_indices, validation_indices)

    # subset the dataset
    train_idx, valid_idx = train_indices, validation_indices
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    training_set = DataLoader(NSPData(dataset), sampler=train_sampler, batch_size=batch_size)
    validation_set = DataLoader(NSPData(dataset), sampler=valid_sampler, batch_size=batch_size)

    return training_set, validation_set


def training(epochs, model, criterion, optimizer, dataset):
    # iterate over the dataset multiple times
    training_loss = []
    validation_loss = []

    train, test = dataset

    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(train), epochs=epochs)

    for epoch in range(epochs):
        print('Epoch:', epoch + 1, ' of ', epochs)

        # training of the model
        running_loss = 0.0
        for i, data in enumerate(train, 0):
            # move data tensors to GPU if possible
            inputs, labels, lengths = data
            inputs, labels = inputs.to(device), labels.to(device)

            padding = labels[:, :, 0]
            outputs = model(inputs, lengths, padding)

            # zero the parameter gradients
            optimizer.zero_grad()
            # backpropagation by custom criterion
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            # scheduler.step()

            running_loss += loss.item()

        training_loss.append(running_loss / len(train))
        print("Training loss: ", round(training_loss[epoch], 3))

        # validation of the model
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test, 0):
                # move data tensors to GPU if possible
                inputs, labels, lengths = data
                inputs, labels = inputs.to(device), labels.to(device)

                padding = labels[:, :, 0]
                outputs = model(inputs, lengths, padding)

                loss = criterion(outputs, labels)

                running_loss += loss.item()

        validation_loss.append(running_loss / len(test))
        print("Validation loss: ", round(validation_loss[epoch], 3))

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    print('Finished Training')

    return training_loss, validation_loss


def main():
    data_dir = "/home/share/huadjyin/home/s_sukui/02_data/05_meta/netsurf/2.0/"
    assert os.path.exists(data_dir), f"{data_dir} is not exit"

    # train_hhblits = np.load(data_dir + "Train_HHblits.npz")
    CB513_hhblits = np.load(data_dir + "CB513_HHblits.npz")
    # TS115_hhblits = np.load(data_dir + "TS115_HHblits.npz")
    # CASP12_hhblits = np.load(data_dir + "CASP12_HHblits.npz")
    #
    # train_mmseqs = np.load(data_dir + "Train_MMseqs.npz")
    # CB513_mmseqs = np.load(data_dir + "CB513_MMseqs.npz")
    # TS115_mmseqs = np.load(data_dir + "TS115_MMseqs.npz")
    # CASP12_mmseqs = np.load(data_dir + "CASP12_MMseqs.npz")
    epochs = 100

    initial_channels = 50
    hidden_neurons = 1024
    learning_rate = 5e-4
    batch_size = 3

    nsp_hhblits = init_model(initial_channels, hidden_neurons, learning_rate)
    train_hhblits = split_dataset(batch_size, CB513_hhblits)

    training_loss, evaluation_loss = training(
        epochs, nsp_hhblits[0],
        nsp_hhblits[1],
        nsp_hhblits[2],
        train_hhblits
    )

    nsp_mmseqs = init_model(initial_channels, hidden_neurons, learning_rate)


if __name__ == '__main__':
    main()
