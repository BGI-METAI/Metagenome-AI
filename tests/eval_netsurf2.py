import os

import numpy as np
import torch
from torch import nn, optim
from torch.nn import BatchNorm1d, init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from transformers import T5EncoderModel, T5Tokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NSPData(Dataset):
    def __init__(self, dataset, tokenizer=None, indices=False):
        """ Constructor
        Args:
            X (np.array): The array that contains the training data
            y (np.array): The array that contains the test data
        """
        self.AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
        self.data = torch.tensor(dataset['data'][:, :, :50]).float()
        self.targets = torch.tensor(dataset['data'][:, :, 50:68]).float()
        self.lengths = torch.tensor([sum(target[:, 0] == 1) for target in self.targets])
        self.tokenizer = tokenizer

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

    def collate_fn(self, batch_sample):
        batch_seq, batch_label, attention_mask = [], [], []
        for X, y, lengths in batch_sample:
            seq_idx = X[:, :20].argmax(dim=1)
            seq = " ".join([self.AMINO_ACIDS[idx] for idx in seq_idx[:-1]])
            # seq_len = int(y[:, 0].sum())
            # seq = seq[:seq_len]
            batch_seq.append(seq)
            batch_label.append(y)
            attention_mask.append(y[:, 0].int())

        tokens = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=batch_seq,
            padding='longest',
            max_length=y.shape[0]
        )

        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.stack(attention_mask)
        # for idx, val in enumerate(batch_label):
        #     tag_tensor = torch.tensor(val)
        #     batch_label[idx] = torch.nn.functional.pad(tag_tensor, (0, input_ids.shape[1] - tag_tensor.shape[0]))
        batch_label = torch.stack(batch_label)
        return input_ids, attention_mask, batch_label


class MaskedBatchNorm1d(nn.Module):
    """ A masked version of nn.BatchNorm1d. Only tested for 3D inputs.
        Args:
            num_features: :math:`C` from an expected input of size
                :math:`(N, C, L)`
            eps: a value added to the denominator for numerical stability.
                Default: 1e-5
            momentum: the value used for the running_mean and running_var
                computation. Can be set to ``None`` for cumulative moving average
                (i.e. simple average). Default: 0.1
            affine: a boolean value that when set to ``True``, this module has
                learnable affine parameters. Default: ``True``
            track_running_stats: a boolean value that when set to ``True``, this
                module tracks the running mean and variance, and when set to ``False``,
                this module does not track such statistics and always uses batch
                statistics in both training and eval modes. Default: ``True``
        Shape:
            - Input: :math:`(N, C, L)`
            - input_mask: (N, 1, L) tensor of ones and zeros, where the zeros indicate locations not to use.
            - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 1))
            self.bias = nn.Parameter(torch.Tensor(num_features, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 1))
            self.register_buffer('running_var', torch.ones(num_features, 1))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, input_mask=None):
        # Calculate the masked mean and variance
        B, C, L = input.shape
        if input_mask is not None and input_mask.shape != (B, 1, L):
            raise ValueError('Mask should have shape (B, 1, L).')
        if C != self.num_features:
            raise ValueError('Expected %d channels but input has %d channels' % (self.num_features, C))
        if input_mask is not None:
            masked = input * input_mask
            n = input_mask.sum()
        else:
            masked = input
            n = B * L
        # Sum
        masked_sum = masked.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True)
        # Divide by sum of mask
        current_mean = masked_sum / n
        current_var = ((masked - current_mean) ** 2).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / n
        # Update running stats
        if self.track_running_stats and self.training:
            if self.num_batches_tracked == 0:
                self.running_mean = current_mean
                self.running_var = current_var
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var
            self.num_batches_tracked += 1
        # Norm the input
        if self.track_running_stats and not self.training:
            normed = (masked - self.running_mean) / (torch.sqrt(self.running_var + self.eps))
        else:
            normed = (masked - current_mean) / (torch.sqrt(current_var + self.eps))
        # Apply affine parameters
        if self.affine:
            normed = normed * self.weight + self.bias
        return normed


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

        self.batch_norm = MaskedBatchNorm1d(init_channels + 64)

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
        x = self.batch_norm(x, mask.unsqueeze(1))

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


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 32, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 3, kernel_size=(7, 1), padding=(3, 0))  # 7
        )

        self.dssp8_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 8, kernel_size=(7, 1), padding=(3, 0))
        )
        self.diso_classifier = torch.nn.Sequential(
            torch.nn.Conv2d(n_final_in, 2, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        x = self.elmo_feature_extractor(x)  # OUT: (B x 32 x L x 1)
        d3_Yhat = self.dssp3_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)
        d8_Yhat = self.dssp8_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 8)
        diso_Yhat = self.diso_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 2)
        return d3_Yhat, d8_Yhat, diso_Yhat


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


def load_sec_struct_model():
    checkpoint_dir = "/home/share/huadjyin/home/s_sukui/02_data/01_model/protT5/secstruct_checkpoint.pt"
    state = torch.load(checkpoint_dir)
    model = ConvNet()
    model.load_state_dict(state['state_dict'])
    model = model.eval()
    model = model.to(device)
    print('Loaded sec. struct. model from epoch: {:.1f}'.format(state['epoch']))

    return model


def get_T5_model():
    model_path_or_name = "/home/share/huadjyin/home/zhangchao5/weight/prot_t5_xl_half_uniref50-enc"
    model = T5EncoderModel.from_pretrained(model_path_or_name)
    model = model.to(device)  # move model to GPU
    model = model.eval()  # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained(model_path_or_name, do_lower_case=False)

    return model, tokenizer


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


def accuracy(pred, labels):
    """ Accuracy coefficient
    Args:
        inputs (1D Tensor): vector with predicted integer values
        labels (1D Tensor): vector with correct integer values
    """

    return (sum((pred == labels)) / len(labels)).item()


def evaluate_ss3(outputs, labels, mask):
    structure_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])

    labels = torch.max(labels[:, :, 7:15] * structure_mask, dim=2)[0].long()[mask == 1]
    outputs = torch.argmax(outputs, dim=2)[mask == 1]

    return accuracy(outputs, labels)


# def evaluation(model, dataset):
#     # iterate through the evaluation dataset
#     with torch.no_grad():
#         for data in dataset:
#             # move data tensors to GPU if possible
#             inputs, labels, lengths = data
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             # add evaluation mask to the masks
#             evaluation_mask = labels[:, :, 2]
#
#             zero_mask = labels[:, :, 0] * evaluation_mask
#
#             # get predictions
#             # model = model.to(device)
#             predictions = model(inputs, lengths, zero_mask)  # predict values
#
#             # move predictions to cpu
#             for i in range(len(predictions)):
#                 predictions[i] = predictions[i].cpu()
#
#             labels = labels.cpu()
#             zero_mask = zero_mask.cpu()
#
#             # evaluate
#             ss3 = evaluate_ss3(predictions[1], labels, zero_mask)
#
#             torch.cuda.empty_cache()
#
#             print("SS3 [Q3]: {}".format(round(ss3, 3)))

def evaluation(model, dataset):
    # iterate through the evaluation dataset
    sec_struct_model = load_sec_struct_model()
    ss3_scores = []
    with torch.no_grad():
        for data in dataset:
            # move data tensors to GPU if possible
            # input_ids, attention_mask, batch_label
            inputs, attention_mask, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # add evaluation mask to the masks
            evaluation_mask = labels[:, :, 2]

            zero_mask = labels[:, :, 0] * evaluation_mask

            # get predictions
            # model = model.to(device)
            embedding_repr = model(inputs, evaluation_mask)  # predict values

            d3_Yhat, d8_Yhat, diso_Yhat = sec_struct_model(embedding_repr.last_hidden_state)

            # move predictions to cpu
            # for i in range(len(predictions)):
            #     predictions[i] = predictions[i].cpu()
            d3_Yhat = d3_Yhat.cpu()

            labels = labels.cpu()
            zero_mask = zero_mask.cpu()

            # evaluate
            ss3 = evaluate_ss3(d3_Yhat, labels, zero_mask)
            print("SS3 [Q3]: {}".format(ss3, 3))
            ss3_scores.append(ss3)

        torch.cuda.empty_cache()

        print("SS3 [Q3]: {}".format(round(np.array(ss3_scores).mean(), 3)))


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

    # training_loss, evaluation_loss = training(
    #     epochs, nsp_hhblits[0],
    #     nsp_hhblits[1],
    #     nsp_hhblits[2],
    #     train_hhblits
    # )
    #
    # nsp_mmseqs = init_model(initial_channels, hidden_neurons, learning_rate)
    model, tokenizer = get_T5_model()
    print("Evaluation HHblits...")
    cb513_data = NSPData(CB513_hhblits, tokenizer=tokenizer)
    CB513_hhblits = DataLoader(
        dataset=cb513_data,
        batch_size=3,
        collate_fn=cb513_data.collate_fn
    )
    print("\nCB513")
    evaluation(model, CB513_hhblits)


if __name__ == '__main__':
    main()
