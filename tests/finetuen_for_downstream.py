import argparse
import logging
import os
import random

import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import SubsetRandomSampler, DataLoader
from transformers import T5EncoderModel, T5Tokenizer

# from proteinNER.base_module.dataset import NSPData
# from proteinNER.classifier.loss_fn import MultiTaskLoss
# from proteinNER.classifier.model import NetsurfConvModel
from tests.eval_netsurf2 import NSPData, NetsurfConvModel


def register_parameters():
    parser = argparse.ArgumentParser(description='Protein Sequence Amino Acids Annotation Framework')
    parser.add_argument(
        '--train_data_dir',
        type=str,
        default='/home/share/huadjyin/home/s_sukui/02_data/05_meta/netsurf/2.0/',
        help='the path of input dataset'
    )
    parser.add_argument(
        '--output_home',
        type=str,
        default='/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Metagenome-AI/output',
    )
    parser.add_argument(
        '--base_model_path_or_name',
        type=str,
        default='/home/share/huadjyin/home/zhangchao5/weight/prot_t5_xl_half_uniref50-enc',
        help='pretrianed pLM model path or name'
    )
    parser.add_argument(
        '--downstream_model_path_or_name',
        type=str,
        default='/home/share/huadjyin/home/s_sukui/02_data/01_model/protT5/secstruct_checkpoint.pt',
        help='pretrianed pLM model path or name'
    )
    parser.add_argument('--inference_length_threshold', type=int, default=50,
                        help='inference domain length threshold')  # 50
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--num_classes', type=int, default=28,
                        help='the number of categories')  # PFAM: 20794, GENE3D: 6595
    parser.add_argument('--add_background', action='store_true', help='add background type to the final categories')

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--loss_weight', type=float, default=1.)
    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument('--k', type=int, default=500, help='Gradient accumulation parameters')
    parser.add_argument('--max_token', type=int, default=512, help='max sequence')
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--is_trainable', action='store_true',
                        help='Whether the LoRA adapter should be trainable or not.')
    parser.add_argument('--mode', type=str, default="best", help='Whether the LoRA adapter should be trainable or not.')

    parser.add_argument('--user_name', type=str, default='sukui', help='wandb register parameter')
    parser.add_argument('--project', type=str, default='ProteinMaskPEFT', help='wandb project name')
    parser.add_argument('--group', type=str, default='MaskTrain', help='wandb group')

    return parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_base_model(base_model_path: str):
    """
    get base model, which including last_hidden_embedding
    :param base_model_path: model path
    :return: model and tokenizer
    """
    # base_model_path = "/home/share/huadjyin/home/zhangchao5/weight/prot_t5_xl_half_uniref50-enc"
    assert os.path.exists(base_model_path), f"{base_model_path} not exits"
    model = T5EncoderModel.from_pretrained(base_model_path)  # move model to GPU
    model = model.eval()  # set model to evaluation model
    model = model.to(device)  # set model to evaluation model
    tokenizer = T5Tokenizer.from_pretrained(base_model_path, do_lower_case=False)
    logging.info(f"loading base model from: {base_model_path}")

    return model, tokenizer


def get_downstream_model(download_model_path: str):
    """
    get downstream model, which input dimension was same with last_hidden_embedding
    :param download_model_path: model path
    :return: model
    """
    model = NetsurfConvModel()
    if os.path.exists(download_model_path):
        state = torch.load(download_model_path)
        model.load_state_dict(state['state_dict'])
        logging.info(f"load state ckpt from: {download_model_path}")
    model = model.to(device)
    model = model.eval()
    logging.info('Loaded sec. struct. model from epoch: {:.1f}'.format(state['epoch']))

    return model


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
        outputs = outputs[1].permute(0, 2, 1)

        return self.cross_entropy(outputs, labels, mask)

    def ss3(self, outputs, labels, mask):
        structure_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]).to(device)

        labels = torch.max(labels[:, :, 7:15] * structure_mask, dim=2)[0].long()
        outputs = outputs[0].permute(0, 2, 1)

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
                        torch.cat(
                            (torch.sin(self.dihedral_to_radians(labels)), torch.cos(self.dihedral_to_radians(labels))),
                            dim=2).squeeze(2), mask)
        return loss

    def psi(self, outputs, labels, mask):
        labels = labels[:, :, 16].unsqueeze(2)
        outputs = outputs[5]

        mask = mask * (labels != 360).squeeze(2).int()
        mask = torch.cat(2 * [mask.unsqueeze(2)], dim=2)

        loss = self.mse(outputs.squeeze(2),
                        torch.cat(
                            (torch.sin(self.dihedral_to_radians(labels)), torch.cos(self.dihedral_to_radians(labels))),
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
        # rsa = self.rsa(outputs, labels, zero_mask * disorder_mask * unknown_mask) * 100
        # phi = self.phi(outputs, labels, zero_mask * disorder_mask * unknown_mask) * 5
        # psi = self.psi(outputs, labels, zero_mask * disorder_mask * unknown_mask) * 5

        loss = torch.stack([ss3, ss8, dis])

        return loss.sum()

    @staticmethod
    def dihedral_to_radians(angle):
        """ Converts angles to radians
        Args:
            angles (1D Tensor): vector with angle values
        """
        return angle * np.pi / 180


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def training(epochs, base_model, downstream_model, criterion, optimizer, dataset):
    # iterate over the dataset multiple times
    training_loss = []
    validation_loss = []

    train, test = dataset
    early_stopping = EarlyStopping(patience=3, verbose=True)

    for epoch in range(epochs):
        print('Epoch:', epoch + 1, ' of ', epochs)

        # training of the model
        running_loss = 0.0
        for i, data in enumerate(train, 0):
            # move data tensors to GPU if possible
            inputs, attention_mask, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            attention_mask = attention_mask.to(device)

            embedding_repr = base_model(inputs, attention_mask)
            outputs = downstream_model(embedding_repr.last_hidden_state)

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
                inputs, attention_mask, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                attention_mask = attention_mask.to(device)

                embedding_repr = base_model(inputs, attention_mask)
                outputs = downstream_model(embedding_repr.last_hidden_state)

                loss = criterion(outputs, labels)

                running_loss += loss.item()

        validation_loss.append(running_loss / len(test))
        print("Validation loss: ", round(validation_loss[epoch], 3))

        early_stopping(validation_loss[epoch], downstream_model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    downstream_model.load_state_dict(torch.load('checkpoint.pt'))

    print('Finished Training')

    return training_loss, validation_loss


def split_dataset(batch_size, dataset, tokenizer=None):
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

    nsp_data = NSPData(dataset, tokenizer=tokenizer)
    training_set = DataLoader(nsp_data, sampler=train_sampler, batch_size=batch_size, collate_fn=nsp_data.collate_fn)
    validation_set = DataLoader(nsp_data, sampler=valid_sampler, batch_size=batch_size, collate_fn=nsp_data.collate_fn)

    return training_set, validation_set


def accuracy(pred, labels):
    """ Accuracy coefficient
    Args:
        inputs (1D Tensor): vector with predicted integer values
        labels (1D Tensor): vector with correct integer values
    """

    return (sum((pred == labels)) / len(labels)).item()


def fpr(pred, labels):
    """ False positive rate
    Args:
        inputs (1D Tensor): vector with predicted binary numeric values
        labels (1D Tensor): vector with correct binary numeric values
    """
    fp = sum((pred == 1) & (labels == 0))
    tn = sum((pred == 0) & (labels == 0))

    return (fp / (fp + tn)).item()


def mcc(pred, labels):
    """ Mathews correlation coefficient
    Args:
        inputs (1D Tensor): vector with predicted binary numeric values
        labels (1D Tensor): vector with correct binary numeric values
    """
    fp = sum((pred == 1) & (labels == 0))
    tp = sum((pred == 1) & (labels == 1))
    fn = sum((pred == 0) & (labels == 1))
    tn = sum((pred == 0) & (labels == 0))

    return ((tp * tn - fp * fn) / torch.sqrt(((tp + fp) * (fn + tn) * (tp + fn) * (fp + tn)).float())).item()


def evaluate_ss3(outputs, labels, mask):
    structure_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])

    labels = torch.max(labels[:, :, 7:15] * structure_mask, dim=2)[0].long()[mask == 1]
    outputs = torch.argmax(outputs, dim=2)[mask == 1]

    return accuracy(outputs, labels)


def evaluate_ss8(outputs, labels, mask):
    labels = torch.argmax(labels[:, :, 7:15], dim=2)[mask == 1]
    outputs = torch.argmax(outputs, dim=2)[mask == 1]

    return accuracy(outputs, labels)


def evaluate_dis(outputs, labels, mask, metric="fpr"):
    labels = labels[:, :, 1].unsqueeze(2)
    labels = torch.argmax(torch.cat([labels, 1 - labels], dim=2), dim=2)[mask == 1]
    outputs = torch.argmax(outputs, dim=2)[mask == 1]
    if metric == "fpr":
        return fpr(outputs, labels)
    else:
        return mcc(outputs, labels)


def evaluation(base_model, downstream_model, dataset):
    # iterate through the evaluation dataset
    ss3_scores = []
    ss8_scores = []
    diso_scores = []
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
            embedding_repr = base_model(inputs, evaluation_mask)  # predict values

            d3_Yhat, d8_Yhat, diso_Yhat = downstream_model(embedding_repr.last_hidden_state)
            d3_Yhat = d3_Yhat.cpu()
            d8_Yhat = d8_Yhat.cpu()
            diso_Yhat = diso_Yhat.cpu()

            labels = labels.cpu()
            zero_mask = zero_mask.cpu()

            # evaluate
            ss3 = evaluate_ss3(d3_Yhat, labels, zero_mask)
            print("SS3 [Q3]: {}".format(ss3, 3))
            ss3_scores.append(ss3)
            ss8 = evaluate_ss8(d8_Yhat, labels, zero_mask)
            print("SS8 [Q8]: {}".format(ss8, 3))
            ss8_scores.append(ss8)
            diso = evaluate_dis(diso_Yhat, labels, zero_mask)
            print("DISO [diso]: {}".format(diso, 3))
            diso_scores.append(diso)

        torch.cuda.empty_cache()

        print("SS3 [Q3]: {}".format(round(np.array(ss3_scores).mean(), 3)))
        print("SS8 [Q8]: {}".format(round(np.array(ss8_scores).mean(), 3)))
        print("DISO [DISO]: {}".format(round(np.array(diso_scores).mean(), 3)))


def main():
    args = register_parameters()
    logging.debug(f"parameters: {args}")
    logging.info("start....")
    random.seed(args.seed)
    batch_size = args.batch_size
    epoch = args.epoch

    base_model, tokenizer = get_base_model(args.base_model_path_or_name)
    if os.path.exists(args.downstream_model_path_or_name) is False:
        downstream_model = get_downstream_model(args.downstream_model_path_or_name)
        criterion = MultiTaskLoss()
        optimizer = optim.Adam([{"params": downstream_model.parameters()}])
        dataset = np.load(args.train_data_dir + "CB513_HHblits.npz")
        dataset = split_dataset(batch_size, dataset, tokenizer)

        training(0, base_model, downstream_model, criterion, optimizer, dataset=dataset)
    else:
        downstream_model = get_downstream_model(args.downstream_model_path_or_name)

    print("Evaluation HHblits...")
    CB513_hhblits = np.load(args.train_data_dir + "CB513_HHblits.npz")
    cb513_data = NSPData(CB513_hhblits, tokenizer=tokenizer)
    CB513_hhblits = DataLoader(
        dataset=cb513_data,
        batch_size=3,
        collate_fn=cb513_data.collate_fn
    )
    evaluation(base_model, downstream_model, CB513_hhblits)
    logging.info("finished..")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
