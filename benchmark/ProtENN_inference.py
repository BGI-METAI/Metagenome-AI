import json
import numpy as np
import tensorflow.compat.v1 as tf
import tqdm
import glob
import pandas as pd
import time

# Suppress noisy log messages.
from tensorflow.python.util import deprecation

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

deprecation._PRINT_DEPRECATION_WARNINGS = False

AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y'
]
_PFAM_GAP_CHARACTER = '.'


def residues_to_one_hot(amino_acid_residues):
    """Given a sequence of amino acids, return one hot array.

    Supports ambiguous amino acid characters B, Z, and X by distributing evenly
    over possible values, e.g. an 'X' gets mapped to [.05, .05, ... , .05].

    Supports rare amino acids by appropriately substituting. See
    normalize_sequence_to_blosum_characters for more information.

    Supports gaps and pads with the '.' and '-' characters; which are mapped to
    the zero vector.

    Args:
      amino_acid_residues: string. consisting of characters from
        AMINO_ACID_VOCABULARY

    Returns:
      A numpy array of shape (len(amino_acid_residues),
       len(AMINO_ACID_VOCABULARY)).

    Raises:
      ValueError: if sparse_amino_acid has a character not in the vocabulary + X.
    """
    to_return = []
    normalized_residues = amino_acid_residues.replace('U', 'C').replace('O', 'X')
    for char in normalized_residues:
        if char in AMINO_ACID_VOCABULARY:
            to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
            to_append[AMINO_ACID_VOCABULARY.index(char)] = 1.
            to_return.append(to_append)
        elif char == 'B':  # Asparagine or aspartic acid.
            to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
            to_append[AMINO_ACID_VOCABULARY.index('D')] = .5
            to_append[AMINO_ACID_VOCABULARY.index('N')] = .5
            to_return.append(to_append)
        elif char == 'Z':  # Glutamine or glutamic acid.
            to_append = np.zeros(len(AMINO_ACID_VOCABULARY))
            to_append[AMINO_ACID_VOCABULARY.index('E')] = .5
            to_append[AMINO_ACID_VOCABULARY.index('Q')] = .5
            to_return.append(to_append)
        elif char == 'X':
            to_return.append(
                np.full(len(AMINO_ACID_VOCABULARY), 1. / len(AMINO_ACID_VOCABULARY)))
        elif char == _PFAM_GAP_CHARACTER:
            to_return.append(np.zeros(len(AMINO_ACID_VOCABULARY)))
        else:
            raise ValueError('Could not one-hot code character {}'.format(char))
    return np.array(to_return)


def pad_one_hot_sequence(sequence: np.ndarray,
                         target_length: int) -> np.ndarray:
    """Pads one hot sequence [seq_len, num_aas] in the seq_len dimension."""
    sequence_length = sequence.shape[0]
    pad_length = target_length - sequence_length
    if pad_length < 0:
        raise ValueError(
            'Cannot set a negative amount of padding. Sequence length was {}, target_length was {}.'
            .format(sequence_length, target_length))
    pad_values = [[0, pad_length], [0, 0]]
    return np.pad(sequence, pad_values, mode='constant')  # 补psd_length个


def batch_iterable(iterable, batch_size):
    """Yields batches from an iterable.

    If the number of elements in the iterator is not a multiple of batch size,
    the last batch will have fewer elements.

    Args:
      iterable: a potentially infinite iterable.
      batch_size: the size of batches to return.

    Yields:
      array of length batch_size, containing elements, in order, from iterable.

    Raises:
      ValueError: if batch_size < 1.
    """
    if batch_size < 1:
        raise ValueError(
            'Cannot have a batch size of less than 1. Received: {}'.format(
                batch_size))

    current = []
    for item in iterable:
        if len(current) == batch_size:
            yield current
            current = []
        current.append(item)

    # Prevent yielding an empty batch. Instead, prefer to end the generation.
    if current:
        yield current


def infer(batch):
    seq_lens = [len(seq) for seq in batch]
    one_hots = [residues_to_one_hot(seq) for seq in batch]
    padded_sequence_inputs = [pad_one_hot_sequence(seq, max(seq_lens)) for seq in one_hots]
    with graph.as_default():
        return sess.run(
            top_pick_signature_tensor_name,
            {
                sequence_input_tensor_name: padded_sequence_inputs,
                sequence_lengths_input_tensor_name: seq_lens,
            })


if __name__ == '__main__':
    T1 = time.time()
    # Load the model into TensorFlow
    sess = tf.Session()
    graph = tf.Graph()

    with graph.as_default():
        saved_model = tf.saved_model.load(sess, ['serve'],
                                          './5356760/trn-_cnn_random__random_sp_gpu-cnn_for_random_pfam-5356760')

    # Load tensors for class prediction
    top_pick_signature = saved_model.signature_def['serving_default']
    top_pick_signature_tensor_name = top_pick_signature.outputs['output'].name

    sequence_input_tensor_name = saved_model.signature_def['confidences'].inputs['sequence'].name
    sequence_lengths_input_tensor_name = saved_model.signature_def['confidences'].inputs['sequence_length'].name

    # Load label_vocab
    with open('./trained_model_pfam_32.0_vocab.json') as f:
        vocab = json.loads(f.read())

    test_path = '/home/share/huadjyin/home/zhangkexin2/data/protENN/data/ourdata/pfam_split_fragment_out.test.txt'
    test_df = pd.read_csv(test_path, sep=' ')
    # infer process
    test_df = test_df.sort_values('sequence', key=lambda col: [len(c) for c in col])
    inference_results = []
    batches = list(batch_iterable(test_df.sequence, 32))
    for seq_batch in tqdm.tqdm(batches, position=0):
        t1 = time.time()
        inference_results.extend(infer(seq_batch))
        t2 = time.time()
        print("batch time", round(t2 - t1, 5), 'seconds')
    test_df['predicted_label'] = [vocab[i] for i in inference_results]
    test_df['true_label'] = test_df.family_accession.apply(lambda s: s.split('.')[0])
    T2 = time.time()
    print("total time", round(T2 - T1, 5), 'seconds')
    result_df = test_df[['predicted_label', 'true_label']]
    result_df.to_csv(
        '/home/share/huadjyin/home/zhangkexin2/code/protENN_code/protENN_model/results/PFAM_test_chunk0_predicted_label.csv',
        index=False)

    pred_list = np.array(result_df['predicted_label'])
    target_list = np.array(result_df['true_label'])
    f1_macro_value = f1_score(y_true=target_list, y_pred=pred_list, average='macro')
    acc_value = accuracy_score(y_true=target_list, y_pred=pred_list)
    recall_macro_value = recall_score(y_true=target_list, y_pred=pred_list, average='macro')
    pre_macro_value = precision_score(y_true=target_list, y_pred=pred_list, average='macro')
    f1_micro_value = f1_score(y_true=target_list, y_pred=pred_list, average='micro')
    recall_micro_value = recall_score(y_true=target_list, y_pred=pred_list, average='micro')
    pre_micro_value = precision_score(y_true=target_list, y_pred=pred_list, average='micro')
    print(acc_value, f1_macro_value, recall_macro_value, pre_macro_value, f1_micro_value, recall_micro_value,
          pre_micro_value)
