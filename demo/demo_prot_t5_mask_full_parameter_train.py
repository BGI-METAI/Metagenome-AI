import argparse
import sys
import random
import os

sys.path.insert(0, "..")

from proteinNER.classifier.model import ProtTransT5MaskPEFTModel, ProtTransT5MaskPretrainModel
from proteinNER.classifier.trainer import ProteinMaskTrainer


def register_parameters():
    parser = argparse.ArgumentParser(description='Protein Sequence Amino Acids Annotation Framework')
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/train.txt',
        help='the path of input dataset'
    )
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/test.txt',
        help='the path of input dataset'
    )
    parser.add_argument(
        '--label_dict_path',
        type=str,
        default='/home/share/huadjyin/home/zhangchao5/dataset/GENE3D_id2label.pkl',
    )
    parser.add_argument(
        '--output_home',
        type=str,
        default='/home/share/huadjyin/home/s_sukui/03_project/01_GeneLLM/Metagenome-AI/output',
    )
    parser.add_argument(
        '--model_path_or_name',
        type=str,
        default='/home/share/huadjyin/home/zhangchao5/weight/prot_t5_xl_half_uniref50-enc',
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


def worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ["WANDB_MODE"] = "offline"
    args = register_parameters()

    # prepare dataset
    train_files = []
    with open(args.train_data_path, 'r') as file:
        for line in file.readlines():
            train_files.extend([line.strip() for line in line.strip().split(' ') if line.endswith("pkl")])
    random.seed(args.seed)
    random.shuffle(train_files)

    test_files = []
    with open(args.test_data_path, 'r') as file:
        for line in file.readlines():
            test_files.extend([line.strip() for line in line.strip().split(' ') if line.endswith("pkl")])

    # initialize trainer class
    trainer = ProteinMaskTrainer(output_home=args.output_home, k=args.k)

    # register dataset
    trainer.register_dataset(
        data_files=train_files,
        mode='train',
        dataset_type='mask',
        batch_size=args.batch_size,
        model_name_or_path=args.model_path_or_name
    )

    trainer.register_dataset(
        data_files=test_files,
        mode='test',
        dataset_type='mask',
        batch_size=args.batch_size,
        model_name_or_path=args.model_path_or_name
    )

    model = ProtTransT5MaskPretrainModel(
        model_name_or_path=args.model_path_or_name,
        num_classes=args.num_classes + 1 if args.add_background else args.num_classes,
    )
    trainer.register_model(
        model=model,
        reuse=args.reuse,
        is_trainable=args.is_trainable,
        learning_rate=args.learning_rate,
        mode=args.mode
    )

    trainer.train(**vars(args))
    trainer.inference(**vars(args))


if __name__ == '__main__':
    worker()