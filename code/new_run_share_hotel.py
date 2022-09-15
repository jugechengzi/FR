import argparse
import os
import time

import torch

from beer import BeerData, BeerAnnotation
from hotel import HotelData,HotelAnnotation
from embedding import get_embeddings,get_glove_embedding
from torch.utils.data import DataLoader

from model import GenEncShareModel,GenEncNoShareModel
from train_util import train_share
from validate_util import validate_share, validate_dev_sentence, validate_annotation_sentence, validate_rationales
from tensorboardX import SummaryWriter


def parse():
    parser = argparse.ArgumentParser(
        description="FR")

    # dataset parameters
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/hotel',
                        help='Path of the dataset')
    parser.add_argument('--aspect',
                        type=int,
                        default=0,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--seed',
                        type=int,
                        default=12252018,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--annotation_path',
                        type=str,
                        default='./data/hotel/annotations',
                        help='Path to the annotation')
    parser.add_argument('--max_length',
                        type=int,
                        default=256,
                        help='Max sequence length [default: 256]')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size [default: 100]')
    # pretrained embeddings
    parser.add_argument('--embedding_dir',
                        type=str,
                        default='./data/hotel/embeddings',
                        help='Dir. of pretrained embeddings [default: None]')
    parser.add_argument('--embedding_name',
                        type=str,
                        default='glove.6B.100d.txt',
                        help='File name of pretrained embeddings [default: None]')

    # model parameters
    parser.add_argument('--cell_type',
                        type=str,
                        default="GRU",
                        help='Cell type: LSTM, GRU [default: GRU]')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='RNN cell layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='Network Dropout')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help='Embedding dims [default: 100]')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=200,
                        help='RNN hidden dims [default: 100]')
    parser.add_argument('--num_class',
                        type=int,
                        default=2,
                        help='Number of predicted classes [default: 2]')

    # ckpt parameters
    parser.add_argument('--output_dir',
                        type=str,
                        default='./res',
                        help='Base dir of output files')

    # learning parameters
    parser.add_argument('--epochs',
                        type=int,
                        default=2,
                        help='Number of training epoch')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0006,
                        help='compliment learning rate [default: 1e-3]')
    parser.add_argument('--sparsity_lambda',
                        type=float,
                        default=12.,
                        help='Sparsity trade-off [default: 1.]')
    parser.add_argument('--continuity_lambda',
                        type=float,
                        default=10.,
                        help='Continuity trade-off [default: 4.]')
    parser.add_argument(
        '--sparsity_percentage',
        type=float,
        default=0.085,
        help='Regularizer to control highlight percentage [default: .2]')
    parser.add_argument(
        '--cls_lambda',
        type=float,
        default=0.9,
        help='lambda for classification loss')
    parser.add_argument('--gpu',
                        type=str,
                        default='1',
                        help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')
    parser.add_argument('--share',
                        type=int,
                        default=1,
                        help='')
    parser.add_argument('--tau',
                        type=float,
                        default=1,
                        help='')
    parser.add_argument('--taudecay',
                        type=float,
                        default=1,
                        help='')
    parser.add_argument(
        '--writer',
        type=str,
        default='./noname',
        help='Regularizer to control highlight percentage [default: .2]')
    args = parser.parse_args()
    return args


#####################
# set random seed
#####################


#####################
# parse arguments
#####################
args = parse()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
args.tau=[args.tau]
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

######################
# device
######################
torch.cuda.set_device(int(args.gpu))
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
# torch.cuda.manual_seed(12252018)

######################
# load embedding
######################
if args.embedding_name=='review+wiki.filtered.200.txt.gz':
    pretrained_embedding, word2idx = get_embeddings(os.path.join(args.embedding_dir, args.embedding_name))
    # load dataset
    ######################
    train_data = BeerData(args.data_dir, args.aspect, 'train', word2idx, balance=True)

    dev_data = BeerData(args.data_dir, args.aspect, 'dev', word2idx)

    annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)
else:
    pretrained_embedding, word2idx = get_glove_embedding(os.path.join(args.embedding_dir, args.embedding_name))
    train_data = HotelData(args.data_dir, args.aspect, 'train', word2idx, balance=True)

    dev_data = HotelData(args.data_dir, args.aspect, 'dev', word2idx)

    annotation_data = HotelAnnotation(args.annotation_path, args.aspect, word2idx)
args.vocab_size = len(word2idx)
args.pretrained_embedding = pretrained_embedding

######################


# shuffle and batch the dataset
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

dev_loader = DataLoader(dev_data, batch_size=args.batch_size)

annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)

######################
# load model
######################
writer=SummaryWriter(args.writer)
if args.share==1:
    model = GenEncShareModel(args)
elif args.share==0:
    model = GenEncNoShareModel(args)
else:
    print('please choose share of 0 or 1')
model.to(device)

######################
# Training
######################

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
best_all = 0
f1_best_dev = [0]
best_dev_epoch = [0]
acc_best_dev = [0]
######################
# Training
######################
strat_time=time.time()
for epoch in range(args.epochs):
    start = time.time()
    model.train()
    precision, recall, f1_score, accuracy = train_share(model, optimizer, train_loader, device, args,(writer,epoch))
    end = time.time()
    print('\nTrain time for epoch #%d : %f second' % (epoch, end - start))
    print("traning epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,
                                                                                                   precision, f1_score,
                                                                                                   accuracy))
    writer.add_scalar('train_acc',accuracy,epoch)
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    model.eval()
    print("Validate")
    with torch.no_grad():
        if (epoch+1)%1==0:
            for (batch, (inputs, masks, labels)) in enumerate(dev_loader):
                inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
                _, logits = model(inputs, masks)
                # pdb.set_trace()
                logits = torch.softmax(logits, dim=-1)
                _, pred = torch.max(logits, axis=-1)
                # compute accuarcy
                # TP predict 和 label 同时为1
                TP += ((pred == 1) & (labels == 1)).cpu().sum()
                # TN predict 和 label 同时为0
                TN += ((pred == 0) & (labels == 0)).cpu().sum()
                # FN predict 0 label 1
                FN += ((pred == 0) & (labels == 1)).cpu().sum()
                # FP predict 1 label 0
                FP += ((pred == 1) & (labels == 0)).cpu().sum()
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1_score = 2 * precision * recall / (recall + precision)
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            print("dev epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(epoch, recall,
                                                                                                       precision,
                                                                                                       f1_score, accuracy))
            # writer.add_scalar('dev_acc',accuracy,epoch)
            # print("Validate Sentence")
            # validate_dev_sentence(model, dev_loader, device, (writer,epoch))
        print("Annotation")
        annotation_results = validate_share(model, annotation_loader, device)
        print(
            "The annotation performance: sparsity: %.4f, precision: %.4f, recall: %.4f, f1: %.4f"
            % (100 * annotation_results[0], 100 * annotation_results[1],
               100 * annotation_results[2], 100 * annotation_results[3]))
        writer.add_scalar('f1',100 * annotation_results[3],epoch)
        # writer.add_scalar('sparsity',100 * annotation_results[0],epoch)
        # writer.add_scalar('p', 100 * annotation_results[1], epoch)
        # writer.add_scalar('r', 100 * annotation_results[2], epoch)
        # print("Annotation Sentence")
        # validate_annotation_sentence(model, annotation_loader, device)
        # print("Rationale")
        # validate_rationales(model, annotation_loader, device,(writer,epoch))
        if accuracy>acc_best_dev[-1]:
            acc_best_dev.append(accuracy)
            best_dev_epoch.append(epoch)
            f1_best_dev.append(annotation_results[3])
        if best_all<annotation_results[3]:
            best_all=annotation_results[3]
    args.tau[0]=args.tau[0]*args.taudecay
print(best_all)

print('best_dev_epoch={}'.format(best_dev_epoch[-1]))