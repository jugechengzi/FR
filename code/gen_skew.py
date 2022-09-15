import argparse
import os
import time

import torch

from beer import BeerData, BeerAnnotation
from embedding import get_embeddings,get_glove_embedding
from torch.utils.data import DataLoader

from model import GenEncShareModel,GenEncNoShareModel
from train_util import train_share, train_skew,train_g_skew
from validate_util import validate_share, validate_dev_sentence, validate_annotation_sentence, validate_rationales
from tensorboardX import SummaryWriter


def parse():
    parser = argparse.ArgumentParser(
        description="Distribution Matching Rationalization")

    # dataset parameters
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/beer',
                        help='Path of the dataset')
    parser.add_argument('--aspect',
                        type=int,
                        default=0,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--annotation_path',
                        type=str,
                        default='./data/beer/annotations.json',
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
                        default='./data/beer',
                        help='Dir. of pretrained embeddings [default: None]')
    parser.add_argument('--embedding_name',
                        type=str,
                        default='review+wiki.filtered.200.txt.gz',
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
                        default=200,
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
                        default=37,
                        help='Number of training epoch')
    parser.add_argument('--skew',
                        type=int,
                        default=100,
                        help='Number of training epoch')
    parser.add_argument('--skew_rate',
                        type=float,
                        default=0.7,
                        help='Continuity trade-off [default: 4.]')
    parser.add_argument('--skew_mask',
                        type=int,
                        default=10,
                        help='Number of training epoch')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0005,
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
        default=0.1,
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
                        type=int,
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
torch.manual_seed(20220405)

#####################
# parse arguments
#####################
args = parse()
args.tau=[args.tau]
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

######################
# device
######################
torch.cuda.set_device(int(args.gpu))
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(12252018)

######################
# load embedding
######################
if args.embedding_name=='review+wiki.filtered.200.txt.gz':
    pretrained_embedding, word2idx = get_embeddings(os.path.join(args.embedding_dir, args.embedding_name))
else:
    pretrained_embedding, word2idx = get_glove_embedding(os.path.join(args.embedding_dir, args.embedding_name))
args.vocab_size = len(word2idx)
args.pretrained_embedding = pretrained_embedding

######################
# load dataset
######################
train_data = BeerData(args.data_dir, args.aspect, 'train', word2idx, balance=True)

dev_data = BeerData(args.data_dir, args.aspect, 'dev', word2idx)

annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)

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

if args.share==1:
    p_para=list(map(id, model.cls_fc.parameters()))
    g_para = filter(lambda p: id(p) not in p_para, model.parameters())
elif args.share==0:
    gfc_para = list(map(id, model.generator.parameters()))
    g_para=model.generator.parameters()
else:
    print('please choose share of 0 or 1')

skew_opt=torch.optim.Adam(g_para, lr=args.lr)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

######################
# Training
######################
strat_time=time.time()
best_all = 0
f1_best_dev = [0]
best_dev_epoch = [0]
acc_best_dev = [0]
for e in range(args.skew):
    model.train()
    precision, recall, f1_score, accuracy = train_g_skew(model, skew_opt, train_loader, device, args)
    print('skew={},p={:.3f},r={:.3f},acc={:.3f}'.format(e,precision,recall,accuracy))
    if accuracy>args.skew_rate:
        break

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
    writer.add_scalar('time',time.time()-strat_time,epoch)
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    model.eval()
    print("Validate")
    with torch.no_grad():
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

        writer.add_scalar('dev_acc',accuracy,epoch)
        print("Validate Sentence")
        validate_dev_sentence(model, dev_loader, device,(writer,epoch))
        print("Annotation")
        annotation_results = validate_share(model, annotation_loader, device)
        print(
            "The annotation performance: sparsity: %.4f, precision: %.4f, recall: %.4f, f1: %.4f"
            % (100 * annotation_results[0], 100 * annotation_results[1],
               100 * annotation_results[2], 100 * annotation_results[3]))
        writer.add_scalar('f1',100 * annotation_results[3],epoch)
        writer.add_scalar('sparsity',100 * annotation_results[0],epoch)
        writer.add_scalar('p', 100 * annotation_results[1], epoch)
        writer.add_scalar('r', 100 * annotation_results[2], epoch)
        print("Annotation Sentence")
        validate_annotation_sentence(model, annotation_loader, device)
        print("Rationale")
        validate_rationales(model, annotation_loader, device,(writer,epoch))
        if accuracy>acc_best_dev[-1]:
            acc_best_dev.append(accuracy)
            best_dev_epoch.append(epoch)
            f1_best_dev.append(annotation_results[3])
        if best_all<annotation_results[3]:
            best_all=annotation_results[3]
    args.tau[0]*=args.taudecay
print(best_all)
print(acc_best_dev)
print(best_dev_epoch)
print(f1_best_dev)
