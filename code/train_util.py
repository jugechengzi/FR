import torch
import torch.nn.functional as F

from metric import get_sparsity_loss, get_continuity_loss, computer_pre_rec
import numpy as np


def train_teacher(teacher, optimizer, dataset, device):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = teacher(inputs, masks)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        soft_pred = F.softmax(logits, -1)
        _, pred = torch.max(soft_pred, dim=-1)

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
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy


def train_att(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        # zero_grad
        optimizer.zero_grad()

        # model
        att_score, rationales, cls_logits = model(inputs, masks)

        # update soft gradient
        cls_loss = args.cls_lambda * F.cross_entropy(cls_logits, labels)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales, masks, args.sparsity_percentage)
        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales)
        loss = cls_loss + sparsity_loss + continuity_loss

        loss.backward()
        optimizer.step()

        cls_soft_logits = torch.softmax(cls_logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

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
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy


def train(generator, classifier, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales
        rationales = generator(inputs, masks)

        # classifier
        # 此时mask应为mask * rationales[:,:,1]
        logits = classifier(inputs, masks, rationales[:, :, -1])

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss

        # update gradient
        loss.backward()
        optimizer.step()

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

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
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy


def train_share(model, optimizer, dataset, device, args,writer_epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l=0
    spar_l=0
    cont_l=0
    train_sp=[]
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales, logits = model(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)
        train_sp.append(
            (torch.sum(rationales[:, :, 1])/torch.sum( masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss

        # update gradient
        loss.backward()
        optimizer.step()

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        # TP predict 和 label 同时为1
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        # TN predict 和 label 同时为0
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        # FN predict 0 label 1
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        # FP predict 1 label 0
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l+=cls_loss.cpu().item()
        spar_l+=sparsity_loss.cpu().item()
        cont_l+=continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    writer_epoch[0].add_scalar('cls',cls_l,writer_epoch[1])
    writer_epoch[0].add_scalar('spar_l', spar_l, writer_epoch[1])
    writer_epoch[0].add_scalar('cont_l', cont_l, writer_epoch[1])
    writer_epoch[0].add_scalar('train_sp', np.mean(train_sp), writer_epoch[1])
    return precision, recall, f1_score, accuracy


def train_noshare(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0

    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales, logits = model(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss

        # update gradient
        loss.backward()
        optimizer.step()

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

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
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy


def train_skew(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        logits=model.train_skew(inputs,masks,labels)
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        cls_loss.backward()
        optimizer.step()
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

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
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy


def train_g_skew(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        logits=model.g_skew(inputs,masks)[:,0,:]
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        cls_loss.backward()
        optimizer.step()
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

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
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy
