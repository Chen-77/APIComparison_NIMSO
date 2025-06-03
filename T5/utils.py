import sys
import json
from tqdm import tqdm
from typing import List

import torch
from transformers import T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, precision_score, recall_score , f1_score, roc_auc_score


def read_json(data_file):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def train_one_epoch(model: T5ForConditionalGeneration, device, data_loader, epoch, optimizer, lr_scheduler):
    model.train()

    predicted_labels = torch.LongTensor([]).to(device)
    ground_truth_labels = torch.LongTensor([]).to(device)

    sum_loss = torch.zeros(1).to(device)  
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs.loss, outputs.logits  # logits.shape = torch.Size([batch size, 2, vocab size])
        pred_labels = torch.max(logits, dim=-1).indices

        ground_truth_labels = torch.cat([ground_truth_labels, labels[:, 0]])
        predicted_labels = torch.cat([predicted_labels, pred_labels[:, 0]])

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
        accuracy = accuracy_score(ground_truth_labels.tolist(), predicted_labels.tolist())

        loss.backward()

        sum_loss += loss.detach()
        avg_loss = sum_loss.item() / (step + 1)

        data_loader.desc = "[train epoch {}] lr: {:.5f}, loss: {:.3f}, acc: {:.3f}".format(
            epoch, optimizer.param_groups[0]["lr"], avg_loss, accuracy
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


@torch.no_grad()
def validate(model: T5ForConditionalGeneration, device, data_loader, label_id_list: List, epoch):
    model.eval()

    predicted_labels = torch.LongTensor([]).to(device)
    ground_truth_labels = torch.LongTensor([]).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=2)
        pred_labels = out[:, 1]

        ground_truth_labels = torch.cat([ground_truth_labels, labels[:, 0]])
        predicted_labels = torch.cat([predicted_labels, pred_labels])

        for pred in pred_labels:
            if pred not in label_id_list:
                print(f"The predicted label is not in label_id_list, its index is {pred}")

        accuracy = accuracy_score(ground_truth_labels.tolist(), predicted_labels.tolist())

        precision = precision_score(ground_truth_labels.tolist(), predicted_labels.tolist(), pos_label=label_id_list[1])
        recall = recall_score(ground_truth_labels.tolist(), predicted_labels.tolist(), pos_label=label_id_list[1])
        f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), pos_label=label_id_list[1])
        data_loader.desc = "[valid epoch {}] acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(
            epoch, accuracy, precision, recall, f1
        )

    return {
        'accuracy': accuracy,
        'precision':precision,
        'recall' : recall,
        'f1': f1
    }


@torch.no_grad()
def test(model: T5ForConditionalGeneration, device, data_loader, label_id_list: List):
    model.eval()

    predicted_labels = torch.LongTensor([]).to(device)
    ground_truth_labels = torch.LongTensor([]).to(device)
    predicted_probs = torch.FloatTensor([]).to(device) 
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)
        with torch.no_grad():
            #out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=2)
            #logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, 0, :]
            #out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=2)
            #probs = torch.softmax(logits, dim=-1)
            #pred_labels = out[:, 1]
            #pred_probs = probs[:, label_id_list[1]]  # Probability of the positive class
            
            decoder_input_ids = model._shift_right(labels)  
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits[:, 0, :]  
            pred_probs = torch.softmax(logits, dim=-1)[:, label_id_list[1]]  
            pred_labels = logits.argmax(dim=-1) 

        for pred in pred_labels:
            if pred not in label_id_list:
                print(f"The predicted label is not in label_id_list, its index is {pred}")

        ground_truth_labels = torch.cat([ground_truth_labels, labels[:, 0]])
        predicted_labels = torch.cat([predicted_labels, pred_labels])
        predicted_probs = torch.cat([predicted_probs, pred_probs])

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
        accuracy = accuracy_score(ground_truth_labels.tolist(), predicted_labels.tolist())
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
        precision = precision_score(ground_truth_labels.tolist(), predicted_labels.tolist(), pos_label=label_id_list[1])
        recall = recall_score(ground_truth_labels.tolist(), predicted_labels.tolist(), pos_label=label_id_list[1])
        f1 = f1_score(ground_truth_labels.tolist(), predicted_labels.tolist(), pos_label=label_id_list[1])
        #auc = roc_auc_score(ground_truth_labels.tolist(), predicted_probs.tolist())


        data_loader.desc = "[test] acc: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(
            accuracy, precision, recall, f1
        )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,

    }
