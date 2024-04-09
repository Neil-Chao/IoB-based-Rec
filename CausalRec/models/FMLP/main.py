import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import torch
import numpy as np
import math
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
from pytz import timezone as ZoneInfo

import CausalRec.config.FMLP as config
from CausalRec.utils.SequenceDataset import SequenceDataset
from CausalRec.utils.preprocessing import music_instrument_input, digital_music_input, luxury_beauty_input
from CausalRec.utils.EarlyStopping import EarlyStopping
from CausalRec.models.FMLP.FMLP import FMLP
from CausalRec.models.FMLP.FMLPC import FMLPC
from CausalRec.models.FMLP.FMLPR import FMLPR
from CausalRec.utils.commonUtils import get_causal, set_seed

def get_dataloader(scenario = config.scenario):
    if scenario == config.MUSIC_INSTRUMENT:
        datas, item_num, user_num = music_instrument_input()
    elif scenario == config.DIGITAL_MUSIC:
        datas, item_num, user_num = digital_music_input()
    elif scenario == config.LUXURY_BEAUTY:
        datas, item_num, user_num = luxury_beauty_input()

    shuffled_indices = np.random.permutation(len(datas))
    train_idx = shuffled_indices[:int(config.TRAIN_PERCENTAGE*len(datas))]
    val_idx = shuffled_indices[int(config.TRAIN_PERCENTAGE*len(datas)):int((config.TRAIN_PERCENTAGE + config.VAL_PERCENTAGE) * len(datas))]
    test_idx = shuffled_indices[int((config.TRAIN_PERCENTAGE + config.VAL_PERCENTAGE) * len(datas)):]
    dataset = SequenceDataset(datas, item_num + 1)
    train_dataloader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
    val_dataloader = DataLoader(dataset=dataset, batch_size=config.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))
    test_dataloader = DataLoader(dataset=dataset, batch_size=config.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(test_idx))

    return train_dataloader, val_dataloader, test_dataloader, item_num, user_num


def cross_entropy(model, seq_out, pos_ids, neg_ids):
    pos_emb = model.item_embeddings(pos_ids)
    neg_emb = model.item_embeddings(neg_ids)
    seq_emb = seq_out[:, -1, :]
    pos_logits = torch.sum(pos_emb * seq_emb, -1)
    neg_logits = torch.sum(neg_emb * seq_emb, -1)
    loss = torch.mean(
        - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
        torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
    )
    return loss


def train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer, logging):
    checkpoint_path = os.path.join(config.OUTPUT_DIR, "FMLP " + datetime.now(ZoneInfo('Asia/Shanghai')).strftime("%Y-%m-%d %H_%M_%S")+".pt")
    es = EarlyStopping(checkpoint_path, patience=16, verbose=True)
    for iter in range(config.MAX_ITER):
        model.train()
        print(iter + 1)
        loss = 0
        for i, (u, X, y, neg_y) in enumerate(train_dataloader):
            u = u.cuda()
            X = X.cuda()
            y = y.to(torch.int64).cuda()
            neg_y = neg_y.to(torch.int64).cuda()
            sequence_output = model(X, y)
            tmp_loss = loss_fn(model, sequence_output, y, neg_y)
            loss += tmp_loss
            optimal_obj = tmp_loss
            optimizer.zero_grad()
            optimal_obj.backward()
            optimizer.step()
            
        logging.info("Loss:{} at iter: {}".format(loss, iter + 1))
        
        model.eval()
        recall, mrr, ndcg = evaluate(model, val_dataloader, config.RECOMMEND_NUM)
        logging.info("Recall@{}:{}. MRR@{}:{}. NDCG@{}:{}".format(config.RECOMMEND_NUM, recall, config.RECOMMEND_NUM,  mrr, config.RECOMMEND_NUM, ndcg))
        es([recall], model, optimizer, logging)
        if es.early_stop:
            logging.info("Early stopping")
            break
        recall, mrr, ndcg = evaluate(model, test_dataloader, config.RECOMMEND_NUM)
        logging.info("Test Result: Recall:{}. MRR:{}. NDCG:{}".format(recall, mrr, ndcg))


@torch.no_grad()
def evaluate(model, val_dataloader, N=5):

    def predict_full(seq_out):
        # [item_num hidden_size]
        test_item_emb = model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    count = 0
    r = 0
    mrr = 0
    ndcg = 0

    for batch_index, (u, X, y, neg_y) in enumerate(val_dataloader):
        count += u.shape[0]
        X = X.cuda()
        y = y.to(torch.int)
        recommend_output = model.predict(X)
        recommend_output = recommend_output[:, -1, :]# 推荐的结果
        res = predict_full(recommend_output)
        values, indices = torch.topk(res.cpu(), N)
        # indices = indices + 1
        for i in range(indices.shape[0]):
            single_indices = indices[i]
            single_y = y[i]
            if single_y in single_indices:
                r += 1
                for i in range(single_indices.shape[0]):
                    if single_indices[i] == single_y:
                        mrr += 1 / (i + 1)
                        ndcg += 1 / (math.log(i + 1) + 1)
                        break
    
    return r / count, mrr / count, ndcg / count

if __name__ == "__main__":
    import os
    from datetime import datetime
    import logging

    
    SEED = random.randint(1, 10000)
    set_seed(SEED)
    
    path = os.path.join("./CausalRec/logs/FMLP", datetime.now().strftime("%Y-%m-%d %H_%M_%S")+".log")
    logging.basicConfig(filename=path, level=logging.DEBUG)
    logging.info("SEED: {}".format(SEED))
    logging.info("LEARNING_RATE: {}".format(config.LEARNING_RATE))
    logging.info("OPTIMIZER: {}".format(config.OPTIMIZER))
    logging.info("BATCH_SIZE: {}".format(config.BATCH_SIZE))
    logging.info("EMB_SIZE: {}".format(config.EMB_SIZE))
    logging.info("MAX_ITER: {}".format(config.MAX_ITER))
    logging.info("LAMBDA1: {}".format(config.LAMBDA2))
    logging.info("SCENARIO: {}".format(config.scenario))
    
    train_dataloader, val_dataloader, test_dataloader, item_num, user_num = get_dataloader(scenario = config.scenario)

    logging.info("------FMLP------")
    model = FMLP(item_num+1, hidden_size=config.EMB_SIZE, max_seq_length=5, hidden_dropout_prob=0.5, initializer_range=0.02,  num_hidden_layers=2, no_filters=False, num_attention_heads=4, attention_probs_dropout_prob=0.5, hidden_act="gelu").cuda()
    if config.OPTIMIZER == config.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999), weight_decay=0)
    elif config.OPTIMIZER == config.SGD:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.LEARNING_RATE)
    else:
        pass
    train(model, train_dataloader, val_dataloader, test_dataloader, cross_entropy, optimizer, logging)

    logging.info("------FMLPC------")
    w, ae = get_causal(config)
    model = FMLPC(item_num+1, ae, w, hidden_size=config.EMB_SIZE, max_seq_length=5, hidden_dropout_prob=0.5, initializer_range=0.02,  num_hidden_layers=2, no_filters=False, num_attention_heads=4, attention_probs_dropout_prob=0.5, hidden_act="gelu").cuda()
    if config.OPTIMIZER == config.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER == config.SGD:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.LEARNING_RATE)
    else:
        pass
    train(model, train_dataloader, val_dataloader, test_dataloader, cross_entropy, optimizer, logging)

    logging.info("------FMLPR------")
    model = FMLPR(item_num+1, hidden_size=config.EMB_SIZE, max_seq_length=5, hidden_dropout_prob=0.5, initializer_range=0.02,  num_hidden_layers=2, no_filters=False, num_attention_heads=4, attention_probs_dropout_prob=0.5, hidden_act="gelu").cuda()
    if config.OPTIMIZER == config.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER == config.SGD:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.LEARNING_RATE)
    else:
        pass
    train(model, train_dataloader, val_dataloader, test_dataloader, cross_entropy, optimizer, logging)