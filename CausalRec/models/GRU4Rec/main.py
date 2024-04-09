import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import math
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from zoneinfo import ZoneInfo
import random

import CausalRec.config.GRU4Rec as config
from CausalRec.utils.SequenceDataset import SequenceDataset
from CausalRec.utils.preprocessing import music_instrument_input, digital_music_input, luxury_beauty_input
from CausalRec.utils.EarlyStopping import EarlyStopping
from CausalRec.models.GRU4Rec.GRU4Rec import GRU4Rec
from CausalRec.models.GRU4Rec.GRU4RecC import GRU4RecC
from CausalRec.models.GRU4Rec.GRU4RecR import GRU4RecR
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

    return train_dataloader, val_dataloader,test_dataloader, item_num, user_num


def loss_fn(pos_y_hat, neg_y_hat):
    gap = (pos_y_hat - neg_y_hat).clone()
    loss = -torch.log(gap + 1e-24)
    return torch.mean(loss)

def train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer, logging):
    checkpoint_path = os.path.join(config.OUTPUT_DIR, "RUM " + datetime.now(ZoneInfo('Asia/Shanghai')).strftime("%Y-%m-%d %H_%M_%S")+".pt")
    es = EarlyStopping(checkpoint_path, patience=16, verbose=True)

    if type(model) == GRU4Rec:
        for iter in range(config.MAX_ITER):
            print(iter + 1)
            loss = 0
            model.train()
            for i, (u, X, y, neg_y) in enumerate(train_dataloader):
                u = u.cuda()
                X = X.cuda()
                y = y.to(torch.int64).cuda()
                neg_y = neg_y.to(torch.int64).cuda()
                y_hat = model(X)
                p_hat = torch.gather(y_hat, 1, y.unsqueeze(1)).squeeze()
                n_hat = torch.gather(y_hat, 1, neg_y.unsqueeze(1)).squeeze()
                tmp_loss = loss_fn(p_hat, n_hat)
                loss += tmp_loss
                optimal_obj = tmp_loss
                optimizer.zero_grad()
                optimal_obj.backward()
                optimizer.step()
            
            model.eval()
            logging.info("Loss:{} at iter: {}".format(loss, iter + 1))
            recall, mrr, ndcg = evaluate(model, val_dataloader, config.RECOMMEND_NUM)
            logging.info("Val Result: Recall@{}:{}. MRR@{}:{}. NDCG@{}:{}".format(config.RECOMMEND_NUM, recall, config.RECOMMEND_NUM,  mrr, config.RECOMMEND_NUM, ndcg))
            es([recall], model, optimizer, logging)
            if es.early_stop:
                logging.info("Early stopping")
                break
            recall, mrr, ndcg = evaluate(model, test_dataloader, config.RECOMMEND_NUM)
            logging.info("Test Result: Recall:{}. MRR:{}. NDCG:{}".format(recall, mrr, ndcg))
    
    else:
        for iter in range(config.MAX_ITER):
            loss = 0
            model.train()
            for i, (u, X, y, neg_y) in enumerate(train_dataloader):
                u = u.cuda()
                X = X.cuda()
                y = y.to(torch.int64).cuda()
                neg_y = neg_y.to(torch.int64).cuda()
                p_hat = model(X, y)
                p_hat = torch.gather(p_hat, 1, y.unsqueeze(1)).squeeze()
                n_hat = model(X, neg_y)
                n_hat = torch.gather(n_hat, 1, neg_y.unsqueeze(1)).squeeze()
                tmp_loss = loss_fn(p_hat, n_hat)
                loss += tmp_loss
                optimal_obj = tmp_loss
                optimizer.zero_grad()
                optimal_obj.backward()
                optimizer.step()
                
            model.eval()
            logging.info("Loss:{} at iter: {}".format(loss, iter + 1))
            recall, mrr, ndcg = evaluate(model, val_dataloader, config.RECOMMEND_NUM)
            logging.info("Val Result: Recall@{}:{}. MRR@{}:{}. NDCG@{}:{}".format(config.RECOMMEND_NUM, recall, config.RECOMMEND_NUM,  mrr, config.RECOMMEND_NUM, ndcg))
            es([recall], model, optimizer, logging)
            if es.early_stop:
                logging.info("Early stopping")
                break
            recall, mrr, ndcg = evaluate(model, test_dataloader, config.RECOMMEND_NUM)
            logging.info("Test Result: Recall:{}. MRR:{}. NDCG:{}".format(recall, mrr, ndcg))
            


@torch.no_grad()
def evaluate(model, val_dataloader, N=5):
    count = 0
    r = 0
    mrr = 0
    ndcg = 0
    if type(model) == GRU4Rec:
        for batch_index, (u, X, y, neg_y) in enumerate(val_dataloader):
            count += u.shape[0]
            X = X.cuda()
            y = y.to(torch.int)
            res = model(X)[:, 1:]
            values, indices = torch.topk(res.cpu(), N, dim=1)
            indices = indices + 1
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
    else:
        for batch_index, (u, X, y, neg_y) in enumerate(val_dataloader):
            count += u.shape[0]
            X = X.cuda()
            y = y.to(torch.int)
            res = model.predict(X)[:, 1:]
            values, indices = torch.topk(res.cpu(), N, dim=1)
            indices = indices + 1
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
    from datetime import datetime
    import logging
    SEED = random.randint(1, 10000)
    set_seed(SEED)

    path = os.path.join("./CausalRec/logs/GRU4Rec", datetime.now().strftime("%Y-%m-%d %H_%M_%S")+".log")
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

    logging.info("------GRU4Rec------")
    model = GRU4Rec(input_size=item_num, emb_size=config.EMB_SIZE, num_layers=config.NUM_LAYERS).cuda()
    if config.OPTIMIZER == config.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER == config.SGD:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.LEARNING_RATE)
    else:
        pass
    train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer, logging)


    logging.info("------GRU4RecC------")
    w, ae = get_causal(config)
    model = GRU4RecC(input_size=item_num, emb_size=config.EMB_SIZE, num_layers=config.NUM_LAYERS, ae=ae, w=w, t=config.THRESHOLD, gamma1=config.GAMMA1, gamma2=config.GAMMA2).cuda()
    if config.OPTIMIZER == config.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER == config.SGD:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.LEARNING_RATE)
    else:
        pass
    train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer, logging)


    logging.info("------GRU4RecR------")
    model = GRU4RecR(input_size=item_num, emb_size=config.EMB_SIZE, num_layers=config.NUM_LAYERS).cuda()
    if config.OPTIMIZER == config.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER == config.SGD:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.LEARNING_RATE)
    else:
        pass
    train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer, logging)