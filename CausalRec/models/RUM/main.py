import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import math
import random
import logging
from zoneinfo import ZoneInfo

import CausalRec.config.RUM as config
from CausalRec.utils.SequenceDataset import SequenceDataset
from CausalRec.utils.preprocessing import music_instrument_input, digital_music_input, luxury_beauty_input
from CausalRec.utils.commonUtils import get_causal, set_seed
from CausalRec.utils.EarlyStopping import EarlyStopping
from CausalRec.models.RUM.RUM import RUM
from CausalRec.models.RUM.RUMC import RUMC
from CausalRec.models.RUM.RUMR import RUMR

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


def train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer, logging):
    checkpoint_path = os.path.join(config.OUTPUT_DIR, "RUM " + datetime.now(ZoneInfo('Asia/Shanghai')).strftime("%Y-%m-%d %H_%M_%S")+".pt")
    es = EarlyStopping(checkpoint_path, patience=16, verbose=True)
    for iter in range(config.MAX_ITER):
        model.train()
        print(iter + 1)
        loss = 0
        for i, (u, X, y, neg_y) in enumerate(train_dataloader):
            u = u.cuda()
            X = X.cuda()
            y = y.to(torch.int).cuda()
            neg_y = neg_y.to(torch.int).cuda()
            p_hat = model(u, X, y)
            n_hat = model(u, X, neg_y)
            tmp_loss = loss_fn(p_hat, n_hat)
            loss+=tmp_loss
            optimal_obj = tmp_loss
            optimizer.zero_grad()
            optimal_obj.backward()
            optimizer.step()
        
        model.eval()
        logging.info("Loss:{} at iter: {}".format(loss, iter + 1))
        recall, mrr, ndcg = causal_evaluate(model, val_dataloader, config.RECOMMEND_NUM)
        logging.info("Val Result: Recall@{}:{}. MRR@{}:{}. NDCG@{}:{}".format(config.RECOMMEND_NUM, recall, config.RECOMMEND_NUM,  mrr, config.RECOMMEND_NUM, ndcg))
        es([recall], model, optimizer, logging)
        if es.early_stop:
            logging.info("Early stopping")
            break
        recall, mrr, ndcg = evaluate(model, test_dataloader, config.RECOMMEND_NUM)
        logging.info("Test Result: Recall:{}. MRR:{}. NDCG:{}".format(recall, mrr, ndcg))
        

    


@torch.no_grad()
def causal_evaluate(model: RUM, val_dataloader, N=5):
    count = 0
    r = 0
    mrr = 0
    ndcg = 0
    for batch_index, (u, X, y, neg_y) in enumerate(val_dataloader):
        count += u.shape[0]
        u = u.cuda()
        X = X.cuda()
        y = y.to(torch.int)
        res = torch.zeros(u.shape[0], model.item_num + 1).cuda()
        for j in range(1, model.item_num + 1):
            res[:, j] = model(u, X, torch.tensor(j).repeat(u.shape[0]).cuda())
        values, indices = torch.topk(res.cpu(), N, dim=1)
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


@torch.no_grad()
def evaluate(model: RUM, val_dataloader, N=5):
    count = 0
    r = 0
    mrr = 0
    ndcg = 0
    for batch_index, (u, X, y, neg_y) in enumerate(val_dataloader):
        count += u.shape[0]
        u = u.cuda()
        X = X.cuda()
        y = y.to(torch.int)

        res = model.predict(u, X).transpose(-1, -2)
        res[:, 0] = 0
        values, indices = torch.topk(res.cpu(), N, dim=1)
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

def loss_fn(pos_y_hat, neg_y_hat):
    pos_loss = -torch.log(pos_y_hat + 1e-24)
    neg_loss = -torch.log(1-neg_y_hat + 1e-24)
    return torch.mean(pos_loss) + torch.mean(neg_loss)

if __name__ == "__main__":
    from datetime import datetime

    SEED = random.randint(1, 10000)
    set_seed(SEED)

    path = os.path.join("./CausalRec/logs/RUM", datetime.now().strftime("%Y-%m-%d %H_%M_%S")+".log")
    logging.basicConfig(filename=path, level=logging.DEBUG)
    logging.info("SEED: {}".format(SEED))
    logging.info("LEARNING_RATE: {}".format(config.LEARNING_RATE))
    logging.info("OPTIMIZER: {}".format(config.OPTIMIZER))
    logging.info("BATCH_SIZE: {}".format(config.BATCH_SIZE))
    logging.info("EMB_SIZE: {}".format(config.EMB_SIZE))
    logging.info("MAX_ITER: {}".format(config.MAX_ITER))
    logging.info("LAMBDA1: {}".format(config.LAMBDA2))
    logging.info("RECOMMEND_NUM: {}".format(config.RECOMMEND_NUM))
    logging.info("SCENARIO: {}".format(config.scenario))

    train_dataloader, val_dataloader, test_dataloader, item_num, user_num = get_dataloader(scenario = config.scenario)

    logging.info("------RUM------")
    model = RUM(item_num=item_num, user_num=user_num, emb_size=config.EMB_SIZE, alpha=config.ALPHA, beta=config.BETA).cuda()

    if config.OPTIMIZER == config.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.LAMBDA2, betas=(0.9, 0.999))
    elif config.OPTIMIZER == config.SGD:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.LEARNING_RATE, weight_decay=config.LAMBDA2)
    elif config.OPTIMIZER == config.RMSprop:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.LAMBDA2)
    else:
        pass

    train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer, logging)

    logging.info("------RUMC------")
    w, ae = get_causal(config)

    model = RUMC(item_num=item_num, user_num=user_num, emb_size=config.EMB_SIZE, ae=ae, w=w, t=config.THRESHOLD, gamma1=config.GAMMA1, gamma2=config.GAMMA2, alpha=config.ALPHA, beta=config.BETA).cuda()
    
    if config.OPTIMIZER == config.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.LAMBDA2)
    elif config.OPTIMIZER == config.SGD:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.LEARNING_RATE, weight_decay=config.LAMBDA2)
    elif config.OPTIMIZER == config.RMSprop:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.LAMBDA2)
    else:
        pass


    train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer, logging)

    logging.info("------RUMR------")
    model = RUMR(item_num=item_num, user_num=user_num, emb_size=config.EMB_SIZE, alpha=config.ALPHA, beta=config.BETA).cuda()

    if config.OPTIMIZER == config.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.LAMBDA2, betas=(0.9, 0.999))
    elif config.OPTIMIZER == config.SGD:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.LEARNING_RATE, weight_decay=config.LAMBDA2)
    elif config.OPTIMIZER == config.RMSprop:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.LAMBDA2)
    else:
        pass

    train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer, logging)