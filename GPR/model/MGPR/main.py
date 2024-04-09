import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import socket 
hostname = socket.gethostname()


import GPR.config.MGPR as config
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from GPR.model.TorchDataset.GroupDatasetWithUserHistory import UnloadedGroupDatasetWithUserHistory, UnloadedGroupDatasetForOthers, UnloadedGroupDatasetWithNegative
from GPR.model.MGPR.MGPR import MGPR
import math
from pytz import timezone as ZoneInfo
from GPR.utils.EarlyStopping import EarlyStopping

output_dir = r'saved_model/' + config.scenario

def get_dataloader(SCENARIO = config.scenario):
    if SCENARIO == config.OSS:
        dataset_path = r"./GPR/dataset/OSS/oss_with_user_history.csv"
        df = pd.read_csv(dataset_path, header=0, usecols=[1,2,3,4,5,6])
        dataset = UnloadedGroupDatasetWithUserHistory(df.values)
        user_num = 19000
        item_num = 17
    
        shuffled_indices = np.random.permutation(len(dataset))
        train_idx = shuffled_indices[:int(config.TRAIN_PERCENTAGE*len(dataset))]
        val_idx = shuffled_indices[int(config.TRAIN_PERCENTAGE*len(dataset)):int((config.TRAIN_PERCENTAGE + config.VAL_PERCENTAGE) * len(dataset))]
        test_idx = shuffled_indices[int((config.TRAIN_PERCENTAGE + config.VAL_PERCENTAGE) * len(dataset)):]
        train_dataloader = DataLoader(dataset=dataset, batch_size=config.TRAIN_BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
        val_dataloader = DataLoader(dataset=dataset, batch_size=config.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))
        test_dataloader = DataLoader(dataset=dataset, batch_size=config.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(test_idx))

        return train_dataloader, val_dataloader, test_dataloader, user_num, item_num
    elif SCENARIO == config.CAMRA2011:
        df = pd.read_csv(r"./GPR/dataset/CAMRa2011/group_CAMRa2011.csv", header=0, usecols=[1,2,3,4,5,6])
        user_num = 602
        item_num = 7039
        dataset = UnloadedGroupDatasetWithNegative(df.values, item_num)
        
        shuffled_indices = np.random.permutation(len(dataset))
        train_idx = shuffled_indices[:int(config.TRAIN_PERCENTAGE*len(dataset))]
        val_idx = shuffled_indices[int(config.TRAIN_PERCENTAGE*len(dataset)):int((config.TRAIN_PERCENTAGE + config.VAL_PERCENTAGE) * len(dataset))]
        test_idx = shuffled_indices[int((config.TRAIN_PERCENTAGE + config.VAL_PERCENTAGE) * len(dataset)):]
        train_dataloader = DataLoader(dataset=dataset, batch_size=config.TRAIN_BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
        val_dataloader = DataLoader(dataset=dataset, batch_size=config.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))
        test_dataloader = DataLoader(dataset=dataset, batch_size=config.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(test_idx))

        return train_dataloader, val_dataloader, test_dataloader, item_num, user_num
    else:
        user_num = 24631
        item_num = 19031
        df = pd.read_csv(r"./GPR/dataset/Meetup/group_meetup.csv", header=0, usecols=[1,2,3,4,5,6])
        dataset = UnloadedGroupDatasetWithNegative(df.values, item_num+1)
        
        shuffled_indices = np.random.permutation(len(dataset))
        train_idx = shuffled_indices[:int(config.TRAIN_PERCENTAGE*len(dataset))]
        val_idx = shuffled_indices[int(config.TRAIN_PERCENTAGE*len(dataset)):int((config.TRAIN_PERCENTAGE + config.VAL_PERCENTAGE) * len(dataset))]
        test_idx = shuffled_indices[int((config.TRAIN_PERCENTAGE + config.VAL_PERCENTAGE) * len(dataset)):]
        train_dataloader = DataLoader(dataset=dataset, batch_size=config.TRAIN_BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
        val_dataloader = DataLoader(dataset=dataset, batch_size=config.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))
        test_dataloader = DataLoader(dataset=dataset, batch_size=config.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(test_idx))

        return train_dataloader, val_dataloader, test_dataloader, item_num, user_num

def pretrain_loss_fn(mask, y_hat, y):
    '''
    mask: B * S
    y_hat: B * S * (d+1) d是item数目
    y: B * S
    '''
    i_pos = []
    j_pos = []
    k_pos = []
    for i in range(mask.size(0)):
        for j in range(mask.size(1)):
            if mask[i][j] == 0:
                i_pos.append(i)
                j_pos.append(j)
                k_pos.append(y[i][j])
    return torch.sum(-torch.log(y_hat[i_pos, j_pos, k_pos])) / len(i_pos)

def loss_fn(y_hat, pos_y, neg_y):
    p_score = torch.gather(y_hat, 1, pos_y.unsqueeze(1)).squeeze().clone()
    n_score = (-torch.gather(y_hat, 1, neg_y.unsqueeze(1)).squeeze() + 1).clone()
    p_score[p_score < 0.000001]=0.000001
    n_score[n_score < 0.000001]=0.000001
    return torch.mean(-torch.log(p_score) - torch.log(n_score))

def pretrain(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer):
    logging.info("pretrain...")
    checkpoint_path = os.path.join(output_dir, "MGPR_pretrain " + datetime.now(ZoneInfo('Asia/Shanghai')).strftime("%Y-%m-%d %H_%M_%S")+".pt")
    es = EarlyStopping(checkpoint_path, patience=16, verbose=True)

    for iter in range(config.MAX_ITER):
        print(iter + 1)
        model.train()
        loss = 0
        for i, (g, gg, X, u, positive, negative) in enumerate(train_dataloader):
            if i % 100 == 0:
                print("batch index: " + str(i) + "; time: " + datetime.now(ZoneInfo('Asia/Shanghai')).strftime("%Y-%m-%d %H_%M_%S"))
            X = X.cuda()
            u = u.cuda()
            positive = positive.cuda()
            negative = negative.cuda()
            p_hat, mask = model.pretrain(g, gg, X, u, train=True)
            tmp_loss = loss_fn(mask, p_hat, X)
            loss += tmp_loss
            optimal_obj = tmp_loss
            optimizer.zero_grad()
            optimal_obj.backward()
            optimizer.step()
            
        logging.info("Loss:{} at iter: {}".format(loss, iter + 1))

        model.eval()
        recall, mrr, ndcg = preevaluate(model, val_dataloader, config.RECOMMEND_NUM)
        logging.info("Recall@{}:{}. MRR@{}:{}. NDCG@{}:{}".format(config.RECOMMEND_NUM, recall, config.RECOMMEND_NUM,  mrr, config.RECOMMEND_NUM, ndcg))
        recall, mrr, ndcg = preevaluate(model, val_dataloader, config.RECOMMEND_NUM_1)
        logging.info("Recall@{}:{}. MRR@{}:{}. NDCG@{}:{}".format(config.RECOMMEND_NUM_1, recall, config.RECOMMEND_NUM_1, mrr, config.RECOMMEND_NUM_1, ndcg))

        es([recall], model, optimizer, logging)
        if es.early_stop:
            logging.info("Early stopping")
            break

@torch.no_grad()
def preevaluate(model, val_dataloader, N=5):
    count = 0
    r = 0
    mrr = 0
    ndcg = 0
    for batch_index, (g, gg, X, u, y, negative) in enumerate(val_dataloader):
        count += y.shape[0]
        X = X.cuda()
        u = u.cuda()
        X[:-1]=0
        p_hat, mask = model.pretrain(g, gg, X, u)
        p_hat = p_hat[:, -1, 1:]
        values, indices = torch.topk(p_hat.cpu(), N, dim=1)
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

def train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer):
    checkpoint_path = os.path.join(output_dir, "MGPR " + datetime.now(ZoneInfo('Asia/Shanghai')).strftime("%Y-%m-%d %H_%M_%S")+".pt")
    es = EarlyStopping(checkpoint_path, patience=16, verbose=True)

    for iter in range(config.MAX_ITER):
        print(iter + 1)
        model.train()
        loss = 0
        for i, (g, gg, X, u, positive, negative) in enumerate(train_dataloader):
            if i % 100 == 0:
                print("batch index: " + str(i) + "; time: " + datetime.now(ZoneInfo('Asia/Shanghai')).strftime("%Y-%m-%d %H_%M_%S"))
            X = X.cuda()
            u = u.cuda()
            positive = positive.cuda()
            negative = negative.to(torch.int64).cuda()
            p_hat, mask = model(g, gg, X, u)
            tmp_loss = loss_fn(p_hat, positive, negative)
            loss += tmp_loss
            optimal_obj = tmp_loss
            optimizer.zero_grad()
            optimal_obj.backward()
            optimizer.step()
            
        logging.info("Loss:{} at iter: {}".format(loss, iter + 1))

        model.eval()
        recall, mrr, ndcg = evaluate(model, val_dataloader, config.RECOMMEND_NUM)
        logging.info("Recall@{}:{}. MRR@{}:{}. NDCG@{}:{}".format(config.RECOMMEND_NUM, recall, config.RECOMMEND_NUM,  mrr, config.RECOMMEND_NUM, ndcg))
        recall, mrr, ndcg = evaluate(model, val_dataloader, config.RECOMMEND_NUM_1)
        logging.info("Recall@{}:{}. MRR@{}:{}. NDCG@{}:{}".format(config.RECOMMEND_NUM_1, recall, config.RECOMMEND_NUM_1, mrr, config.RECOMMEND_NUM_1, ndcg))

        es([recall], model, optimizer, logging)
        if es.early_stop:
            logging.info("Early stopping")
            break
  
        recall, mrr, ndcg = evaluate(model, test_dataloader, config.RECOMMEND_NUM)
        logging.info("Test Result: Recall@{}:{}. MRR@{}:{}. NDCG@{}:{}".format(config.RECOMMEND_NUM, recall, config.RECOMMEND_NUM,  mrr, config.RECOMMEND_NUM, ndcg))


@torch.no_grad()
def evaluate(model, val_dataloader, N=5):
    count = 0
    r = 0
    mrr = 0
    ndcg = 0
    for batch_index, (g, gg, X, u, y, negative) in enumerate(val_dataloader):
        count += y.shape[0]
        X = X.cuda()
        u = u.cuda()
        p_hat, mask = model(g, gg, X, u)
        # p_hat, mask = model(g, gg, X, y)
        p_hat = p_hat[:, 1:]
        values, indices = torch.topk(p_hat.cpu(), N, dim=1)
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
    import os
    from datetime import datetime
    import pandas as pd
    import logging

    train_dataloader, val_dataloader, test_dataloader, item_num, user_num = get_dataloader(SCENARIO = config.scenario)

    model = MGPR(user_num, item_num, user_emb_size=32, item_emb_size=32, alpha=config.ALPHA, pooling=config.POOLING).cuda()


    path = os.path.join("./GPR/logs/MGPR", datetime.now(ZoneInfo('Asia/Shanghai')).strftime("%Y-%m-%d %H_%M_%S")+".log")

    logging.basicConfig(filename=path, level=logging.DEBUG)
    logging.info("BERT")
    logging.info("HOSTNAME: {}".format(hostname))
    logging.info("LEARNING_RATE: {}".format(config.LEARNING_RATE))
    logging.info("OPTIMIZER: {}".format(config.OPTIMIZER))
    logging.info("BATCH_SIZE: {}".format(config.TRAIN_BATCH_SIZE))
    logging.info("EMB_SIZE: {}".format(config.EMB_SIZE))
    logging.info("MAX_ITER: {}".format(config.MAX_ITER))
    logging.info("LAMBDA1: {}".format(config.LAMBDA2))
    logging.info("ALPHA: {}".format(config.ALPHA))
    logging.info("POOLING: {}".format(config.POOLING))
    logging.info("SCENARIO: {}".format(config.scenario))
    
    
    if config.OPTIMIZER == config.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999))
    elif config.OPTIMIZER == config.SGD:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.LEARNING_RATE)
    elif config.OPTIMIZER == config.RMSprop:
        optimizer = torch.optim.RMSprop(model.parameters(), momentum=0.9, lr=config.LEARNING_RATE)
    else:
        pass

    pretrain(model, train_dataloader, val_dataloader, test_dataloader, pretrain_loss_fn, optimizer)

    train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer)