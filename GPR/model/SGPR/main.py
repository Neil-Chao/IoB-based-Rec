import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import math
from GPR.model.TorchDataset.GroupDatasetWithUserHistory import UnloadedGroupDatasetWithUserHistory, UnloadedGroupDatasetForOthers
from GPR.model.SGPR.SGPR import SGPR
import GPR.config.SGPR as GPR_CONFIG
import pandas as pd

import GPR.config.constant as CONSTANT
from zoneinfo import ZoneInfo
from GPR.utils.EarlyStopping import EarlyStopping

output_dir = r'GPR/saved_model/' + GPR_CONFIG.SCENARIO

def get_dataloader(SCENARIO = GPR_CONFIG.SCENARIO):
    if SCENARIO == GPR_CONFIG.OSS:
        dataset_path = r"./GPR/dataset/OSS/oss_with_user_history.csv"
        df = pd.read_csv(dataset_path, header=0, usecols=[1,2,3,4,5,6])
        dataset = UnloadedGroupDatasetWithUserHistory(df.values)
        user_num = 19000
        item_num = 17
    
        shuffled_indices = np.random.permutation(len(dataset))
        train_idx = shuffled_indices[:int(GPR_CONFIG.TRAIN_PERCENTAGE*len(dataset))]
        val_idx = shuffled_indices[int(GPR_CONFIG.TRAIN_PERCENTAGE*len(dataset)):int((GPR_CONFIG.TRAIN_PERCENTAGE + GPR_CONFIG.VAL_PERCENTAGE) * len(dataset))]
        test_idx = shuffled_indices[int((GPR_CONFIG.TRAIN_PERCENTAGE + GPR_CONFIG.VAL_PERCENTAGE) * len(dataset)):]
        train_dataloader = DataLoader(dataset=dataset, batch_size=GPR_CONFIG.TRAIN_BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
        val_dataloader = DataLoader(dataset=dataset, batch_size=GPR_CONFIG.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))
        test_dataloader = DataLoader(dataset=dataset, batch_size=GPR_CONFIG.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(test_idx))

        return train_dataloader, val_dataloader, test_dataloader, user_num, item_num
    elif SCENARIO == GPR_CONFIG.CAMRA2011:
        df = pd.read_csv(r"./GPR/dataset/CAMRa2011/group_CAMRa2011.csv", header=0, usecols=[1,2,3,4,5,6])
        user_num = 602
        item_num = 7039
        dataset = UnloadedGroupDatasetForOthers(df.values)
        
        shuffled_indices = np.random.permutation(len(dataset))
        train_idx = shuffled_indices[:int(GPR_CONFIG.TRAIN_PERCENTAGE*len(dataset))]
        val_idx = shuffled_indices[int(GPR_CONFIG.TRAIN_PERCENTAGE*len(dataset)):int((GPR_CONFIG.TRAIN_PERCENTAGE + GPR_CONFIG.VAL_PERCENTAGE) * len(dataset))]
        test_idx = shuffled_indices[int((GPR_CONFIG.TRAIN_PERCENTAGE + GPR_CONFIG.VAL_PERCENTAGE) * len(dataset)):]
        train_dataloader = DataLoader(dataset=dataset, batch_size=GPR_CONFIG.TRAIN_BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
        val_dataloader = DataLoader(dataset=dataset, batch_size=GPR_CONFIG.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))
        test_dataloader = DataLoader(dataset=dataset, batch_size=GPR_CONFIG.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(test_idx))

        return train_dataloader, val_dataloader, test_dataloader, user_num, item_num
    else:
        user_num = 24631
        item_num = 19031
        df = pd.read_csv(r"./GPR/dataset/Meetup/group_meetup.csv", header=0, usecols=[1,2,3,4,5,6])
        dataset = UnloadedGroupDatasetForOthers(df.values)
        
        shuffled_indices = np.random.permutation(len(dataset))
        train_idx = shuffled_indices[:int(GPR_CONFIG.TRAIN_PERCENTAGE*len(dataset))]
        val_idx = shuffled_indices[int(GPR_CONFIG.TRAIN_PERCENTAGE*len(dataset)):int((GPR_CONFIG.TRAIN_PERCENTAGE + GPR_CONFIG.VAL_PERCENTAGE) * len(dataset))]
        test_idx = shuffled_indices[int((GPR_CONFIG.TRAIN_PERCENTAGE + GPR_CONFIG.VAL_PERCENTAGE) * len(dataset)):]
        train_dataloader = DataLoader(dataset=dataset, batch_size=GPR_CONFIG.TRAIN_BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
        val_dataloader = DataLoader(dataset=dataset, batch_size=GPR_CONFIG.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))
        test_dataloader = DataLoader(dataset=dataset, batch_size=GPR_CONFIG.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(test_idx))

        return train_dataloader, val_dataloader, test_dataloader, user_num, item_num

def loss_fn(p_score, n_score):
    return torch.mean(torch.pow((p_score - n_score - 1), 2), dim=0)

def train(model, dataloader, loss_fn, optimizer, logging, max_iter=100, lr=0.0001, lambda2=0.1, conv=1000):
    checkpoint_path = os.path.join(output_dir, "SGPR " + datetime.now(ZoneInfo('Asia/Shanghai')).strftime("%Y-%m-%d %H_%M_%S")+".pt")
    es = EarlyStopping(checkpoint_path, patience=16, verbose=True)
    
    loss_list = []
    for iter in range(max_iter):
        print(iter + 1)
        loss = 0
        for i, (group_user, group_behavior, user_history, target_user, target_behavior, negative_behavior) in enumerate(dataloader):
            if i % 100 == 0:
                print("batch index: " + str(i) + "; time: " + datetime.now(ZoneInfo('Asia/Shanghai')).strftime("%Y-%m-%d %H_%M_%S"))
            target_user = target_user.cuda()
            target_behavior = target_behavior.cuda()
            negative_behavior = negative_behavior.cuda()
            p_score = model(group_user, group_behavior, target_user, target_behavior)
            n_score = model(group_user, group_behavior, target_user, negative_behavior)
            tmp_loss = loss_fn(p_score, n_score)
            loss += tmp_loss
            optimal_obj = tmp_loss
            optimizer.zero_grad()
            optimal_obj.backward()
            optimizer.step()

        logging.info("Loss:{} at iter: {}".format(loss, iter + 1))
        recall, mrr, ndcg = evaluate(model, val_dataloader, GPR_CONFIG.RECOMMEND_NUM)
        logging.info("Recall@{}:{}. MRR@{}:{}. NDCG@{}:{}".format(GPR_CONFIG.RECOMMEND_NUM, recall, GPR_CONFIG.RECOMMEND_NUM,  mrr, GPR_CONFIG.RECOMMEND_NUM, ndcg))
        recall, mrr, ndcg = evaluate(model, val_dataloader, GPR_CONFIG.RECOMMEND_NUM_1)
        logging.info("Recall@{}:{}. MRR@{}:{}. NDCG@{}:{}".format(GPR_CONFIG.RECOMMEND_NUM_1, recall, GPR_CONFIG.RECOMMEND_NUM_1, mrr, GPR_CONFIG.RECOMMEND_NUM_1, ndcg))

        es([recall], model, optimizer, logging)
        if es.early_stop:
            logging.info("Early stopping")
            break
  
        recall, mrr, ndcg = evaluate(model, test_dataloader, GPR_CONFIG.RECOMMEND_NUM)
        logging.info("Test Result: Recall@{}:{}. MRR@{}:{}. NDCG@{}:{}".format(GPR_CONFIG.RECOMMEND_NUM, recall, GPR_CONFIG.RECOMMEND_NUM,  mrr, GPR_CONFIG.RECOMMEND_NUM, ndcg))


@torch.no_grad()
def evaluate(model, dataloader, behavior_num, N=5):
    count = 0
    r = 0
    mrr = 0
    ndcg = 0
    for i, (group_user, group_behavior, user_history, target_user, target_behavior, negative_behavior) in enumerate(dataloader):  
        target_user = target_user.cuda()
        count += target_behavior.shape[0]

        res = model.predict(group_user, group_behavior, target_user).squeeze()

        values, indices = torch.topk(res.cpu(), N)
        for i in range(indices.shape[0]):
            single_indices = indices[i]
            single_y = target_behavior[i]
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
    path = os.path.join("./GPR/logs/SGPR", datetime.now(ZoneInfo('Asia/Shanghai')).strftime("%Y-%m-%d %H_%M_%S")+".log")
    logging.basicConfig(filename=path, level=logging.DEBUG)
    logging.info("LEARNING_RATE: {}".format(GPR_CONFIG.LEARNING_RATE))
    logging.info("BATCH_SIZE: {}".format(GPR_CONFIG.BATCH_SIZE))
    logging.info("MAX_ITER: {}".format(GPR_CONFIG.MAX_ITER))
    logging.info("TRAIN_PERCENTAGE: {}".format(GPR_CONFIG.TRAIN_PERCENTAGE))
    logging.info("USER_EMB_SIZE: {}".format(GPR_CONFIG.USER_EMB_SIZE))
    logging.info("ITEM_EMB_SIZE: {}".format(GPR_CONFIG.ITEM_EMB_SIZE))
    logging.info("RECOMMEND_NUM: {}".format(GPR_CONFIG.RECOMMEND_NUM))
    logging.info("FACTOR: {}".format(GPR_CONFIG.FACTOR))
    logging.info("SCENARIO: {}".format(GPR_CONFIG.SCENARIO))


    train_dataloader, val_dataloader, test_dataloader, user_num, item_num = get_dataloader()
    if GPR_CONFIG.SCENARIO == GPR_CONFIG.OSS:
        similarity_vec = torch.load(r"GPR/saved_model/GGCF/AGGCF-OSS.pt")["user_emb.weight"].cuda()
    elif GPR_CONFIG.SCENARIO == GPR_CONFIG.MEETUP:
        similarity_vec = torch.load(r"GPR/saved_model/GGCF/AGGCF-MEETUP.pt")["user_emb.weight"].cuda()
    else:
        similarity_vec = torch.load(r"GPR/saved_model/GGCF/AGGCF-CAMRa2011.pt")["user_emb.weight"].cuda()

    model = SGPR(user_num, item_num, GPR_CONFIG.USER_EMB_SIZE, GPR_CONFIG.ITEM_EMB_SIZE, similarity_vec, GPR_CONFIG.FACTOR).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=GPR_CONFIG.LEARNING_RATE, momentum=0.9, weight_decay=0.001)
    train(model, train_dataloader, loss_fn, optimizer, logging, GPR_CONFIG.MAX_ITER, GPR_CONFIG.LEARNING_RATE)
    