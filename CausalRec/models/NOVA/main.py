import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

import torch
import numpy as np
import math
from torch.utils.data import DataLoader, SubsetRandomSampler
from pytz import timezone as ZoneInfo

from CausalRec.utils.SequenceDataset import NOVADataset
from CausalRec.utils.preprocessing import music_instrument_input_of_nova, digital_music_input_of_nova, luxury_beauty_input_of_nova
from CausalRec.utils.EarlyStopping import EarlyStopping
import CausalRec.config.NOVA as config
from CausalRec.models.NOVA.NOVA import NOVA
from CausalRec.models.NOVA.NOVAC import NOVAC
from CausalRec.models.NOVA.NOVAR import NOVAR
from CausalRec.utils.commonUtils import get_causal

def get_dataloader(scenario = config.scenario):
    if scenario == config.MUSIC_INSTRUMENT:
        datas, item_num, user_num = music_instrument_input_of_nova(config.BERT_MAX_LEN)
    elif scenario == config.DIGITAL_MUSIC:
        datas, item_num, user_num = digital_music_input_of_nova(config.BERT_MAX_LEN)
    elif scenario == config.LUXURY_BEAUTY:
        datas, item_num, user_num = luxury_beauty_input_of_nova(config.BERT_MAX_LEN)
    else:
        pass

    shuffled_indices = np.random.permutation(len(datas))
    train_idx = shuffled_indices[:int(config.TRAIN_PERCENTAGE*len(datas))]
    val_idx = shuffled_indices[int(config.TRAIN_PERCENTAGE*len(datas)):int((config.TRAIN_PERCENTAGE + config.VAL_PERCENTAGE) * len(datas))]
    test_idx = shuffled_indices[int((config.TRAIN_PERCENTAGE + config.VAL_PERCENTAGE) * len(datas)):]
    dataset = NOVADataset(datas)
    train_dataloader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
    val_dataloader = DataLoader(dataset=dataset, batch_size=config.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))
    test_dataloader = DataLoader(dataset=dataset, batch_size=config.VAL_BATCH_SIZE, sampler=SubsetRandomSampler(test_idx))

    return train_dataloader, val_dataloader, test_dataloader, item_num, user_num

def loss_fn(mask, y_hat, y):
    '''
    mask: B * S
    y_hat: B * S * (d+1)
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
    y_hat_clone = y_hat[i_pos, j_pos, k_pos].clone()
    return torch.sum(-torch.log(y_hat_clone + 1e-16)) / len(i_pos)

def train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer, logging):
    checkpoint_path = os.path.join(config.OUTPUT_DIR, "NOVA " + datetime.now(ZoneInfo('Asia/Shanghai')).strftime("%Y-%m-%d %H_%M_%S")+".pt")
    es = EarlyStopping(checkpoint_path, patience=16, verbose=True)

    for iter in range(config.MAX_ITER):
        loss = 0
        model.train()
        print(iter + 1)
        for i, (u, X, overall, timeDiff) in enumerate(train_dataloader):
            X = X.cuda()
            p_hat, mask = model(X, overall.cuda(), timeDiff.cuda(), train=True)
            tmp_loss = loss_fn(mask, p_hat, X)
            loss += tmp_loss
            optimal_obj = tmp_loss
            optimizer.zero_grad()
            optimal_obj.backward()
            optimizer.step()

        model.eval()
        logging.info("Loss:{} at iter: {}".format(loss, iter + 1))
        recall, mrr, ndcg = evaluate(model, val_dataloader, config.RECOMMEND_NUM)
        logging.info("Recall:{}. MRR:{}. NDCG:{}".format(recall, mrr, ndcg))
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
    for batch_index, (u, X, overall, timeDiff) in enumerate(val_dataloader):
        y = X[:, -1].squeeze()
        count += u.shape[0]
        X[:, -1] = 0
        X = X.cuda()
        p_hat, mask = model.predict(X, overall.cuda(), timeDiff.cuda())
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

if __name__ == "__main__":
    import os
    from datetime import datetime
    import logging
    
    path = os.path.join("./CausalRec/logs/NOVA", datetime.now().strftime("%Y-%m-%d %H_%M_%S")+".log")

    logging.basicConfig(filename=path, level=logging.DEBUG)
    logging.info("NOVA")
    logging.info("LEARNING_RATE: {}".format(config.LEARNING_RATE))
    logging.info("OPTIMIZER: {}".format(config.OPTIMIZER))
    logging.info("BATCH_SIZE: {}".format(config.BATCH_SIZE))
    logging.info("EMB_SIZE: {}".format(config.EMB_SIZE))
    logging.info("MAX_ITER: {}".format(config.MAX_ITER))
    logging.info("LAMBDA1: {}".format(config.LAMBDA2))
    logging.info("SCENARIO: {}".format(config.scenario))
    
    train_dataloader, val_dataloader, test_dataloader, item_num, user_num = get_dataloader(scenario = config.scenario)

    logging.info("------NOVA------")
    model = NOVA(bert_max_len=config.BERT_MAX_LEN, num_items=item_num + 1).cuda()
    if config.OPTIMIZER == config.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER == config.SGD:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.LEARNING_RATE)
    else:
        pass
    train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer, logging)


    logging.info("------NOVAC------")
    w, ae = get_causal(config)
    model = NOVAC(bert_max_len=config.BERT_MAX_LEN, num_items=item_num + 1, ae=ae, w=w).cuda()
    if config.OPTIMIZER == config.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER == config.SGD:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.LEARNING_RATE)
    else:
        pass
    train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer, logging)

    logging.info("------NOVAR------")
    model = NOVAR(bert_max_len=config.BERT_MAX_LEN, num_items=item_num + 1).cuda()
    if config.OPTIMIZER == config.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    elif config.OPTIMIZER == config.SGD:
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=config.LEARNING_RATE)
    else:
        pass
    train(model, train_dataloader, val_dataloader, test_dataloader, loss_fn, optimizer, logging)