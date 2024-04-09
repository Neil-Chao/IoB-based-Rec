import torch.optim as optim
import torch


def train(model, dataloader, max_iter, optimizer: optim.Optimizer, loss_fn, multiplier):
    loss_list = []
    for iter in range(max_iter):
        
        for i, x in enumerate(dataloader):
            x = (x @ multiplier).to(torch.int)
            optimizer.zero_grad()
            p_hat, mask = model(x, train=True)
            loss = loss_fn(mask, p_hat, x)
            loss.backward()
            optimizer.step()
            loss_list.append(loss)
        if (iter + 1) % 10 == 0:
            print("iter:{}".format(iter+1))
    
    plt.title("Train loss vs. epochs", fontsize=20)
    loss_list = [i.cpu().detach().numpy() for i in loss_list]
    x_list = range(1, len(loss_list) + 1)
    plt.plot(x_list, loss_list)
    plt.grid()
    plt.show()

@torch.no_grad()
def validate(model, dataloader, multiplier, k=3):
    r_cnt = 0
    r = 0
    acc_cnt = 0
    acc = 0
    mrr = 0
    for i, x in enumerate(dataloader):
        r_cnt += 1
        acc_cnt += 3
        x = (x @ multiplier).to(torch.int)
        p_hat, mask = model(x, train=False)
        scores = p_hat.squeeze()[-1]
        values, indices = torch.topk(scores, k)
        if x.squeeze()[-1] in indices:
            r += 1
            acc += 1
            for i in range(indices.shape[0]):
                if indices[i] == x.squeeze()[-1]:
                    mrr += 1 / (i + 1)
                    break
    return r / r_cnt, acc / acc_cnt, mrr / r_cnt