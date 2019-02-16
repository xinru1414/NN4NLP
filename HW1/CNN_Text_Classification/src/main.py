from preprocess import *
from model import CNNClassify
import config
import torch
import numpy as np
from random import randint
from tqdm import tqdm

CUDA = torch.cuda.is_available()
np.random.seed(config.random_seed)
torch.manual_seed(config.random_seed)

if CUDA:
    gpu_cpu = torch.device('cuda')
    torch.cuda.manual_seed(config.random_seed)
else:
    torch.device('cpu')


def get_long_tensor(x):
    return torch.LongTensor(x).to(gpu_cpu)

def make_prediction_and_acc(model: CNNClassify, examples: Examples):
    '''
    Takes the model and examples and returns the labels for the sentences.
    :param model: CNNClassify
    :param examples: Examples
    :return: predictions and acc score
    '''
    predictions = []
    for b in batch(examples, config.batch_size):
        b_sents = get_long_tensor(b.sents)
        predictions += model.predict(b_sents).tolist()
    acc = np.sum(examples.labels == np.array(predictions)) * 1.0 / len(examples.labels)
    return np.array(predictions), acc

def train_and_evaluate(dl: DataLoader, model: CNNClassify):
    prev_best = 0
    patience = 0
    decay = 0
    lr = config.lr

    optim = torch.optim.Adam(model.parameters(), lr=config.lr)
    for epoch in tqdm(range(config.max_epochs)):
        # randomly get a train_dev set from the K-fold
        i = randint(0, config.k - 1)
        train, dev = dl.folds[i]
        train_loss = 0
        dev_loss = 0

        print('\ntrain model on the train set')
        model.train()
        for x, y in batch(train.shuffled(), config.batch_size):
            x, y = get_long_tensor(x), get_long_tensor(y)
            loss = model.loss(x, y)
            train_loss += loss.item()

            model.zero_grad()
            loss.backward()
            optim.step()

        print('evaluate model on the dev set')
        model.eval()
        for x, y in batch(dev, config.batch_size):
            x, y = get_long_tensor(x), get_long_tensor(y)

            loss = model.loss(x, y)
            dev_loss += loss.item()

        # calculating train acc and dev acc
        train_predictions, train_acc = make_prediction_and_acc(model,train)
        dev_predictions, dev_acc = make_prediction_and_acc(model,dev)
        print(f'Epoch {epoch}, fold{i}, train loss {np.mean(train_loss)}, train accuracy {train_acc}, dev loss {dev_loss}, dev acc {dev_acc}')


        if dev_acc < prev_best:
            patience += 1
            if patience == 3:
                lr *= 0.5
                optim = torch.optim.Adam(model.parameters(), lr=lr)
                tqdm.write('dev accuracy did not increase in 3 epochs, halfing the learning rate')
                patience = 0
                decay += 1
        else:
            prev_best = dev_acc
            print('save the best model')
            model.save()


        if decay >= 3:
            print('evaluating model on test set')
            model.load()
            print('load the best model')
            test_predictions, test_acc = make_prediction_and_acc(model, dl.test_examples)
            print(f'test acc on dev {test_acc}')
            with open(f'{config.test_result_path}_{test_acc}.txt', 'w') as f:
                for i in test_predictions:
                    f.write(dl.i2l[i] + '\n')

def main():
#if True:
    # load data
    dl = DataLoader(config)
    # load the model
    model = CNNClassify(config, dl, gpu_cpu)
    # train and evaluate the model
    train_and_evaluate(dl,model)


if __name__ == "__main__":
    main()


