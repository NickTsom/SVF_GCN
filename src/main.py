
import dgl
from dgl.data import MiniGCDataset
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import Model

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def train(model, data_loader, loss_func, optimiser, num_epochs):
    epoch_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model.forward(bg)
            loss = loss_func(prediction, label)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.detach().item()

        epoch_loss /= (iter + 1)
        print('Epoch {:05d} | loss {:.4f}'.format(epoch+1, epoch_loss))
        epoch_losses.append(epoch_loss)

    return epoch_losses

def evaluate(testset, model):
    test_X, test_Y = map(list, zip(*testset))
    test_bg = dgl.batch(test_X)
    test_Y = torch.tensor(test_Y).float().view(-1, 1)
    probs_Y = torch.softmax(model.forward(test_bg), 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1).float()
    correct_predictions = (test_Y == argmax_Y).sum().item()
    accuracy_percent = correct_predictions / len(test_Y) * 100

    print('Accuracy of predictions on the test set: {:.2f}%'.format(accuracy_percent))

def main():
    # Create training and test sets.
    trainset = MiniGCDataset(320, 10, 20)
    testset = MiniGCDataset(80, 10, 20)
    data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                            collate_fn=collate)

    # Create model
    model = Model(1, 256, trainset.num_classes)
    loss_func = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 200
    epoch_losses = train(model, data_loader, loss_func, optimiser, num_epochs)

    evaluate(testset, model)

    # Plot losses
    plt.title('Cross Entropy Loss Over Minibatches')
    plt.plot(epoch_losses)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss Value')
    plt.show()

    return 0

if __name__ == "__main__":
    main()
