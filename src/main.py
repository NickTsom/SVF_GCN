
import dgl
from dgl.data.minigc import MiniGCDataset
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from classifier import Classifier
from svfg_loader import SVFGDataset


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

def train(model, data_loader, loss_func, optimiser, num_epochs):
    epoch_losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model(bg)
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
    probs_Y = torch.softmax(model(test_bg), 1)
    argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1).float()
    correct_predictions = (test_Y == argmax_Y).sum().item()
    accuracy_percent = correct_predictions / len(test_Y) * 100

    print('Accuracy of predictions on the test set: {:.2f}%'.format(accuracy_percent))

def main():
    # Parameters
    num_epochs = 200
    learning_rate = 0.001
    dataset_batch_size = 32
    model_input_dimensions = 1
    model_output_dimensions = 256
    dropout = 0.2

    train_dataset = MiniGCDataset(320, 10, 200)
    test_dataset = MiniGCDataset(80, 10, 200)

    #train_edges_file = 'data/svfg_edges.csv'
    #train_properties_file = 'data/svfg_properties.csv'
    #train_dataset = SVFGDataset(train_edges_file, train_properties_file)
    #test_dataset = SVFGDataset()

    data_loader = DataLoader(train_dataset, batch_size=dataset_batch_size,
                             shuffle=True, collate_fn=collate)

    # Create model
    model = Classifier(model_input_dimensions, model_output_dimensions,
                       train_dataset.num_classes, dropout)
    loss_func = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    epoch_losses = train(model, data_loader, loss_func, optimiser, num_epochs)

    model.eval()
    #valuate(test_dataset, model)

    # Plot losses
    plt.title('Cross Entropy Loss Over Minibatches')
    plt.plot(epoch_losses)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss Value')
    plt.show()

    return 0

if __name__ == "__main__":
    main()
