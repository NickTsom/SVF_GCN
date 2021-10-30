
import dgl
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import Model
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
    # Parameters
    num_epochs = 200
    learning_rate = 0.001
    dataset_batch_size = 32
    model_input_dimensions = 1
    model_output_dimensions = 256

    # Load training and test sets.
    train_edges_path = 'data/train_graph_edges.csv'
    train_properties_path = 'data/train_graph_properties.csv'
    train_dataset = SVFGDataset(train_edges_path, train_properties_path)

    test_edges_path = 'data/test_graph_edges.csv'
    test_properties_path = 'data/test_graph_properties.csv'
    test_dataset = SVFGDataset(test_edges_path, test_properties_path)

    data_loader = DataLoader(train_dataset, batch_size=dataset_batch_size,
                             shuffle=True, collate_fn=collate)

    # # Display each graph
    # for iter, (bg, label) in enumerate(data_loader):
    #     fig, ax = plt.subplots()
    #     nx.draw(bg.to_networkx(), ax=ax)
    #     plt.show()

    # Create model
    model = Model(model_input_dimensions, model_output_dimensions,
                  train_dataset.num_classes)
    loss_func = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)
    
    epoch_losses = train(model, data_loader, loss_func, optimiser, num_epochs)

    evaluate(test_dataset, model)

    # Plot losses
    plt.title('Cross Entropy Loss Over Minibatches')
    plt.plot(epoch_losses)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss Value')
    plt.show()

    return 0

if __name__ == "__main__":
    main()
