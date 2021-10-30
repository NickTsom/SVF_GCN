
import dgl
from dgl.data import DGLDataset
import pandas as pd
import torch


class SVFGDataset(DGLDataset):
    def __init__(self, edges_file_path, properties_file_path):
        self.edges_file_path = edges_file_path
        self.properties_file_path = properties_file_path
        super().__init__(name='svfg')

    def process(self):
        edges = pd.read_csv(self.edges_file_path)
        properties = pd.read_csv(self.properties_file_path)
        self.graphs = []
        self.labels = []
        self.num_classes = 0

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            self.graphs.append(g)
            self.labels.append(label)

        self.num_classes = max(self.labels) + 1

        # Convert the label list to tensor for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)
