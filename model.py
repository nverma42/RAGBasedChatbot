import re
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split

# Define the image classifer CNN model
class cnn_model(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(cnn_model, self).__init__()

        # Define the convolutional neural network architecture
        # The first convolutional layer has 16 filters of size 3x3.
        # Output Convolution Layer = (Input Size - Kernel Size + 2 * Padding) / Stride + 1
        # The output tensor shape is: batch size x 128 x 382.
        self.conv1 = nn.Conv1d(in_channels=num_channels, out_channels=128, kernel_size=3)
        
        # Perform max pooling for kernel size of 2. 
        # Output Pooling Layer = (Input Size - Kernel Size) / Stride + 1
        # The output tensor shape is: batch size x 128 x 191.
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Add 2 fully connected layers by flattening the tensor.
        # Please note that the input size is 128 * 191 = 24,448.
        self.fc1 = nn.Linear(in_features=24448, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        # Pass through the first convolution layer
        x = x.unsqueeze(1)  # Add a channel dimension
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        # Flatten the tensor to pass to first fully connected layer.
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        
        # Pass through the second fully connected layer
        # The cross entropy loss function will apply the softmax function 
        # to get the class probabilities.
        logits = self.fc2(x)
                
        return logits

class rag_model:
    def train_model(self, model, train_dataloader, criterion, optimizer):
        train_avg_loss = []
        curr_loss = 0.0
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            curr_loss = 0.0
            for X, y_true in train_dataloader:
                optimizer.zero_grad()
                y_pred = model(X)
                loss = criterion(y_pred, y_true)
                curr_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_avg_loss.append(curr_loss/len(train_dataloader))
        return train_avg_loss

    def evaluate_model(self, model, test_dataloader):
        y_true = []
        y_pred = []
        model.eval()
        for X, y_true_batch in test_dataloader:
            y_pred_batch = model(X)
            y_true.extend(y_true_batch.detach().numpy())
            y_pred.extend(y_pred_batch.detach().numpy())
        y_pred = [np.argmax(v) for v in y_pred]
        return y_true, y_pred

    def get_metrics(self, y_true, y_pred):
        # Get the accuracy score.
        accuracy = accuracy_score(y_true, y_pred)
        # Get the precision score.
        precision = precision_score(y_true, y_pred, average='weighted')
        # Get the recall score.
        recall = recall_score(y_true, y_pred, average='weighted')
       # Get the f1 score.
        f1score = f1_score(y_true, y_pred, average='weighted')
        return accuracy, precision, recall, f1score

    def __init__(self):
        # Set the random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Load pretrained Sentence-BERT model.
        self.encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2')

        self.label_encoder = {'Normal': 0, 'Depression': 1, 'Suicidal': 2, 'Anxiety': 3, 'Bipolar': 4, 'Stress': 5, 'Personality disorder': 6}
        
        self.label_decoder = {}
        for k, v in self.label_encoder.items():
            self.label_decoder[v] = k

        # Create the CNN model
        self.model = cnn_model(1, 7)

        # Train the model
        self.train_cnn_model()

    def train_cnn_model(self):
        # Train the CNN model and print metrics
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        # Read the data
        df = pd.read_csv("Combined Data.csv")

        #remove missing data
        df.dropna(inplace = True)

        statements = df['statement'].values
        status = df['status'].values

        label_encoder = {'Normal': 0, 'Depression': 1, 'Suicidal': 2, 'Anxiety': 3, 'Bipolar': 4, 'Stress': 5, 'Personality disorder': 6}
        y = [label_encoder[status[i]] for i in range(len(status))]

        # Encode the statements
        X = self.encoder.encode(statements)
        label_encoder = {'Normal': 0, 'Depression': 1, 'Suicidal': 2, 'Anxiety': 3, 'Bipolar': 4, 'Stress': 5, 'Personality disorder': 6}
        y = [label_encoder[status[i]] for i in range(len(status))]

        # Convert the data to tensors
        X_train_tensor = torch.tensor(X, dtype=torch.float32)
        y_train_tensor = torch.tensor(y, dtype=torch.long)

        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)

        # Create dataloaders
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

        cnn_train_avg_loss = self.train_model(self.model, train_dataloader, criterion, optimizer)

        # Plot the F1 scores vs noise percentage
        plt.figure(figsize=(10, 5))
        plt.plot(cnn_train_avg_loss, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Training Loss')
        plt.show()

    def build_knowledge_graph(self):
        # Read the conversation data
        conv_df = pd.read_json('hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json', lines=True)

        context = conv_df['Context'].values
        response = conv_df['Response'].values

        # Encode context
        self.encoded_context = self.encoder.encode(context)

        encoded_context_tensor = torch.tensor(self.encoded_context, dtype=torch.float32)

        # Make model predictions
        logits = self.model(encoded_context_tensor)
        probs = F.softmax(logits, dim=1)
        probs = probs.detach().numpy()

        # Associate rows with labels
        self.pred_labels = {}
        for i, row in enumerate(probs):
            label = np.argmax(row)
            self.pred_labels[i] = label

        # Randomly shuffle and split the conversation dataset
        train_df = conv_df.sample(frac=0.8, random_state=42)  # 80% sample
        self.test_df = conv_df.drop(train_df.index) # Remaining 20%

        # Build knowledge graph with conversation training data.
        self.kg = nx.DiGraph()
        for idx, row in train_df.iterrows():
            label = self.pred_labels[idx]

            category_node = f"category_{self.label_decoder[label]}"
            if category_node not in self.kg.nodes:
                self.kg.add_node(category_node, type="category")

            context_node = f"context_{idx}"
            response_node = f"response_{idx}"

            context_data = self.encoded_context[idx]
            response_data = response[idx]

            self.kg.add_node(context_node, data=context_data, type="context")
            self.kg.add_node(response_node, data=response_data, type="response")
            
            self.kg.add_edge(category_node, context_node)
            self.kg.add_edge(context_node, response_node)
        
        # knowledge graph summary
        print("Number of nodes:", self.kg.number_of_nodes())
        print("Number of edges:", self.kg.number_of_edges())

        # print a subset of nodes to visualize the graph
        few_nodes = list(self.kg.nodes)[:9]
        
        # Create a subgraph
        H = self.kg.subgraph(few_nodes)

        # Draw the subgraph
        nx.draw(H, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=12, font_weight='bold')
        plt.show()

    def retrieve_data(self):
        retrieved_data_list = []
        for idx, row in self.test_df.iterrows():
            label = self.pred_labels[idx]
            category_node = f"category_{self.label_decoder[label]}"
            query = self.test_df['Context'][idx]
            query_vec = self.encoded_context[idx]

            # Check if the query is similar to the context node
            best_cos_sim = 0.0
            best_context_node = None
            for context_node in self.kg.neighbors(category_node):
                context_data = self.kg.nodes[context_node]['data']
                cos_sim = np.dot(query_vec, context_data) / (np.linalg.norm(query_vec) * np.linalg.norm(context_data))
                if cos_sim > best_cos_sim:
                    best_cos_sim = cos_sim
                    best_context_node = context_node

            response_node = list(self.kg.neighbors(best_context_node))[0]
            response = self.kg.nodes[response_node]['data']

            retrieved_data_list.append((query, response))

        return retrieved_data_list