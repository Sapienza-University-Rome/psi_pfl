import ray
import multiprocessing

# Define number of CPUs to use
CPUs_to_use = max(1, multiprocessing.cpu_count() // 4)

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import random
import copy
import warnings

from folktables import ACSDataSource, ACSIncome

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, silhouette_score
from sklearn.cluster import DBSCAN, OPTICS, KMeans

from fedartml import SplitAsFederatedData
from fedartml.function_base import jensen_shannon_distance, hellinger_distance, earth_movers_distance

from collections import Counter, defaultdict

import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
from flwr.client import Client
from flwr.common import Context  # Import Context as required by the new signature

from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences


from fedlab.utils.functional import evaluate
from fedlab.contrib.algorithm.fedavg import FedAvgServerHandler
from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.core.client.trainer import SerialClientTrainer
from fedlab.core.standalone import StandalonePipeline
from fedlab.utils import Logger
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms


from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

from PIL import Image


class DNN_pytorch(nn.Module):
    def __init__(self, seq_len, num_classes, n_hidden, embedding_matrix):
        super(DNN_pytorch, self).__init__()
        
        # Embedding layer (non-trainable)
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # Make embedding non-trainable
        
        # LSTM layer with reduced units and dropout
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=n_hidden // 2, batch_first=True, dropout=0.2)
        
        # Fully connected dense layer with reduced size
        self.fc1 = nn.Linear(n_hidden // 2, 16)
        
        # Output layer for binary classification (softmax activation)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        # Ensure input is of type LongTensor for embedding lookup
        x = x.long()
        
        # Pass input through the embedding layer
        x = self.embedding(x)
        
        # Pass through LSTM layer
        x, (hidden_state, cell_state) = self.lstm(x)
        
        # Use the last hidden state from the LSTM output
        x = hidden_state[-1]
        
        # Pass through the first fully connected dense layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Pass through the output layer with softmax activation
        x = F.softmax(self.fc2(x), dim=1)
        
        return x


def DNN(seq_len, num_classes, n_hidden, embedding_matrix):
    initializer = tf.keras.initializers.GlorotUniform(seed=random_state)

    # Define the model
    model = tf.keras.models.Sequential([
        # Embedding layer (non-trainable)
        tf.keras.layers.Embedding(
            input_dim=embedding_matrix.shape[0],  # vocab size
            output_dim=embedding_matrix.shape[1],  # embedding dimension
            weights=[embedding_matrix],
            input_length=seq_len,
            trainable=False
        ),

        # Single LSTM layer with reduced units and dropout
        tf.keras.layers.LSTM(n_hidden // 2, dropout=0.2, recurrent_dropout=0.2),

        # Fully connected dense layer with reduced size
        tf.keras.layers.Dense(16, activation='relu', kernel_initializer=initializer),

        # Output layer for binary classification (softmax activation)
        tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer)
    ])

    # Build the model by specifying the input shape
    model.build(input_shape=(None, seq_len))
    
    return model

def DNN_celeba(input_shape, num_classes):
    initializer = tf.keras.initializers.GlorotUniform(seed=random_state)

    model = tf.keras.models.Sequential([
        # First Conv2D + ReLU + MaxPooling
        tf.keras.layers.Conv2D(8, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=initializer, input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # Second Conv2D + ReLU + MaxPooling
        tf.keras.layers.Conv2D(16, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=initializer),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # Third Conv2D + ReLU + MaxPooling
        tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu', kernel_initializer=initializer),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        # Flatten the output for Fully Connected Layer
        tf.keras.layers.Flatten(),

        # Fully Connected Layer with ReLU
        tf.keras.layers.Dense(8 * 8 * 32, activation='relu', kernel_initializer=initializer),

        # Output Layer with Softmax
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model


class DNN_celeba_pytorch(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(DNN_celeba_pytorch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=8, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Compute the flattened feature size after convolutions
        dummy_input = torch.zeros(1, *input_shape)  # (Batch size, C, H, W)
        with torch.no_grad():
            x = self.pool(torch.relu(self.conv1(dummy_input)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.pool(torch.relu(self.conv3(x)))
            flatten_size = x.view(1, -1).shape[1]
        
        self.fc1 = nn.Linear(flatten_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def evaluate_DNN_FL_celeba_pytorch(curr_global_params_retrieved, curr_list_x_test, curr_list_y_test):
    """
    Evaluate the global model on each client's test data using the global parameters.
    
    Args:
        curr_global_params_retrieved: The global model parameters retrieved from the server.
        curr_list_x_test: A list containing test data (features) for each client.
        curr_list_y_test: A list containing test labels for each client.
    
    Returns:
        A dictionary with two keys:
        - 'client_metrics': A dictionary where keys are client indices and values are tuples of metrics 
          (loss, accuracy, precision, recall, f1score) and the number of examples for each client.
        - 'weighted_averages': A dictionary containing the weighted average of each metric across all clients.
    """
    # Initialize results storage
    client_metrics = {}

    # Iterate through each client's test data
    for client_id, (x_test_client, y_test_client) in enumerate(zip(curr_list_x_test, curr_list_y_test)):
        # Convert test data to PyTorch tensors
        # curr_x_test_tensor = torch.tensor(np.array(x_test_client), dtype=torch.float32)
        curr_x_test_tensor = torch.tensor(np.array(x_test_client).reshape(-1, 3, 32, 32), dtype=torch.float32)
        # curr_y_test_tensor = torch.tensor(np.array(y_test_client), dtype=torch.long)
        curr_y_test_tensor = torch.tensor(np.array(y_test_client, dtype=np.int64), dtype=torch.long)
        # Create a DataLoader
        curr_test_dataset = TensorDataset(curr_x_test_tensor, curr_y_test_tensor)
        curr_test_loader = DataLoader(curr_test_dataset, batch_size=batch_size, shuffle=False)

        # Define model
        net = DNN_celeba_pytorch(curr_x_test_tensor.shape[1:], len(torch.unique(y_test_tensor)))

        # Load global parameters
        SerializationTool.deserialize_model(net, curr_global_params_retrieved)

        # Evaluate the model on this client's test data
        loss, accuracy = evaluate(net, torch.nn.CrossEntropyLoss(), curr_test_loader)
        precision, recall, f1score = 0, 0, 0
        # Store metrics and number of examples for this client
        client_metrics[client_id] = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1score": f1score,
            "num_examples": len(x_test_client)
        }
    
    # Calculate weighted averages
    total_examples = sum(client["num_examples"] for client in client_metrics.values())
    
    weighted_averages = {
        "loss": sum(client["loss"] * client["num_examples"] for client in client_metrics.values()) / total_examples,
        "accuracy": sum(client["accuracy"] * client["num_examples"] for client in client_metrics.values()) / total_examples,
        "precision": sum(client["precision"] * client["num_examples"] for client in client_metrics.values()) / total_examples,
        "recall": sum(client["recall"] * client["num_examples"] for client in client_metrics.values()) / total_examples,
        "f1score": sum(client["f1score"] * client["num_examples"] for client in client_metrics.values()) / total_examples
    }

    return {
        "client_metrics": client_metrics,
        "weighted_averages": weighted_averages
    }


class FedLabStandardPipeline(StandalonePipeline):
    def __init__(self, handler, trainer, test_loader):
        super().__init__(handler, trainer)
        self.test_loader = test_loader
        self.communication_size = 0
        self.time_list = [(0,0)]
        self.loss = []
        self.acc = []
        self.selected_clients = []
        self.selected_clusters = []

    def get_communication_size(self):
        return self.communication_size

    def get_time_list(self):
        return self.time_list[1:]

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.acc

    def get_selected_clients(self):
        return self.selected_clients

    def get_selected_clusters(self):
        return self.selected_clusters

    def show(self):
        plt.figure(figsize=(12, 5))

        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(np.arange(len(self.loss)), self.loss, label='Training Loss', color='blue', linewidth=2)
        ax1.set_xlabel("Communication Round", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Test Loss", fontsize=14)
        ax1.legend()
        ax1.grid(True)

        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(np.arange(len(self.acc)), self.acc, label='Accuracy', color='green', linewidth=2)
        ax2.set_xlabel("Communication Round", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_title("Test Accuracy", fontsize=14)
        ax2.legend()
        ax2.grid(True)

        plt.subplots_adjust(wspace=0.3)
        plt.show()


def get_m_list(d):
    return [int(d / i) for i in range(2, d) if d % i == 0][:2]




class SGDSerialClientTrainer(SerialClientTrainer):
    """
    Train multiple clients in a single process.

    Args:
        model (torch.nn.Module): Model used in this federation.
        num_clients (int): Number of clients in current trainer.
        cuda (bool): Use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        logger (Logger, optional): Object of :class:`Logger`.
        personal (bool, optional): If Ture is passed, SerialModelMaintainer will generate the copy of local parameters list and maintain them respectively. These paremeters are indexed by [0, num-1]. Defaults to False.
    """
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, personal)
        self._LOGGER = Logger() if logger is None else logger
        self.cache = []

    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size.
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss()

    @property
    def uplink_package(self):
        package = copy.deepcopy(self.cache)
        self.cache = []
        return package

    def local_process(self, payload, id_list, server_round):
        model_parameters = payload[0]
        for id in (progress_bar := tqdm(id_list)):
            progress_bar.set_description(f"Training on client {id}", refresh=True)
            data_loader = DataLoader(self.dataset[id], batch_size=self.batch_size)
            pack = self.train(model_parameters, data_loader, server_round)
            self.cache.append(pack)

    def train(self, model_parameters, train_loader, server_round):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        self.set_model(model_parameters)
        self._model.train()
        
        # lr = args.lr

        if server_round < 301:
            lr = learn_rate
        elif 301 <= server_round < 601:
            lr = learn_rate / 2
        else:
            lr = learn_rate / 4

        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return [self.model_parameters]




def hellinger_distance_pariwise(p, q):
    return (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

def compute_pairwise_hellinger(histograms):
    num_clients = len(histograms)
    distance_matrix = np.zeros((num_clients, num_clients))

    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                distance_matrix[i, j] = hellinger_distance_pariwise(histograms[i], histograms[j])

    return distance_matrix

warnings.filterwarnings("ignore")

lower_min_samples = 2
upper_min_samples = 4
noise_threshold = 0.7
max_eps_values = np.linspace(0.2, 0.8, num=2)
cluster_methods = ['xi']
xi_values = np.linspace(0.05, 0.8, num=2)

def get_scores_and_labels_OPTICS(X):
    scores = []
    all_labels_list = []
    params_list = []
    n_clusters = []

    total_iterations = upper_min_samples - lower_min_samples

    for min_samples in tqdm(range(lower_min_samples, upper_min_samples), total=total_iterations, desc="Finding Optimum Parameters"):
        for max_eps in max_eps_values:
            for cluster_method in cluster_methods:
                for xi in xi_values:
                    optics = OPTICS(metric='precomputed', min_samples=min_samples, max_eps=max_eps,
                                    cluster_method=cluster_method, xi=xi)
                    clusters = optics.fit_predict(X)

                    num_clusters = len(set(clusters)) - (-1 in set(clusters))  # Subtract 1 if noise is present
                    clusters_interval_met = 2 <= num_clusters < 50
                    non_noise_mask = clusters != -1
                    threshold_not_met = np.sum(non_noise_mask) > int(args.K * noise_threshold)
                    # Only calculate silhouette score if the number of clusters is appropriate
                    if clusters_interval_met and threshold_not_met:
                        # Calculate silhouette score only for non-noise points
                        X_sub = X[non_noise_mask][:, non_noise_mask]  # Extract the sub-matrix for non-noise points
                        score = silhouette_score(X_sub, clusters[non_noise_mask], metric='precomputed')
                        scores.append(score)
                        n_clusters.append(num_clusters)
                        all_labels_list.append(clusters)
                        params_list.append((min_samples, max_eps, cluster_method, xi))
                    else:
                        # Append -1 to indicate an invalid number of clusters for scoring
                        scores.append(-1)
                        n_clusters.append(-1)
                        all_labels_list.append(None)
                        params_list.append((min_samples, max_eps, cluster_method, xi))

    best_index = np.argmax(scores)
    best_params = params_list[best_index]
    best_labels = all_labels_list[best_index]
    best_n_clusters = n_clusters[best_index]
    best_score = scores[best_index]

    return {
        "best_min_samples": best_params[0],
        "best_max_eps": best_params[1],
        "best_cluster_method": best_params[2],
        "best_xi": best_params[3],
        "best_labels": best_labels,
        "best_n_clusters": best_n_clusters,
        "best_score": best_score,
    }

class ClusteredPowerofchoice(SyncServerHandler):
    def __init__(self, *args, **kwargs):
        super(ClusteredPowerofchoice, self).__init__(*args, **kwargs)

    def setup_optim(self, d):
        self.d = d

    def setup_clusters(self, cluster_dict, cluster_sampling, w, m, rho=0.5):
        self.cluster_dict = cluster_dict
        self.clients_list = [client for cluster in cluster_dict.values() for client in cluster]
        self.cluster_sampling = cluster_sampling
        self.best_clients = {}
        self.w = w
        self.z = int(self.d / self.w)   # fraction of d to be reserved to each cluster
        self.m = m
        self.rho = rho

    def sample_candidates(self):
        if self.cluster_sampling == "client":
            selected_clusters, cluster_keys = self.sample_clusters_by_client()

        elif self.cluster_sampling == "data":
            selected_clusters, cluster_keys = self.sample_clusters_by_data()

        elif self.cluster_sampling == "uniform":
            selected_clusters, cluster_keys = self.sample_clusters_uniform()

        selected_candidates = self.sample_candidates_from_clusters(selected_clusters)

        n_selected_candidates = sum([len(cluster) for cluster in selected_candidates])

        if n_selected_candidates < self.d:
            extra_clients = self.select_extra_clients(
                [client for cluster in selected_candidates for client in cluster],
                self.d
            )
            selected_candidates = self.distribute_extra_clients(selected_candidates, extra_clients)

        return selected_candidates, cluster_keys

    def sample_clusters_by_client(self):
        keys = list(self.cluster_dict.keys())
        prob_distr = self.normalize_elements([len(cluster) for cluster in self.cluster_dict.values()])
        size = min(self.w, len(keys))
        random_keys = np.random.choice(keys, p=prob_distr, size=size, replace=False)
        cluster_keys = list([int(key) for key in random_keys])
        selected_clusters = [self.cluster_dict[i] for i in random_keys]

        return selected_clusters, cluster_keys

    def sample_clusters_by_data(self):
        keys = list(self.cluster_dict.keys())
        prob_distr = [np.sum([client[1] for client in cluster]) for cluster in self.cluster_dict.values()]
        size = min(self.w, len(keys))
        random_keys = np.random.choice(keys, p=prob_distr, size=size, replace=False)
        cluster_keys = list([int(key) for key in random_keys])
        selected_clusters = [self.cluster_dict[i] for i in random_keys]

        return selected_clusters, cluster_keys

    def sample_clusters_uniform(self):
        # Note: in the paper, clusters are selected based on higher average local loss and latency (replacement is allowed), not uniformly at random
        keys = list(self.cluster_dict.keys())
        random_keys = random.sample(keys, min(self.w, len(keys)))
        cluster_keys = list([int(key) for key in random_keys])
        selected_clusters = [self.cluster_dict[i] for i in random_keys]

        return selected_clusters, cluster_keys

    def sample_candidates_from_clusters(self, selected_clusters):
        selected_candidates = []

        # Sample at random a number "z" of clients out of each cluster based on each client's data fraction
        for cluster in selected_clusters:
            selected_elements = [element[0] for element in cluster]
            prob_distr = self.normalize_elements([element[1] for element in cluster])
            size = min(self.z, len(selected_elements))
            random_elements = np.random.choice(selected_elements, p=prob_distr, size=size, replace=False)
            selected_candidates.append(sorted(random_elements, reverse=True))     # DESC

        return selected_candidates

    def sample_clients(self, candidates, losses):
        chosen_clients = []

        if type(losses) == dict:
            losses = [[losses[client] for client in cluster] for cluster in candidates]

        # Note: in the paper, clients are selected based on their latency, not by Power-Of-Choice
        for i in range(len(candidates)):
            candidates_list = candidates[i]
            losses_list = losses[i]

            sort = np.array(losses_list).argsort().tolist()
            sort.reverse()
            n_clients = min(self.m, len(candidates_list))
            selected_clients = np.array(candidates_list)[sort][0:n_clients]
            chosen_clients.extend(selected_clients)

        if len(chosen_clients) < self.num_clients_per_round:
            extra_clients = self.select_extra_clients(chosen_clients, self.num_clients_per_round)
            chosen_clients.extend(extra_clients)

        for client in chosen_clients:
            if client not in self.best_clients:
                self.best_clients[client] = 1
            else:
                self.best_clients[client] += 1

        return chosen_clients

    def sample_best_clients(self):
        # Sample a number "d" of clients among the pool of best ones
        size = min(self.d, len(self.best_clients))

        clients = [clients for clients in self.best_clients.keys()]
        prob_distr = self.normalize_elements([occurrences for occurrences in self.best_clients.values()])
        sampled_clients = [random.choices(clients, weights=prob_distr, k=size)]

        return sampled_clients

    def select_extra_clients(self, chosen_clients, target):
        n_missing_clients = target - len(chosen_clients)
        available_clients = [client[0] for client in self.clients_list if client[0] not in chosen_clients]
        prob_distr = self.normalize_elements([client[1] for client in self.clients_list if client[0] not in chosen_clients])
        extra_clients = list(np.random.choice(available_clients, p=prob_distr, size=n_missing_clients, replace=False))

        return extra_clients

    def fill_with_extra_cluster(self, client_losses, cluster_keys, n_selected_candidates):
        n_missing_clients = self.d - n_selected_candidates
        keys = list(self.cluster_dict.keys())

        client_losses_dict = {
            key: [(tup[0], client_losses[tup[0]]) for tup in self.cluster_dict[key]]
            for key in keys if int(key) not in cluster_keys
        }

        cluster_list = [[client for client in client_losses_dict[key]] for key in client_losses_dict.keys()]
        available_clients = [client[0] for cluster in cluster_list for client in cluster]
        losses = [client[1] for cluster in cluster_list for client in cluster]
        prob_distr = self.normalize_elements(losses)

        chosen_indices = np.random.choice(len(available_clients), p=prob_distr, size=n_missing_clients, replace=False)
        chosen_clients = [(available_clients[i], losses[i]) for i in chosen_indices]

        return chosen_clients

    def distribute_extra_clients(self, selected_candidates, extra_clients):
        n_extra_clients = len(extra_clients)
        n_candidates = len(selected_candidates)

        clients_per_list = [n_extra_clients // n_candidates] * n_candidates  # Distribute as evenly as possible
        remainder = n_extra_clients % n_candidates  # Handle the remainder

        # Distribute the remaining integers (if any) among the first few lists
        for i in range(remainder):
            clients_per_list[i] += 1

        complete_candidates = [cluster.copy() for cluster in selected_candidates]

        start_idx = 0
        for i, cluster in enumerate(complete_candidates):
            end_idx = start_idx + clients_per_list[i]
            cluster.extend(extra_clients[start_idx:end_idx])
            start_idx = end_idx

        return complete_candidates

    def normalize_elements(self, input_list):
        total_sum = sum(input_list)
        normalized_list = [element / total_sum for element in input_list]

        return normalized_list
        
class HACCS(ClusteredPowerofchoice):
    def __init__(self, *args, **kwargs):
        super(HACCS, self).__init__(*args, **kwargs)

    def sample_candidates_by_loss(self, client_losses, client_latencies, rho):
        selected_clusters, cluster_keys = self.sample_clusters_by_loss(client_losses, client_latencies, rho)
        selected_candidates = self.sample_candidates_from_clusters_by_latency(selected_clusters)

        # n_selected_candidates = sum([len(cluster) for cluster in selected_candidates])

        # if n_selected_candidates < self.d:
        #     extra_clients = self.fill_with_extra_cluster(client_losses, cluster_keys, n_selected_candidates)

        #     selected_candidates = self.distribute_extra_clients(selected_candidates, extra_clients)

        candidates = [[el[0] for el in result] for result in selected_candidates]
        losses = [[el[1] for el in result] for result in selected_candidates]

        return candidates, losses, cluster_keys

    def sample_clusters_by_loss(self, client_losses, client_latencies, rho):
        keys = list(self.cluster_dict.keys())

        # Assign to each cluster its current average loss
        client_losses_dict = {
            key: ([tup[0] for tup in self.cluster_dict[key]], np.mean([client_losses[tup[0]] for tup in self.cluster_dict[key]]))
            for key in keys
        }

        
        client_latencies_dict = {
            key: ([tup[0] for tup in self.cluster_dict[key]], np.mean([client_latencies[tup[0]] for tup in self.cluster_dict[key]]))
            for key in keys
        }

        latency_max = max(client_latencies.values())

        tau_dict = {
            key: (value[0], 1 - (value[1] / latency_max))
            for key, value in client_latencies_dict.items()
        }

        # Calculate the sum of the second elements (mean losses) for the clusters
        total_loss = sum(value[1] for value in client_losses_dict.values())

        # Apply the normalization over loss
        normalized_losses_dict = {
            key: (value[0], value[1] / total_loss)
            for key, value in client_losses_dict.items()
        }
        
        # Combine tau_dict and normalized_losses_dict using the formula of theta
        theta_dict = {
            key: (tau_dict[key][0], rho * tau_dict[key][1] + (1-rho) * normalized_losses_dict[key][1])
            for key in tau_dict.keys()
        }
        # print(rho)
        # sorted_clusters_by_avg_loss = dict(sorted(client_losses_dict.items(), key=lambda x: x[1][1], reverse=True))     # DESC

        # get probabilities for sampling
        probabilities = np.array([value[1] for value in theta_dict.values()])
        probabilities /= probabilities.sum()
        
        # Select the top w clusters with the higher average local loss
        n_clusters = min(self.w, len(keys))
        
        # Select a sample of keys based on the given probabilities without replacement
        selected_keys = list(np.random.choice(keys, size=n_clusters, replace=False, p=probabilities))
        # print(probabilities)
        # print(keys)
        # print(n_clusters)
        # print(len(selected_keys))
        # print(selected_keys)
        
        top_clusters_loss = {key: client_losses_dict[key] for key in selected_keys}
        top_clusters_latency = {key: client_latencies_dict[key] for key in selected_keys}
        cluster_keys = list([int(key) for key in top_clusters_loss.keys()])
        # selected_clusters = [[(client, client_losses[client]) for client in top_clusters_loss[cluster][0]] for cluster in top_clusters_loss]
        
        # Select information needed for clusters
        selected_clusters = [
            [(client, client_losses[client], top_clusters_latency[cluster][1]) for client in top_clusters_loss[cluster][0]]
            for cluster in top_clusters_loss
        ]

        return selected_clusters, cluster_keys

    def sample_candidates_from_clusters_by_latency(self, selected_clusters):
        # Select out of each cluster the clients with the highest local loss (up to "z")
        # selected_candidates = [
        #     sorted(inner_list, key=lambda x: x[1], reverse=True)[:min(self.z, len(inner_list))]     # DESC
        #     for inner_list in selected_clusters
        # ]

        selected_candidates = [
            sorted(inner_list, key=lambda x: x[2], reverse=False)[:1]     # ASC
            for inner_list in selected_clusters
        ]

        return selected_candidates


class HACCSSerialClientTrainer(SGDSerialClientTrainer):
    def evaluate(self, id_lists, model_parameters):
        self.set_model(model_parameters)
        loss_function = torch.nn.CrossEntropyLoss()
        losses = {}
        for id_list in id_lists:
            for id in id_list:
                dataloader = DataLoader(self.dataset[id], batch_size=self.batch_size)
                loss, _ = evaluate(self._model, loss_function, dataloader)
                losses[id] = loss

        return losses


class HACCSPipeline(FedLabStandardPipeline):
    def main(self, min_comm_round):
        t = 0
        time_idx = 0
        accuracy_thresholds = np.linspace(0.2, 0.8, num=100)
        client_list = [[client[0] for client in cluster] for cluster in self.handler.cluster_dict.values()]

        desired_accuracy = accuracy_thresholds[time_idx]
        start_time = time.time()
        
        # while self.handler.if_stop is False:
        for comm in range(0,comm_round):

            if min_comm_round == 0 or t < min_comm_round:
                
                # compute all local losses
                client_losses = self.trainer.evaluate(client_list, self.handler.model_parameters)
                # client_latencies = self.trainer.evaluate(client_list, self.handler.model_parameters)
                client_latencies = generateDelays(client_list)
                self.communication_size += (sys.getsizeof(self.handler.model_parameters) * (len(client_losses) + len(client_latencies)))                                # send current parameters to args.K clients
                self.communication_size += sys.getsizeof(client_losses)                                                                       # receive local losses from args.K clients

                candidates, losses, clusters = self.handler.sample_candidates_by_loss(client_losses, client_latencies, self.handler.rho)
            else:
                # compute local loss only for the selected candidates
                candidates, clusters = self.handler.sample_candidates()
                self.communication_size += (sys.getsizeof(self.handler.model_parameters) * sum([len(cluster) for cluster in candidates]))     # send current parameters to each candidate (exactly args.d)
                losses = self.trainer.evaluate(candidates, self.handler.model_parameters)
                self.communication_size += sys.getsizeof(losses)                                                                              # receive local losses from each candidate (exactly args.d)

            sampled_clients = self.handler.sample_clients(candidates, losses)
            self.selected_clients.append(sampled_clients)
            self.selected_clusters.append(clusters)
            broadcast = self.handler.downlink_package

            # client side
            self.communication_size += (sys.getsizeof(broadcast) * self.handler.num_clients_per_round)                                        # send current parameters to args.m clients
            self.trainer.local_process(broadcast, sampled_clients, t)
            uploads = self.trainer.uplink_package
            self.communication_size += sys.getsizeof(uploads)                                                                                 # receive updated parameters from args.m clients

            # server side
            for pack in uploads:
                self.handler.load(pack)

            client_loss, client_accuracy = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
            print("Round {}, Validation Loss {:.4f}, Validation Accuracy {:.4f}".format(t, client_loss, client_accuracy))

            if client_accuracy >= accuracy_thresholds[time_idx]:
                elapsed_time = time.time() - start_time
                self.time_list.append((accuracy_thresholds[time_idx], elapsed_time))

                desired_accuracy = accuracy_thresholds[time_idx]
                time_idx += 1
            else:
                self.time_list.append(self.time_list[-1])

            t += 1
            self.loss.append(client_loss)
            self.acc.append(client_accuracy)
            




LOCK_TRACE = False
EXPECTED_EPOCH_DURATION = 10.0

def generateNetworkDelays(n=30):

    MODEL_SIZE_IN_BITS = float(8 * 10 * 1024 * 1024)   # 10 MB

    # np.random.seed(1111)
    delays = []

    # fast, medium, slow, very slow probabilities
    probs = np.array([0.6, 0.2, 0.14, 0.06])
    samples = np.random.multinomial(1, probs, n)

    for sample in samples:

        # Just generate a random latency in seconds (20-200ms)
        latency = np.random.uniform(0.02, 0.2)

        # BW values in megabits per sec
        draw = np.array([np.random.uniform(75.0, 100.0),
                         np.random.uniform(50.0,  75.0),
                         np.random.uniform(25.0,  50.0),
                         np.random.uniform( 1.0,  25.0)])

        bw_arr = np.multiply(sample, draw)
        bw_idx = np.nonzero(bw_arr)
        bw = float(bw_arr[bw_idx]) * 1024.0 * 1024.0

        nw_delay = latency + (MODEL_SIZE_IN_BITS / bw)
        delays.append(nw_delay)

    return delays

                
def generateDelays(id_lists):
    # Get number of clients
    n = sum(len(sublist) for sublist in id_lists)
    
    global EXPECTED_EPOCH_DURATION

    # np.random.seed(1111)
    delays = []

    nw_delays = generateNetworkDelays(n)

    # fast, medium, slow, very slow probabilities
    probs = np.array([0.6, 0.2, 0.14, 0.06])
    samples = np.random.multinomial(1, probs, n)

    for idx, sample in enumerate(samples):
        draw = np.array([np.random.uniform(0.0, 0.5),
                         np.random.uniform(0.5, 1.0),
                         np.random.uniform(1.0, 1.5),
                         np.random.uniform(1.5, 2.0)])

        delay_arr = np.multiply(sample, draw)
        delay_idx = np.nonzero(delay_arr)
        cpu_delay = float(delay_arr[delay_idx]) * float(EXPECTED_EPOCH_DURATION)
        delays.append(cpu_delay + nw_delays[idx])
    
    delays_dict = {}
    for id_list in id_lists:
        for id in id_list:
            delays_dict[id] = delays[id]    
    
    return delays_dict







def encoding(label_list, all_unique_labels):
    """
    Converts a list of labels into a one-hot encoding matrix based on all unique labels.
    """
    label_to_onehot = {label: np.eye(len(all_unique_labels))[i] for i, label in enumerate(all_unique_labels)}
    
    # Create one-hot encoded representation for each label in label_list
    onehot_encoded = np.zeros(len(all_unique_labels))
    
    for label in label_list:
        onehot_encoded += label_to_onehot[label]
        
    return onehot_encoded / len(label_list)  # Normalize by the number of labels for the client

def hamming_distance(vec1, vec2):
    """
    Calculates the Hamming distance between two binary vectors.
    """
    return np.sum(vec1 != vec2)

def clustering(Z, threshold=0.2):
    """
    Performs clustering based on the Hamming distance with a threshold for similarity.
    """
    N = len(Z)
    clusters = []
    cluster_labels = [-1] * N  # Initialize with -1, indicating unclustered clients
    current_cluster = 0
    
    for i in range(N):
        if cluster_labels[i] == -1:  # If client i is not yet in a cluster
            # Start a new cluster
            cluster = [i]
            cluster_labels[i] = current_cluster
            for j in range(i + 1, N):
                if cluster_labels[j] == -1:  # If client j is not yet in a cluster
                    d = hamming_distance(Z[i], Z[j])
                    # If Hamming distance is below threshold, put them in the same cluster
                    if d <= threshold * len(Z[i]):  # Normalized threshold based on vector length
                        cluster.append(j)
                        cluster_labels[j] = current_cluster
            
            # Check if the cluster has only one client; if so, reset its label to -1
            if len(cluster) == 1:
                cluster_labels[i] = -1
            else:
                clusters.append(cluster)
                current_cluster += 1

    return clusters, cluster_labels

def clients_clustering(label_per_client, threshold=0.2):
    """
    Main function to apply the clients clustering algorithm with a distance threshold.
    """
    # Step 1: Gather all unique labels across all clients
    all_unique_labels = sorted(set(label for client_labels in label_per_client for label in client_labels))
    
    # Step 2: Encode labels for each client
    Z = [encoding(client_labels, all_unique_labels) for client_labels in label_per_client]
    
    # Step 3: Perform clustering with a threshold
    clusters, cluster_labels = clustering(Z, threshold)
    
    return clusters, cluster_labels


class ClusteredPowerofchoiceSerialClientTrainer(SGDSerialClientTrainer):
    def evaluate(self, id_lists, model_parameters):
        self.set_model(model_parameters)
        loss_function = torch.nn.CrossEntropyLoss()
        losses = {}
        for id_list in id_lists:
            for id in id_list:
                dataloader = DataLoader(self.dataset[id], batch_size=self.batch_size)
                loss, _ = evaluate(self._model, loss_function, dataloader)
                losses[id] = loss

        return losses
        
class FedCLS(ClusteredPowerofchoice):
    def __init__(self, *args, **kwargs):
        super(FedCLS, self).__init__(*args, **kwargs)

    def sample_candidates_by_loss(self, client_losses):
        selected_clusters, cluster_keys = self.sample_clusters_by_loss(client_losses)
        selected_candidates = self.sample_candidates_from_clusters_by_loss(selected_clusters)

        n_selected_candidates = sum([len(cluster) for cluster in selected_candidates])

        if n_selected_candidates < self.d:
            extra_clients = self.fill_with_extra_cluster(client_losses, cluster_keys, n_selected_candidates)

            selected_candidates = self.distribute_extra_clients(selected_candidates, extra_clients)

        candidates = [[el[0] for el in result] for result in selected_candidates]
        losses = [[el[1] for el in result] for result in selected_candidates]

        return candidates, losses, cluster_keys

    def sample_clusters_by_loss(self, client_losses):
        keys = list(self.cluster_dict.keys())

        # Assign to each cluster its current average loss
        client_losses_dict = {
            key: ([tup[0] for tup in self.cluster_dict[key]], np.mean([client_losses[tup[0]] for tup in self.cluster_dict[key]]))
            for key in keys
        }
        sorted_clusters_by_avg_loss = dict(sorted(client_losses_dict.items(), key=lambda x: x[1][1], reverse=True))     # DESC

        # Select the top w clusters with the higher average local loss
        n_clusters = min(self.w, len(keys))
        top_clusters = dict(list(sorted_clusters_by_avg_loss.items())[:n_clusters])
        cluster_keys = list([int(key) for key in top_clusters.keys()])
        selected_clusters = [[(client, client_losses[client]) for client in top_clusters[cluster][0]] for cluster in top_clusters]

        return selected_clusters, cluster_keys

    def sample_candidates_from_clusters_by_loss(self, selected_clusters):
        # Select out of each cluster the clients with the highest local loss (up to "z")
        selected_candidates = [
            sorted(inner_list, key=lambda x: x[1], reverse=True)[:min(self.z, len(inner_list))]     # DESC
            for inner_list in selected_clusters
        ]

        return selected_candidates

class FedCLSPipeline(FedLabStandardPipeline):
    def main(self, min_comm_round):
        t = 0
        time_idx = 0
        accuracy_thresholds = np.linspace(0.2, 0.8, num=100)
        client_list = [[client[0] for client in cluster] for cluster in self.handler.cluster_dict.values()]

        desired_accuracy = accuracy_thresholds[time_idx]
        start_time = time.time()

        while self.handler.if_stop is False:
            if min_comm_round == 0 or t < min_comm_round:
                # compute all local losses
                # client_losses = self.trainer.evaluate(client_list, self.handler.model_parameters)
                
                keys = list(range(0,args.K))
                client_losses = {key: random.uniform(0, 100) for key in keys}
                
                self.communication_size += (sys.getsizeof(self.handler.model_parameters) * len(client_losses))                                # send current parameters to args.K clients
                self.communication_size += sys.getsizeof(client_losses)                                                                       # receive local losses from args.K clients

                candidates, losses, clusters = self.handler.sample_candidates_by_loss(client_losses)
            else:
                # compute local loss only for the seclients_clusteringlected candidates
                candidates, clusters = self.handler.sample_candidates()
                self.communication_size += (sys.getsizeof(self.handler.model_parameters) * sum([len(cluster) for cluster in candidates]))     # send current parameters to each candidate (exactly args.d)
                losses = self.trainer.evaluate(candidates, self.handler.model_parameters)
                self.communication_size += sys.getsizeof(losses)                                                                              # receive local losses from each candidate (exactly args.d)

            sampled_clients = self.handler.sample_clients(candidates, losses)
            self.selected_clients.append(sampled_clients)
            self.selected_clusters.append(clusters)
            broadcast = self.handler.downlink_package

            # client side
            self.communication_size += (sys.getsizeof(broadcast) * self.handler.num_clients_per_round)                                        # send current parameters to args.m clients
            self.trainer.local_process(broadcast, sampled_clients, t)
            uploads = self.trainer.uplink_package
            self.communication_size += sys.getsizeof(uploads)                                                                                 # receive updated parameters from args.m clients

            # server side
            for pack in uploads:
                self.handler.load(pack)

            client_loss, client_accuracy = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
            
            print("Round {}, Validation Loss {:.4f}, Validation Accuracy {:.4f}".format(t, client_loss, client_accuracy))

            if client_accuracy >= accuracy_thresholds[time_idx]:
                elapsed_time = time.time() - start_time
                self.time_list.append((accuracy_thresholds[time_idx], elapsed_time))

                desired_accuracy = accuracy_thresholds[time_idx]
                time_idx += 1
            else:
                self.time_list.append(self.time_list[-1])

            t += 1
            self.loss.append(client_loss)
            self.acc.append(client_accuracy)






##################
#
# CFL Server (Fixed)
#
##################

class CFLServerHandler(SyncServerHandler):
    def __init__(self, model, total_rounds, sample_ratio, 
                 epsilon1=1e-5, epsilon2=0.1, gamma_max=0.5):
        super().__init__(model, total_rounds, sample_ratio)
        self.total_rounds = total_rounds
        self.current_round = 0
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.gamma_max = gamma_max
        self.cluster_gradients = []
        self.child_clusters = []
        self.alpha_matrix = None
        # Convert state_dict to serialized tensor format
        self._model_parameters = SerializationTool.serialize_model(self.model)

    @property
    def model_parameters(self):
        """Return serialized parameters"""
        return self._model_parameters

    @model_parameters.setter
    def model_parameters(self, params):
        """Accept both serialized and deserialized parameters"""
        if isinstance(params, dict):  # Handle state_dict case
            # Load state_dict into model first
            self.model.load_state_dict(params)
            # Then serialize from model
            self._model_parameters = SerializationTool.serialize_model(self.model)
        else:
            # Directly use serialized parameters
            self._model_parameters = params.clone()
        
    def compute_cosine_similarities(self, gradients):
        """Compute pairwise cosine similarities between client gradients"""
        similarities = torch.zeros(len(gradients), len(gradients))
        for i in range(len(gradients)):
            grad_i = gradients[i].flatten()
            for j in range(len(gradients)):
                grad_j = gradients[j].flatten()
                sim = F.cosine_similarity(grad_i, grad_j, dim=0)
                similarities[i,j] = sim.item()
        return similarities

    def optimal_bipartition(self, similarities):
        """Correct implementation matching paper's Algorithm 1"""
        M = similarities.shape[0]
        indices = torch.argsort(similarities.view(-1), descending=True)
        
        # Initialize as individual clusters
        clusters = [set([i]) for i in range(M)]
        
        for idx in indices:
            i = idx // M
            j = idx % M
            
            # Find clusters containing i or j
            found = []
            for c in clusters:
                if i in c or j in c:
                    found.append(c)
            
            # Merge clusters if they're different
            if len(found) > 1:
                clusters.remove(found[0])
                clusters.remove(found[1])
                clusters.append(found[0].union(found[1]))
            elif len(found) == 1:
                found[0].add(i)
                found[0].add(j)
            
            # Stop when we have 2 clusters
            if len(clusters) == 2:
                break
        
        return [list(clusters[0]), list(clusters[1])]

    def check_stopping_criteria(self, gradients):
        """Check equations (30) and (31) from the paper"""
        grad_norms = [torch.norm(g).item() for g in gradients]
        avg_grad_norm = torch.mean(torch.tensor(grad_norms))
        max_grad_norm = max(grad_norms)
        
        return (avg_grad_norm < self.epsilon1) and (max_grad_norm > self.epsilon2)

    def global_update(self, buffer):
        self.current_round += 1
        # Standard FL update
        parameters_list = [ele[0] for ele in buffer]
        new_parameters = Aggregators.fedavg_aggregate(parameters_list)
        self.set_model(new_parameters)
        self.model_parameters = copy.deepcopy(self.model.state_dict())  # Update cluster model
        
        # Store client gradients for clustering
        gradients = [p - self.model_parameters for p in parameters_list]
        self.cluster_gradients.append((gradients, [ele[1] for ele in buffer]))

        cluster_result = None

        # Check if we should perform clustering
        if self.check_stopping_criteria(gradients):
            similarities = self.compute_cosine_similarities(gradients)
            self.alpha_matrix = similarities
            
            # Perform bipartitioning
            client_ids = [ele[1] for ele in buffer]
            cluster1_idx, cluster2_idx = self.optimal_bipartition(similarities)
            
            # Calculate alpha_cross_max
            cross_similarities = similarities[cluster1_idx][:, cluster2_idx]
            alpha_cross_max = cross_similarities.max()
            
            # Check gamma_max criterion (equation 36)
            gamma = torch.sqrt((1 - alpha_cross_max) / 2)
            if gamma > self.gamma_max:
                # Create child clusters with remaining rounds
                remaining_rounds = self.total_rounds - self.current_round
                cluster1_ids = [client_ids[i] for i in cluster1_idx]
                cluster2_ids = [client_ids[i] for i in cluster2_idx]
                
                child1 = CFLServerHandler(
                    copy.deepcopy(self.model),
                    remaining_rounds,
                    self.sample_ratio,
                    self.epsilon1,
                    self.epsilon2,
                    self.gamma_max
                )
                child2 = CFLServerHandler(
                    copy.deepcopy(self.model),
                    remaining_rounds,
                    self.sample_ratio,
                    self.epsilon1,
                    self.epsilon2,
                    self.gamma_max
                )
                
                self.child_clusters = [child1, child2]
                
                cluster_result = {
                    'clusters': [
                        (child1, cluster1_ids),
                        (child2, cluster2_ids)
                    ],
                    'alpha_matrix': similarities
                }

        return cluster_result

##################
#
# CFL Client (Fixed)
#
##################

class CFLSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, cuda=False, device=None, logger=None):
        super().__init__(model, num_clients, cuda, device, logger)
        self.client_parameters = [copy.deepcopy(model.state_dict()) for _ in range(num_clients)]
        self.client_datasets = []

    def setup_dataset(self, dataset):
        """Handle client datasets"""
        # Add validation check
        if len(dataset) != self.num_clients:
            raise ValueError(f"Expected {self.num_clients} client datasets, got {len(dataset)}")
            
        self.client_datasets = dataset

    def setup_optim(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def local_process(self, payload, id_list, server_round):
        model_parameters, cluster_id = payload
        for client_id in id_list:
            # Add bounds checking
            if client_id >= len(self.client_datasets):
                raise IndexError(f"Client {client_id} has no dataset (total clients: {len(self.client_datasets)})")
                
            current_params = copy.deepcopy(model_parameters)
            data_loader = DataLoader(self.client_datasets[client_id], self.batch_size)
            pack = self.train(client_id, current_params, data_loader)
            self.cache.append(pack)

    def train(self, client_id, model_parameters, train_loader):
        # Convert serialized parameters back to model format
        if isinstance(model_parameters, torch.Tensor):
            SerializationTool.deserialize_model(self.model, model_parameters)
        else:
            # Handle existing state_dict format
            self.model.load_state_dict(model_parameters)
            
        self.model.train()
        
        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        
        return [SerializationTool.serialize_model(self.model), client_id]

##################
#
# CFL Pipeline (Fixed)
#
##################

class CFLPipeline(FedLabStandardPipeline):
    def __init__(self, root_handler, trainer, test_loader):
        self.root_handler = root_handler
        self.trainer = trainer
        self.test_loader = test_loader
        self.cluster_tree = []
        self.acc = []
        self.global_round_count = 0

    def evaluate_clusters(self, cluster_handlers):
        """Evaluate each cluster separately as per paper"""
        results = []
        for handler in cluster_handlers:
            total_loss = 0.0
            correct = 0
            total = 0
            handler.model.eval()
            with torch.no_grad():
                for data, target in self.test_loader:
                    if next(handler.model.parameters()).is_cuda:
                        data, target = data.cuda(), target.cuda()
                    output = handler.model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            results.append(correct / total)
        return sum(results)/len(results)  # Average accuracy

    def train_cluster(self, handler, client_ids, depth=0):
        while handler.current_round < handler.total_rounds:
            # Client sampling
            sample_size = max(1, int(handler.sample_ratio * len(client_ids)))
            sampled = np.random.choice(client_ids, sample_size, replace=False)
            
            # Train with cluster-specific model
            self.trainer.local_process((handler.model_parameters, id(handler)), sampled, handler.current_round)
            
            # Process updates
            updates = [p for p in self.trainer.uplink_package]
            if not updates:
                continue
            
            # Cluster handling
            cluster_result = handler.global_update(updates)
            self.global_round_count += 1
            
            # Evaluate all active clusters
            leaf_handlers = self.get_leaf_handlers(self.root_handler)
            accuracy = self.evaluate_clusters(leaf_handlers)
            self.acc.append(accuracy)
            print(f"Global Round {self.global_round_count}, Avg Accuracy: {accuracy:.4f}")

            # Handle cluster splitting
            if cluster_result is not None:
                handler.current_round = handler.total_rounds
                self.cluster_tree.append({
                    'parent': id(handler),
                    'children': [id(c) for c, _ in cluster_result['clusters']],
                    'alpha_matrix': cluster_result['alpha_matrix']
                })
                for child_handler, child_clients in cluster_result['clusters']:
                    self.train_cluster(child_handler, child_clients, depth + 1)
                return

    def get_leaf_handlers(self, handler):
        if not handler.child_clusters:
            return [handler]
        leaves = []
        for child in handler.child_clusters:
            leaves.extend(self.get_leaf_handlers(child))
        return leaves

    def get_accuracy(self):
        return self.acc
        
    def main(self):
        initial_clients = list(range(self.trainer.num_clients))
        self.train_cluster(self.root_handler, initial_clients)
        return self.get_leaf_handlers(self.root_handler)








class FedSoftServerHandler(SyncServerHandler):
    def __init__(self, model, num_clusters, total_rounds, sample_ratio, 
                 num_clients, tau=2, sigma=1e-4, lambda_=0.1):  # Add num_clients here
        super().__init__(
            model=model,
            global_round=total_rounds,
            sample_ratio=sample_ratio
        )
        # Explicitly set num_clients from parameter
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.total_rounds = total_rounds
        self.tau = tau
        self.sigma = sigma
        self.lambda_ = lambda_
        
        # Initialize cluster centers
        self.cluster_models = [copy.deepcopy(model) for _ in range(num_clusters)]
        self.importance_weights = np.ones((num_clients, num_clusters)) / num_clusters
        self.current_round = 0

    @property
    def model_parameters(self):
        # Flatten all cluster parameters into a single tensor
        return torch.cat([
            torch.cat([param.data.view(-1) for param in model.parameters()])
            for model in self.cluster_models
        ])
        
    def estimate_importance_weights(self, selected_clients, client_datasets):
        """Estimate importance weights for selected clients' data"""
        if len(selected_clients) == 0:
            print("Warning: No clients selected for weight estimation")
            return
    
        weights = []
        for idx in selected_clients:
            if idx >= self.num_clients:
                raise ValueError(f"Client index {idx} out of range (0-{self.num_clients-1})")
                
            client_data = client_datasets[idx]
            if len(client_data) == 0:
                raise ValueError(f"Client {idx} has empty dataset")
                
            client_weights = np.zeros(self.num_clusters)
            
            # Calculate loss-based weights
            for x, y in client_data:
                losses = []
                for model in self.cluster_models:
                    with torch.no_grad():
                        output = model(x.unsqueeze(0))
                        loss = torch.nn.functional.cross_entropy(output, y.unsqueeze(0))
                    losses.append(loss.item())
                best_cluster = np.argmin(losses)
                client_weights[best_cluster] += 1
                
            # Apply smoothing and normalization
            client_weights = np.maximum(client_weights / len(client_data), self.sigma)
            client_weights /= client_weights.sum()  # Ensure proper normalization
            weights.append(client_weights)
    
        # Reshape for dimension consistency
        weights_array = np.array(weights).reshape(-1, self.num_clusters)
        
        # Validate array dimensions before assignment
        if weights_array.shape != (len(selected_clients), self.num_clusters):
            raise ValueError(f"Weights shape mismatch: Expected {(len(selected_clients), self.num_clusters)}, "
                             f"got {weights_array.shape}")
        
        # Update weights with defensive copy
        self.importance_weights[selected_clients] = weights_array.copy()

    def aggregate(self, client_models, client_indices):
        """Aggregate client updates using importance weights"""
        aggregated_models = [copy.deepcopy(model) for model in self.cluster_models]
        
        for s in range(self.num_clusters):
            # Calculate aggregation weights
            weights = self.importance_weights[client_indices, s]
            weights /= weights.sum() + 1e-10
            
            # Update cluster model s
            for param in aggregated_models[s].parameters():
                param.data.zero_()
                
            # client_models should be a list of tuples (client_idx, model)
            for idx, (client_idx, model) in enumerate(client_models):
                client_weight = weights[idx]
                for server_param, client_param in zip(aggregated_models[s].parameters(), model.parameters()):
                    server_param.data += client_weight * client_param.data
            
        self.cluster_models = aggregated_models

    def global_update(self, buffer, client_datasets=None):
        self.current_round += 1
        
        # Extract client indices and models from buffer
        client_indices = [item[0] for item in buffer]
        client_models = [item[1] for item in buffer]
        
        # Estimate importance weights periodically
        if self.current_round % self.tau == 0 or self.current_round == 1:
            if client_datasets is None:
                raise ValueError("Client datasets must be provided for weight estimation")
            self.estimate_importance_weights(client_indices, client_datasets)
            
        # Perform aggregation only if we have client models
        if len(client_models) > 0:
            self.aggregate(list(zip(client_indices, client_models)), client_indices)
        
        return self.cluster_models

class FedSoftClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clusters, num_clients, cuda=False):
        super().__init__(model, num_clients, cuda)
        self.num_clusters = num_clusters
        self.cluster_models = [copy.deepcopy(model) for _ in range(num_clusters)]         
        self.client_models = [copy.deepcopy(model) for _ in range(num_clusters)]
        self.importance_weights = None
        self.client_datasets = []
        self.lambda_ = 0.1

    def setup_dataset(self, dataset):
        """Handle client datasets"""
        # Add validation check
        if len(dataset) != self.num_clients:
            raise ValueError(f"Expected {self.num_clients} client datasets, got {len(dataset)}")
            
        self.client_datasets = dataset

    def setup_optim(self, epochs, batch_size, lr):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    
    def local_process(self, payload, id_list):
        cluster_params, self.importance_weights = payload[0], payload[1]
        self.cache = []
        
        # Reconstruct cluster models from serialized parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        cluster_param_tensors = torch.split(cluster_params, total_params)
        
        for i, param_tensor in enumerate(cluster_param_tensors):
            with torch.no_grad():
                current_index = 0
                for param in self.cluster_models[i].parameters():
                    numel = param.numel()
                    param.data = param_tensor[current_index:current_index+numel].view(param.size())
                    current_index += numel
        
        # Train each selected client
        for client_id in id_list:
            # Get client's data loader
            client_dataset = self.client_datasets[client_id]
            data_loader = torch.utils.data.DataLoader(client_dataset, 
                                                     batch_size=self.batch_size, 
                                                     shuffle=True)
            
            # Get client's importance weights
            weights = self.importance_weights[client_id]
            
            # Train personalized model
            trained_model = self.train(client_id, self.cluster_models, weights, data_loader)
            
            # Store the trained model in cache with client ID
            self.cache.append((client_id, copy.deepcopy(trained_model)))  # Note the tuple format

    def train(self, client_id, cluster_models, weights, data_loader):
        # Initialize personalized model as weighted average of cluster models
        personalized_model = copy.deepcopy(self.model)
        
        # Zero out the parameters first
        with torch.no_grad():
            for param in personalized_model.parameters():
                param.data.zero_()
            
            # Weighted average of cluster models
            for s in range(self.num_clusters):
                for p_personal, p_cluster in zip(personalized_model.parameters(), 
                                               cluster_models[s].parameters()):
                    p_personal.data += weights[s] * p_cluster.data
                
        # Proximal training
        personalized_model.train()
        optimizer = torch.optim.SGD(personalized_model.parameters(), lr=self.lr)
        
        for _ in range(self.epochs):
            for x, y in data_loader:
                if self.cuda:
                    x, y = x.cuda(self.device), y.cuda(self.device)
                
                optimizer.zero_grad()
                output = personalized_model(x)
                loss = self.criterion(output, y)
                
                # Add proximal regularization
                prox_term = 0
                for s in range(self.num_clusters):
                    for p, cluster_p in zip(personalized_model.parameters(), 
                                          cluster_models[s].parameters()):
                        prox_term += weights[s] * torch.norm(p - cluster_p)**2
                
                loss += self.lambda_ * prox_term
                loss.backward()
                optimizer.step()
        
        return personalized_model

class FedSoftPipeline(FedLabStandardPipeline):
    def __init__(self, handler, trainer, test_loader, cuda=False):
        super().__init__(handler, trainer, test_loader)
        self.cluster_accuracies = []
        self.cuda = cuda
        self.device = torch.device("cuda" if cuda else "cpu")
        self.handler = handler
        
    def get_accuracy(self):
        """Returns list of average accuracies from each round"""
        if not self.cluster_accuracies:  # Handle empty case
            return [0.0]  # Return default minimum accuracy
        return [np.mean(acc) for acc in self.cluster_accuracies]
    
    def main(self):
        # Initial evaluation
        print("\nInitial Evaluation:")
        initial_acc = self.evaluate_clusters(self.handler.cluster_models)
        
        for round in range(self.handler.total_rounds):
            print(f"\n=== Round {round + 1}/{self.handler.total_rounds} ===")
            
            # Server selects clients
            selected = self.handler.sample_clients()
            
            # Clients perform local training
            self.trainer.local_process(
                (self.handler.model_parameters, self.handler.importance_weights),
                selected
            )
            
            # Server aggregates updates
            updated_models = self.handler.global_update(
                self.trainer.cache,
                client_datasets=self.trainer.client_datasets
            )
            
            # Evaluate and print cluster models
            current_acc = self.evaluate_clusters(updated_models)
            
    def evaluate_clusters(self, cluster_models):
        accuracies = []
        for i, model in enumerate(cluster_models):
            correct, total = 0, 0
            model.eval()
            with torch.no_grad():
                for x, y in self.test_loader:
                    if self.cuda:
                        x = x.to(self.device)
                        y = y.to(self.device)
                        model = model.to(self.device)
                    output = model(x)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(y).sum().item()
                    total += y.size(0)
            accuracy = correct / total
            accuracies.append(accuracy)
            print(f"Cluster {i+1} Accuracy: {accuracy:.4f}")
        
        # Calculate and print average accuracy
        avg_accuracy = np.mean(accuracies)
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        
        self.cluster_accuracies.append(accuracies)
        return accuracies








class ClientDataset(Dataset):
    """
    Dataset class for creating PyTorch datasets for each client in Federated Learning.

    Args:
        features (np.ndarray): Array of input features for the dataset.
        labels (np.ndarray): Array of labels for the dataset.
        transform (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in a target and transforms it.
    """

    def __init__(self, features, labels, transform=None, target_transform=None):
        # Assuming that features and labels are already in the form of np.ndarrays
        self.features = features
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (feature, label) where label is the label of the feature.
        """
        feature = self.features[index]
        label = self.labels[index]

        if self.transform is not None:
            feature = self.transform(feature)
        if self.target_transform is not None:
            label = self.target_transform(label)

        # Convert features and labels to torch tensors
        feature = torch.from_numpy(feature).float()
        label = torch.tensor(label).long()

        return feature, label

    def __len__(self):
        return len(self.labels)





class Powerofchoice(SyncServerHandler):
    def __init__(self, prob_distribution, *args, **kwargs):
        super(Powerofchoice, self).__init__(*args, **kwargs)

        self.prob_distribution = prob_distribution

    def setup_optim(self, d):
        self.d = d

    def sample_candidates(self):
        selected_clients = np.random.choice(self.num_clients,
                                            size=self.d,
                                            replace=False,
                                            p=self.prob_distribution)

        selection = sorted(selected_clients)    # ASC
        return selection

    def sample_clients(self, candidates, losses):
        sort = np.array(losses).argsort().tolist()
        sort.reverse()
        selected_clients = np.array(candidates)[sort][0:self.num_clients_per_round]
        return selected_clients.tolist()

class PowerofchoiceSerialClientTrainer(SGDSerialClientTrainer):
    def evaluate(self, id_list, model_parameters):
        self.set_model(model_parameters)
        loss_function = torch.nn.CrossEntropyLoss()
        losses = []
        for id in id_list:
            dataloader = DataLoader(self.dataset[id], batch_size=self.batch_size)
            loss, _ = evaluate(self._model, loss_function, dataloader)
            losses.append(loss)
        return losses

        
class PowerofchoicePipeline(FedLabStandardPipeline):
    def main(self):
        t = 0
        time_idx = 0
        accuracy_thresholds = np.linspace(0.2, 0.8, num=100)

        desired_accuracy = accuracy_thresholds[time_idx]
        start_time = time.time()

        # while self.handler.if_stop is False:
        for epoch in range(self.handler.global_round):    
            candidates = self.handler.sample_candidates()
            self.communication_size += (sys.getsizeof(self.handler.model_parameters) * len(candidates))
            losses = self.trainer.evaluate(candidates, self.handler.model_parameters)
            self.communication_size += sys.getsizeof(losses)

            # server side
            sampled_clients = self.handler.sample_clients(candidates, losses)
            self.selected_clients.append(sampled_clients)
            broadcast = self.handler.downlink_package

            # client side
            self.communication_size += (sys.getsizeof(broadcast) * self.handler.num_clients_per_round)
            self.trainer.local_process(broadcast, sampled_clients, t)
            uploads = self.trainer.uplink_package
            self.communication_size += sys.getsizeof(uploads)

            # server side
            for pack in uploads:
                self.handler.load(pack)

            client_loss, client_accuracy = evaluate(self.handler.model, nn.CrossEntropyLoss(), self.test_loader)
            print("Round {}, Validation Loss {:.4f}, Validation Accuracy {:.4f}".format(t, client_loss, client_accuracy))

            if client_accuracy >= accuracy_thresholds[time_idx]:
                elapsed_time = time.time() - start_time
                self.time_list.append((accuracy_thresholds[time_idx], elapsed_time))

                desired_accuracy = accuracy_thresholds[time_idx]
                time_idx += 1
            else:
                self.time_list.append(self.time_list[-1])

            t += 1
            self.loss.append(client_loss)
            self.acc.append(client_accuracy)

class LRModel_pytorch(nn.Module):
    def __init__(self, shape, classes, random_state=None):
        super(LRModel_pytorch, self).__init__()
        
        # Set random seed for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)
        
        # Define the linear layer
        self.fc = nn.Linear(shape, classes)

    def forward(self, x):
        # Apply the linear layer
        x = self.fc(x)
        return x

def evaluate_LR_FL_pytorch(curr_global_params_retrieved, curr_list_x_test, curr_list_y_test):
    """
    Evaluate the global model on each client's test data using the global parameters.
    
    Args:
        curr_global_params_retrieved: The global model parameters retrieved from the server.
        curr_list_x_test: A list containing test data (features) for each client.
        curr_list_y_test: A list containing test labels for each client.
    
    Returns:
        A dictionary with two keys:
        - 'client_metrics': A dictionary where keys are client indices and values are tuples of metrics 
          (loss, accuracy, precision, recall, f1score) and the number of examples for each client.
        - 'weighted_averages': A dictionary containing the weighted average of each metric across all clients.
    """
    # Initialize results storage
    client_metrics = {}

    # Iterate through each client's test data
    for client_id, (x_test_client, y_test_client) in enumerate(zip(curr_list_x_test, curr_list_y_test)):
        # Convert test data to PyTorch tensors
        curr_x_test_tensor = torch.tensor(np.array(x_test_client), dtype=torch.float32)
        # curr_y_test_tensor = torch.tensor(np.array(y_test_client), dtype=torch.long)
        curr_y_test_tensor = torch.tensor(np.array(y_test_client, dtype=np.int64), dtype=torch.long)
        # Create a DataLoader
        curr_test_dataset = TensorDataset(curr_x_test_tensor, curr_y_test_tensor)
        curr_test_loader = DataLoader(curr_test_dataset, batch_size=batch_size, shuffle=False)

        # Define model
        net = LRModel_pytorch(curr_x_test_tensor.shape[1], len(torch.unique(y_test_tensor)))
            
            
        # Load global parameters
        SerializationTool.deserialize_model(net, curr_global_params_retrieved)

        # Evaluate the model on this client's test data
        loss, accuracy = evaluate(net, torch.nn.CrossEntropyLoss(), curr_test_loader)
        precision, recall, f1score = 0, 0, 0
        # Store metrics and number of examples for this client
        client_metrics[client_id] = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1score": f1score,
            "num_examples": len(x_test_client)
        }
    
    # Calculate weighted averages
    total_examples = sum(client["num_examples"] for client in client_metrics.values())
    
    weighted_averages = {
        "loss": sum(client["loss"] * client["num_examples"] for client in client_metrics.values()) / total_examples,
        "accuracy": sum(client["accuracy"] * client["num_examples"] for client in client_metrics.values()) / total_examples,
        "precision": sum(client["precision"] * client["num_examples"] for client in client_metrics.values()) / total_examples,
        "recall": sum(client["recall"] * client["num_examples"] for client in client_metrics.values()) / total_examples,
        "f1score": sum(client["f1score"] * client["num_examples"] for client in client_metrics.values()) / total_examples
    }

    return {
        "client_metrics": client_metrics,
        "weighted_averages": weighted_averages
    }





# This function can be used in order to get the value of desired metric
def retrieve_global_metrics(
    hist: None,
    metric_type: None,
    metric: None,
    best_metric: None,
) -> None:
    """Function to plot from Flower server History.
    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    metric_type : Literal["centralized", "distributed"]
        Type of metric to retrieve.
    metric : Literal["accuracy","precision","recall","f1score"]
        Metric to retrieve.
    """
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    rounds, values = zip(*metric_dict[metric])
    if best_metric:
      metric_return = max(values)
    else:
      metric_return = values[-1]
    return metric_return
    
def test_model(model, X_test, Y_test):
    cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
    logits = model.predict(X_test, batch_size=32, verbose=0, callbacks=[GarbageCollectorCallback()])
    y_pred = tf.argmax(logits, axis=1)
    loss = cce(Y_test, logits).numpy()
    acc = accuracy_score(y_pred, Y_test)
    pre = precision_score(y_pred, Y_test, average='weighted',zero_division = 0)
    rec = recall_score(y_pred, Y_test, average='weighted',zero_division = 0)
    f1s = f1_score(y_pred, Y_test, average='weighted',zero_division = 0)

    return loss, acc, pre, rec, f1s

# Custom Callback To Include in Callbacks List At Training Time
class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        
def LRModel(shape, classes):
    initializer = tf.keras.initializers.GlorotUniform(seed=random_state)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(classes, activation='softmax', input_shape=(shape,), kernel_initializer=initializer)
    ])

    return model

# Define local training/evaluation function
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, epochs_client, partition_id) -> None:
        self.partition_id = partition_id
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.epochs_client = epochs_client

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, validation_split=0.2, epochs=self.epochs_client, verbose=2)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=2)
        return loss, len(self.x_test), {"accuracy": acc}

# The `evaluate` function will be called by Flower after every round
def evaluate_LR_CL(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    global global_params_retrieved  # Declare that the global variable

    # Define model
    if dataset_used == "acs_income" or dataset_used == "dutch":
        net = LRModel(len(covariates), len(np.unique(y_all)))
    elif dataset_used == "sent140":
        net = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
    elif dataset_used == "celeba":
        net = DNN_celeba(input_shape=X_test[0].shape, num_classes=len(np.unique(y_all)))
            
    net.set_weights(parameters)  # Update model with the latest parameters
    loss, accuracy, precision, recall, f1score = test_model(net, X_test, np.array(Y_test))
    global_params_retrieved = parameters  # Update the global variable
    print(f"@@@@@@ Server-side evaluation loss {loss} / accuracy {accuracy} / f1score {f1score} @@@@@@")
    return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1score": f1score}
    
# The `evaluate` function will be called by Flower after every round
def evaluate_LR_CL_clust(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    global global_params_retrieved  # Declare that the global variable
    
    # Define model
    if dataset_used == "acs_income" or dataset_used == "dutch":
        net = LRModel(len(covariates), len(np.unique(y_all)))
    elif dataset_used == "sent140":
        net = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
    elif dataset_used == "celeba":
        net = DNN_celeba(input_shape=X_test[0].shape, num_classes=len(np.unique(y_all)))
        
    net.set_weights(parameters)  # Update model with the latest parameters
    loss, accuracy, precision, recall, f1score = test_model(net, X_test_clust, np.array(Y_test_clust))
    global_params_retrieved = parameters  # Update the global variable
    print(f"@@@@@@ Server-side evaluation loss {loss} / accuracy {accuracy} / f1score {f1score} @@@@@@")
    return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1score": f1score}
    
def evaluate_LR_FL(global_params_retrieved, curr_list_x_test, curr_list_y_test):
    """
    Evaluate the global model on each client's test data using the global parameters.
    
    Args:
        global_params_retrieved: The global model parameters retrieved from the server.
        curr_list_x_test: A list containing test data (features) for each client.
        curr_list_y_test: A list containing test labels for each client.
    
    Returns:
        A dictionary with two keys:
        - 'client_metrics': A dictionary where keys are client indices and values are tuples of metrics 
          (loss, accuracy, precision, recall, f1score) and the number of examples for each client.
        - 'weighted_averages': A dictionary containing the weighted average of each metric across all clients.
    """
    # Initialize results storage
    client_metrics = {}

    # Iterate through each client's test data
    for client_id, (x_test_client, y_test_client) in enumerate(zip(curr_list_x_test, curr_list_y_test)):
        # Convert test data to numpy arrays if not already
        x_test_client = np.array(x_test_client, dtype=float)
        y_test_client = np.array(y_test_client, dtype=float)

        # Define model
        if dataset_used == "acs_income" or dataset_used == "dutch":
            net = LRModel(len(covariates), len(np.unique(y_all)))
        elif dataset_used == "sent140":
            net = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
        elif dataset_used == "celeba":
            net = DNN_celeba(input_shape=X_test[0].shape, num_classes=len(np.unique(y_all)))
            
        # Set the global weights to the model
        net.set_weights(global_params_retrieved)

        # Evaluate the model on this client's test data
        loss, accuracy, precision, recall, f1score = test_model(net, x_test_client, y_test_client)

        # Store metrics and number of examples for this client
        client_metrics[client_id] = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1score": f1score,
            "num_examples": len(x_test_client)
        }

    # Calculate weighted averages
    total_examples = sum(client["num_examples"] for client in client_metrics.values())
    
    weighted_averages = {
        "loss": sum(client["loss"] * client["num_examples"] for client in client_metrics.values()) / total_examples,
        "accuracy": sum(client["accuracy"] * client["num_examples"] for client in client_metrics.values()) / total_examples,
        "precision": sum(client["precision"] * client["num_examples"] for client in client_metrics.values()) / total_examples,
        "recall": sum(client["recall"] * client["num_examples"] for client in client_metrics.values()) / total_examples,
        "f1score": sum(client["f1score"] * client["num_examples"] for client in client_metrics.values()) / total_examples
    }

    return {
        "client_metrics": client_metrics,
        "weighted_averages": weighted_averages
    }



# Define function to convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
def from_FedArtML_to_Flower_format(clients_dict):
  # initialize list that contains clients (features and labels) to extract later from client_fn in Flower
  list_x_train = []
  list_y_train = []

  # Get the name of the clients from the dictionary
  client_names = list(clients_dict.keys())

  # Iterate over each client
  for client in client_names:
    # Get data from each client
    each_client_train=np.array(clients_dict[client],dtype=object)

    # Extract features for each client
    feat=[]
    x_tra=np.array(each_client_train[:, 0])
    for row in x_tra:
      feat.append(row)
    feat=np.array(feat)

    # Extract labels from each client
    y_tra=np.array(each_client_train[:, 1])

    # Append in list features and labels to extract later from client_fn in Flower
    list_x_train.append(feat)
    list_y_train.append(y_tra)

  return list_x_train, list_y_train


def train_psi_pfl(clients_glob, clients_glob_test, local_nodes_glob, random_state, seeds, epochs, comms_round, psi_thresholds):
    print("\nCalculating PSI...")

    # Calculate PSI for clients
    psi_per_client_l, psi_per_client_per_class_l,global_freq_l, group_tables_l, wpsi_l = calculate_psi_label_skew(clients_glob)
    print("PSI per client:", psi_per_client_l)
    print("Global frequencies:", global_freq_l)
    print("Weighted PSI (WPSI_L):", wpsi_l)

    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(clients_dict=clients_glob)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_test, list_y_test = from_FedArtML_to_Flower_format(clients_dict=clients_glob_test)
    
    if local_nodes_glob>=50:
        psi_thresholds = list(np.unique(np.percentile(list(psi_per_client_l.values()), psi_thresholds)))

    else:
        psi_thresholds = list(np.unique(np.percentile(list(psi_per_client_l.values()), psi_thresholds)))

    print("psi_thresholds:",psi_thresholds)

    print(f"\nTraining the {agg_method_used} aggregation algorithm using Flower...")
    # Define function to pass to each local node (client)
    def client_fn(context: Context) -> fl.client.Client:
        # Define model
        if dataset_used == "acs_income" or dataset_used == "dutch":
            model = LRModel(len(covariates), len(np.unique(y_all)))
        elif dataset_used == "sent140":
            model = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
        elif dataset_used == "celeba":
            model = DNN_celeba(input_shape=X_test[0].shape, num_classes=len(np.unique(y_all)))
            
        # Set optimizer
        optimizer = Adam(learning_rate=learn_rate)
    
        # Compile model
        model.compile(optimizer=optimizer, loss=loss_inic, metrics=metrics)
    
        # Load train data partition of each client ID (cid)
        x_train_cid = np.array(list_x_train_selected[int(context.node_config['partition-id'])], dtype=float)
        y_train_cid = np.array(list_y_train_selected[int(context.node_config['partition-id'])], dtype=float)

        # Load test data partition of each client ID (cid)
        x_test_cid = np.array(list_x_test[int(context.node_config['partition-id'])], dtype=float)
        y_test_cid = np.array(list_y_test[int(context.node_config['partition-id'])], dtype=float)

        # Create and return client
        return FlowerClient(model, x_train_cid, y_train_cid, x_test_cid, y_test_cid, epochs, context.node_config['partition-id'])
        
    # Initialize variables to store the best results
    best_psi_threshold = None
    best_avg_accuracy = -1  # Initialize to a very low value
    best_results = None
    
    # Loop through each PSI threshold
    for psi_threshold in tqdm(psi_thresholds,desc="Fine tuning"):
        print(f"\nEvaluating for PSI threshold: {psi_threshold}")
        
        # Select clients whose PSI is below the threshold
        selected_clients = [client for client, psi in psi_per_client_l.items() if psi <= psi_threshold]

        # Extract indices of the selected clients
        selected_indices = [int(client.split('_')[1]) - 1 for client in selected_clients]

        # Filter corresponding data from list_x_train and list_y_train
        list_x_train_selected = [list_x_train[i] for i in selected_indices]
        list_y_train_selected = [list_y_train[i] for i in selected_indices]

        # Debugging output
        print(f"Selected clients: {selected_clients}")
        print(f"Selected indices: {selected_indices}")
        print(f"Number of selected clients: {len(selected_clients)}")
        print(f"Filtered x_train: {len(list_x_train_selected)} partitions")
        print(f"Filtered y_train: {len(list_y_train_selected)} partitions")

        # Replicate single-client clusters to 4 copies
        if len(list_x_train_selected) <= 2:
            list_x_train_selected *= 4  # Creates 4 identical copies
            list_y_train_selected *= 4
            print(f"Replicated single client to {len(list_x_train_selected)} clients")
            
        # Initialize storage for metrics and training times
        global_acc_tests, global_pre_tests, global_rec_tests, global_f1s_tests = [], [], [], []
        weighted_global_acc_tests, weighted_global_pre_tests, weighted_global_rec_tests, weighted_global_f1s_tests = [], [], [], []
        training_times = []
        all_commun_metrics_histories = []  # For saving history of all communication rounds

        # Initialize dataframe to store results for this psi_threshold
        threshold_results = pd.DataFrame(columns=[
            'agg_method_used', 'psi_threshold', 'seed', 'training_time', 
            'global_acc_test', 'global_pre_test', 'global_rec_test', 'global_f1s_test',
            'weighted_global_acc_test', 'weighted_global_pre_test', 'weighted_global_rec_test', 'weighted_global_f1s_test'
        ])

        metrics_per_client_per_seed = []
        model_parameters_per_seed = [] 
        
        # Train model for each seed
        for seed in tqdm(seeds,desc=f"\nTuning parameter {psi_threshold}\n"):
            
            print(f"\nRunning simulation with seed {seed}\n")
            
            np.random.seed(seed)
            tf.random.set_seed(seed)
            
            # Create Federated strategy
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=0.5,
                fraction_evaluate=0.5,
                evaluate_fn=evaluate_LR_CL
            )
            
            # Start training simulation
            start_time = time.time()
            commun_metrics_history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=len(list_x_train_selected),
                config=fl.server.ServerConfig(num_rounds=comms_round),
                strategy=strategy,
                ray_init_args={"num_cpus": CPUs_to_use},                
            )
            training_time = time.time() - start_time
            
            # Append communication metrics history
            all_commun_metrics_histories.append(commun_metrics_history)

            # Retrieve metrics
            global_acc_test = retrieve_global_metrics(commun_metrics_history, "centralized", "accuracy", False)
            global_pre_test = retrieve_global_metrics(commun_metrics_history, "centralized", "precision", False)
            global_rec_test = retrieve_global_metrics(commun_metrics_history, "centralized", "recall", False)
            global_f1s_test = retrieve_global_metrics(commun_metrics_history, "centralized", "f1score", False)
        
            # Calculate metrics per client (to do weighted average)
            metrics_weighted = evaluate_LR_FL(global_params_retrieved, list_x_test, list_y_test)
            metrics_per_client = metrics_weighted["client_metrics"]
            weighted_averages = metrics_weighted["weighted_averages"]
            
            # Retrieve metrics
            weighted_global_acc_test = weighted_averages['accuracy']
            weighted_global_pre_test = weighted_averages['precision']
            weighted_global_rec_test = weighted_averages['recall']
            weighted_global_f1s_test = weighted_averages['f1score']
       
            # Store results
            global_acc_tests.append(global_acc_test)
            global_pre_tests.append(global_pre_test)
            global_rec_tests.append(global_rec_test)
            global_f1s_tests.append(global_f1s_test)
            weighted_global_acc_tests.append(weighted_global_acc_test)
            weighted_global_pre_tests.append(weighted_global_pre_test)
            weighted_global_rec_tests.append(weighted_global_rec_test)
            weighted_global_f1s_tests.append(weighted_global_f1s_test)   
            training_times.append(training_time)

            metrics_per_client_per_seed.append(dict(metrics_per_client))
            model_parameters_per_seed.append(global_params_retrieved)
            
            new_row = pd.DataFrame([{
                'agg_method_used': agg_method_used,
                'psi_threshold': psi_threshold,
                'seed': seed,
                'training_time': training_time,
                'global_acc_test': global_acc_test,
                'global_pre_test': global_pre_test,
                'global_rec_test': global_rec_test,
                'global_f1s_test': global_f1s_test,
                'weighted_global_acc_test': weighted_global_acc_test,
                'weighted_global_pre_test': weighted_global_pre_test,
                'weighted_global_rec_test': weighted_global_rec_test,
                'weighted_global_f1s_test': weighted_global_f1s_test                
            }])
            
            threshold_results = pd.concat([threshold_results, new_row], ignore_index=True)

        # Compute means and standard deviations
        avg_global_acc_test = np.mean(global_acc_tests)
        std_global_acc_test = np.std(global_acc_tests)
        
        avg_global_pre_test = np.mean(global_pre_tests)
        std_global_pre_test = np.std(global_pre_tests)
        
        avg_global_rec_test = np.mean(global_rec_tests)
        std_global_rec_test = np.std(global_rec_tests)
        
        avg_global_f1s_test = np.mean(global_f1s_tests)
        std_global_f1s_test = np.std(global_f1s_tests)
        
        avg_training_time = np.mean(training_times)
        std_training_time = np.std(training_times)

        # Compute means and standard deviations for weight values
        weighted_avg_global_acc_test = np.mean(weighted_global_acc_tests)
        weighted_std_global_acc_test = np.std(weighted_global_acc_tests)
        
        weighted_avg_global_pre_test = np.mean(weighted_global_pre_tests)
        weighted_std_global_pre_test = np.std(weighted_global_pre_tests)
        
        weighted_avg_global_rec_test = np.mean(weighted_global_rec_tests)
        weighted_std_global_rec_test = np.std(weighted_global_rec_tests)
        
        weighted_avg_global_f1s_test = np.mean(weighted_global_f1s_tests)
        weighted_std_global_f1s_test = np.std(weighted_global_f1s_tests)
    
        # Average metrics history across seeds
        avg_commun_metrics_history = {
            "metrics_centralized": {
                metric: [(round_idx, np.mean([h.metrics_centralized[metric][round_idx][1]
                                                 for h in all_commun_metrics_histories]))
                         for round_idx in range(comms_round + 1)]
                for metric in ["accuracy", "precision", "recall", "f1score"]
            }
        }

        

        if avg_global_acc_test > best_avg_accuracy:
            best_avg_accuracy = avg_global_acc_test
            best_psi_threshold = psi_threshold
            
            # Save all best results
            best_results = {
                'psi_threshold': psi_threshold,
                'metrics': {
                    'accuracy': {'mean': avg_global_acc_test, 'std': std_global_acc_test},
                    'precision': {'mean': avg_global_pre_test, 'std': std_global_pre_test},
                    'recall': {'mean': avg_global_rec_test, 'std': std_global_rec_test},
                    'f1score': {'mean': avg_global_f1s_test, 'std': std_global_f1s_test},
                    'weighted_accuracy': {'mean': weighted_avg_global_acc_test, 'std': weighted_std_global_acc_test},
                    'weighted_precision': {'mean': weighted_avg_global_pre_test, 'std': weighted_std_global_pre_test},
                    'weighted_recall': {'mean': weighted_avg_global_rec_test, 'std': weighted_std_global_rec_test},
                    'weighted_f1score': {'mean': weighted_avg_global_f1s_test, 'std': weighted_std_global_f1s_test},
                    'metrics_per_client_per_seed': metrics_per_client_per_seed,
                    'model_parameters_per_seed': model_parameters_per_seed,
                },
                'training_time': {'mean': avg_training_time, 'std': std_training_time},
                'avg_commun_metrics_history': avg_commun_metrics_history,
                'selected_indices': selected_indices,
                
            }
    
    best_psi_threshold = best_results['psi_threshold']
            
    # Compute means and standard deviations
    avg_global_acc_test = best_results["metrics"]["accuracy"]["mean"]
    std_global_acc_test = best_results["metrics"]["accuracy"]["std"]
    
    avg_global_pre_test = best_results["metrics"]["precision"]["mean"]
    std_global_pre_test = best_results["metrics"]["precision"]["std"]
    
    avg_global_rec_test = best_results["metrics"]["recall"]["mean"]
    std_global_rec_test = best_results["metrics"]["recall"]["std"]
    
    avg_global_f1s_test = best_results["metrics"]["f1score"]["mean"]
    std_global_f1s_test = best_results["metrics"]["f1score"]["std"]
    
    avg_training_time = best_results["training_time"]["mean"]
    std_training_time = best_results["training_time"]["std"]
    
    weighted_avg_global_acc_test = best_results["metrics"]["weighted_accuracy"]["mean"]
    weighted_std_global_acc_test = best_results["metrics"]["weighted_accuracy"]["std"]
    
    weighted_avg_global_pre_test = best_results["metrics"]["weighted_precision"]["mean"]
    weighted_std_global_pre_test = best_results["metrics"]["weighted_precision"]["std"]
    
    weighted_avg_global_rec_test = best_results["metrics"]["weighted_recall"]["mean"]
    weighted_std_global_rec_test = best_results["metrics"]["weighted_recall"]["std"]
    
    weighted_avg_global_f1s_test = best_results["metrics"]["weighted_f1score"]["mean"]
    weighted_std_global_f1s_test = best_results["metrics"]["weighted_f1score"]["std"]
    
    avg_commun_metrics_history = best_results["avg_commun_metrics_history"]
    selected_indices = best_results["selected_indices"]


    # Print best results
    print("\n\nBEST RESULTS:")
    
    print(f'PSI Thresholds: {psi_thresholds}')
    print(f'Best PSI Threshold: {best_psi_threshold}')
    print(f'Global Accuracy: Mean = {avg_global_acc_test:.4f}, 'f'Std = {std_global_acc_test:.4f}')
    print(f'Global Precision: Mean = {avg_global_pre_test:.4f}, 'f'Std = {std_global_pre_test:.4f}')
    print(f'Global Recall: Mean = {avg_global_rec_test:.4f}, 'f'Std = {std_global_rec_test:.4f}')
    print(f'Global F1-Score: Mean = {avg_global_f1s_test:.4f}, 'f'Std = {std_global_f1s_test:.4f}')
    print("************************************************************************************")
    # print(f'Weighted Accuracy: Mean = {weighted_avg_global_acc_test:.4f}, Std = {weighted_std_global_acc_test:.4f}')
    # print(f'Weighted Precision: Mean = {weighted_avg_global_pre_test:.4f}, Std = {weighted_std_global_pre_test:.4f}')
    # print(f'Weighted Recall: Mean = {weighted_avg_global_rec_test:.4f}, Std = {weighted_std_global_rec_test:.4f}')
    # print(f'Weighted F1-Score: Mean = {weighted_avg_global_f1s_test:.4f}, Std = {weighted_std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')
    print(f'Selected clients: {selected_indices}')









def train_fedavg(clients_glob, clients_glob_test, local_nodes_glob, random_state, seeds, epochs, comms_round):

    print(f"\nTraining the {agg_method_used} aggregation algorithm using Flower...")    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(clients_dict=clients_glob)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_test, list_y_test = from_FedArtML_to_Flower_format(clients_dict=clients_glob_test)
    
    # Initialize storage for metrics and training times
    global_acc_tests, global_pre_tests, global_rec_tests, global_f1s_tests = [], [], [], []
    weighted_global_acc_tests, weighted_global_pre_tests, weighted_global_rec_tests, weighted_global_f1s_tests = [], [], [], []
    
    training_times = []
    all_commun_metrics_histories = []  # For saving history of all communication rounds
    metrics_per_client_per_seed = []
    model_parameters_per_seed = []
    # Define function to pass to each local node (client)
    def client_fn(context: Context) -> fl.client.Client:
        # Define model
        if dataset_used == "acs_income" or dataset_used == "dutch":
            model = LRModel(len(covariates), len(np.unique(y_all)))
        elif dataset_used == "sent140":
            model = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
        elif dataset_used == "celeba":
            model = DNN_celeba(input_shape=X_test[0].shape, num_classes=len(np.unique(y_all)))
    
        # Set optimizer
        optimizer = Adam(learning_rate=learn_rate)
    
        # Compile model
        model.compile(optimizer=optimizer, loss=loss_inic, metrics=metrics)
    
        # Load train data partition of each client ID (cid)
        x_train_cid = np.array(list_x_train[int(context.node_config['partition-id'])], dtype=float)
        y_train_cid = np.array(list_y_train[int(context.node_config['partition-id'])], dtype=float)

        # Load test data partition of each client ID (cid)
        x_test_cid = np.array(list_x_test[int(context.node_config['partition-id'])], dtype=float)
        y_test_cid = np.array(list_y_test[int(context.node_config['partition-id'])], dtype=float)

        # Create and return client
        return FlowerClient(model, x_train_cid, y_train_cid, x_test_cid, y_test_cid, epochs, context.node_config['partition-id'])
        
    # Train model for each seed
    for seed in tqdm(seeds,desc="Looping on seeds"):
        print(f"Running simulation with seed {seed}")
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Create Federated strategy
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=0.5,
            fraction_evaluate=0.5,
            evaluate_fn=evaluate_LR_CL
        )
        
        # Start training simulation
        start_time = time.time()
        commun_metrics_history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=local_nodes_glob,
            config=fl.server.ServerConfig(num_rounds=comms_round),
            strategy=strategy,
            ray_init_args={"num_cpus": CPUs_to_use},            
        )
        training_time = time.time() - start_time
        
        # Append communication metrics history
        all_commun_metrics_histories.append(commun_metrics_history)
        
        # Retrieve metrics
        global_acc_test = retrieve_global_metrics(commun_metrics_history, "centralized", "accuracy", False)
        global_pre_test = retrieve_global_metrics(commun_metrics_history, "centralized", "precision", False)
        global_rec_test = retrieve_global_metrics(commun_metrics_history, "centralized", "recall", False)
        global_f1s_test = retrieve_global_metrics(commun_metrics_history, "centralized", "f1score", False)
        
        # Calculate metrics per client (to do weighted average)
        metrics_weighted = evaluate_LR_FL(global_params_retrieved, list_x_test, list_y_test)
        metrics_per_client = metrics_weighted["client_metrics"]
        weighted_averages = metrics_weighted["weighted_averages"]
        
        # Retrieve metrics
        weighted_global_acc_test = weighted_averages['accuracy']
        weighted_global_pre_test = weighted_averages['precision']
        weighted_global_rec_test = weighted_averages['recall']
        weighted_global_f1s_test = weighted_averages['f1score']
        
        # Store results
        global_acc_tests.append(global_acc_test)
        global_pre_tests.append(global_pre_test)
        global_rec_tests.append(global_rec_test)
        global_f1s_tests.append(global_f1s_test)
        weighted_global_acc_tests.append(weighted_global_acc_test)
        weighted_global_pre_tests.append(weighted_global_pre_test)
        weighted_global_rec_tests.append(weighted_global_rec_test)
        weighted_global_f1s_tests.append(weighted_global_f1s_test)        
        training_times.append(training_time)

        metrics_per_client_per_seed.append(dict(metrics_per_client))
        model_parameters_per_seed.append(global_params_retrieved)

    # Compute means and standard deviations
    avg_global_acc_test = np.mean(global_acc_tests)
    std_global_acc_test = np.std(global_acc_tests)
    
    avg_global_pre_test = np.mean(global_pre_tests)
    std_global_pre_test = np.std(global_pre_tests)
    
    avg_global_rec_test = np.mean(global_rec_tests)
    std_global_rec_test = np.std(global_rec_tests)
    
    avg_global_f1s_test = np.mean(global_f1s_tests)
    std_global_f1s_test = np.std(global_f1s_tests)
    
    avg_training_time = np.mean(training_times)
    std_training_time = np.std(training_times)

    # Compute means and standard deviations for weight values
    weighted_avg_global_acc_test = np.mean(weighted_global_acc_tests)
    weighted_std_global_acc_test = np.std(weighted_global_acc_tests)
    
    weighted_avg_global_pre_test = np.mean(weighted_global_pre_tests)
    weighted_std_global_pre_test = np.std(weighted_global_pre_tests)
    
    weighted_avg_global_rec_test = np.mean(weighted_global_rec_tests)
    weighted_std_global_rec_test = np.std(weighted_global_rec_tests)
    
    weighted_avg_global_f1s_test = np.mean(weighted_global_f1s_tests)
    weighted_std_global_f1s_test = np.std(weighted_global_f1s_tests)
    
    avg_training_time = np.mean(training_times)
    std_training_time = np.std(training_times)
    # Average metrics history across seeds
    avg_commun_metrics_history = {
        "metrics_centralized": {
            metric: [(round_idx + 1, np.mean([h.metrics_centralized[metric][round_idx][1]
                                             for h in all_commun_metrics_histories]))
                     for round_idx in range(comms_round)]
            for metric in ["accuracy", "precision", "recall", "f1score"]
        }
    }

           
    # Final results display
    print("\n\nFINAL RESULTS:")
    print(f'Test Results after {len(seeds)} runs:')
    print(f'Global Accuracy: Mean = {avg_global_acc_test:.4f}, Std = {std_global_acc_test:.4f}')
    print(f'Global Precision: Mean = {avg_global_pre_test:.4f}, Std = {std_global_pre_test:.4f}')
    print(f'Global Recall: Mean = {avg_global_rec_test:.4f}, Std = {std_global_rec_test:.4f}')
    print(f'Global F1-Score: Mean = {avg_global_f1s_test:.4f}, Std = {std_global_f1s_test:.4f}')
    print("************************************************************************************")
    # print(f'Weighted Accuracy: Mean = {weighted_avg_global_acc_test:.4f}, Std = {weighted_std_global_acc_test:.4f}')
    # print(f'Weighted Precision: Mean = {weighted_avg_global_pre_test:.4f}, Std = {weighted_std_global_pre_test:.4f}')
    # print(f'Weighted Recall: Mean = {weighted_avg_global_rec_test:.4f}, Std = {weighted_std_global_rec_test:.4f}')
    # print(f'Weighted F1-Score: Mean = {weighted_avg_global_f1s_test:.4f}, Std = {weighted_std_global_f1s_test:.4f}')
    
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')





def train_fedprox(clients_glob, clients_glob_test, local_nodes_glob, random_state, seeds, epochs, comms_round):

    print(f"\nTraining the {agg_method_used} aggregation algorithm using Flower...")    
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(clients_dict=clients_glob)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_test, list_y_test = from_FedArtML_to_Flower_format(clients_dict=clients_glob_test)
    
    # Define function to pass to each local node (client)
    def client_fn(context: Context) -> fl.client.Client:
        # Define model
        if dataset_used == "acs_income" or dataset_used == "dutch":
            model = LRModel(len(covariates), len(np.unique(y_all)))
        elif dataset_used == "sent140":
            model = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
        elif dataset_used == "celeba":
            model = DNN_celeba(input_shape=X_test[0].shape, num_classes=len(np.unique(y_all)))
    
        # Set optimizer
        optimizer = Adam(learning_rate=0.001)
    
        # Compile model
        model.compile(optimizer=optimizer, loss=loss_inic, metrics=metrics)
    
        # Load data partition of each client ID (cid)
        x_train_cid = np.array(list_x_train[int(context.node_config['partition-id'])], dtype=float)
        y_train_cid = np.array(list_y_train[int(context.node_config['partition-id'])], dtype=float)

        # Load data partition of each client ID (cid)
        x_test_cid = np.array(list_x_test[int(context.node_config['partition-id'])], dtype=float)
        y_test_cid = np.array(list_y_test[int(context.node_config['partition-id'])], dtype=float)

        # Create and return client
        return FlowerClient(model, x_train_cid, y_train_cid, x_test_cid, y_test_cid, epochs, context.node_config['partition-id'])
        
    # Initialize variables to store the best results
    best_mu_threshold = None
    best_avg_accuracy = -1  # Initialize to a very low value
    best_results = None
    
    # Loop through each PSI threshold
    for mu_threshold in tqdm(mu_thresholds,desc="Fine tuning"):
        print(f"\nEvaluating for PSI threshold: {mu_threshold}")

        # Initialize storage for metrics and training times
        global_acc_tests, global_pre_tests, global_rec_tests, global_f1s_tests = [], [], [], []
        weighted_global_acc_tests, weighted_global_pre_tests, weighted_global_rec_tests, weighted_global_f1s_tests = [], [], [], []
        training_times = []
        all_commun_metrics_histories = []  # For saving history of all communication rounds

        # Initialize dataframe to store results for this mu_threshold
        threshold_results = pd.DataFrame(columns=[
            'agg_method_used', 'mu_threshold', 'seed', 'training_time', 
            'global_acc_test', 'global_pre_test', 'global_rec_test', 'global_f1s_test',
            'weighted_global_acc_test', 'weighted_global_pre_test', 'weighted_global_rec_test', 'weighted_global_f1s_test'
        ])

        metrics_per_client_per_seed = []
        model_parameters_per_seed = [] 
        
        # Train model for each seed
        for seed in tqdm(seeds,desc=f"Looping on seeds for tuning parameter{mu_threshold}"):
            print(f"Running simulation with seed {seed}")
            np.random.seed(seed)
            tf.random.set_seed(seed)
            
            # Create Federated strategy
            strategy = fl.server.strategy.FedProx(
                fraction_fit=0.5,
                # min_available_clients=int(0.5 * local_nodes_glob),
                # min_fit_clients=min(int(0.5 * local_nodes_glob),CPUs_to_use),
               
                fraction_evaluate=0.5,
                evaluate_fn=evaluate_LR_CL,
                proximal_mu=mu_threshold
            )
            
            # Start training simulation
            start_time = time.time()
            commun_metrics_history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=len(list_x_train),
                config=fl.server.ServerConfig(num_rounds=comms_round),
                strategy=strategy,
                ray_init_args={"num_cpus": CPUs_to_use},
            )
            training_time = time.time() - start_time
            
            # Append communication metrics history
            all_commun_metrics_histories.append(commun_metrics_history)

            # Retrieve metrics
            global_acc_test = retrieve_global_metrics(commun_metrics_history, "centralized", "accuracy", False)
            global_pre_test = retrieve_global_metrics(commun_metrics_history, "centralized", "precision", False)
            global_rec_test = retrieve_global_metrics(commun_metrics_history, "centralized", "recall", False)
            global_f1s_test = retrieve_global_metrics(commun_metrics_history, "centralized", "f1score", False)

            # Calculate metrics per client (to do weighted average)
            metrics_weighted = evaluate_LR_FL(global_params_retrieved, list_x_test, list_y_test)
            metrics_per_client = metrics_weighted["client_metrics"]
            weighted_averages = metrics_weighted["weighted_averages"]
            
            # Retrieve metrics
            weighted_global_acc_test = weighted_averages['accuracy']
            weighted_global_pre_test = weighted_averages['precision']
            weighted_global_rec_test = weighted_averages['recall']
            weighted_global_f1s_test = weighted_averages['f1score']
        
            # Store results
            global_acc_tests.append(global_acc_test)
            global_pre_tests.append(global_pre_test)
            global_rec_tests.append(global_rec_test)
            global_f1s_tests.append(global_f1s_test)
            weighted_global_acc_tests.append(weighted_global_acc_test)
            weighted_global_pre_tests.append(weighted_global_pre_test)
            weighted_global_rec_tests.append(weighted_global_rec_test)
            weighted_global_f1s_tests.append(weighted_global_f1s_test)             
            training_times.append(training_time)

            metrics_per_client_per_seed.append(dict(metrics_per_client))
            model_parameters_per_seed.append(global_params_retrieved)
        
            new_row = pd.DataFrame([{
                'agg_method_used': agg_method_used,
                'mu_threshold': mu_threshold,
                'seed': seed,
                'training_time': training_time,
                'global_acc_test': global_acc_test,
                'global_pre_test': global_pre_test,
                'global_rec_test': global_rec_test,
                'global_f1s_test': global_f1s_test,
                'weighted_global_acc_test': weighted_global_acc_test,
                'weighted_global_pre_test': weighted_global_pre_test,
                'weighted_global_rec_test': weighted_global_rec_test,
                'weighted_global_f1s_test': weighted_global_f1s_test                   
            }])
            
            threshold_results = pd.concat([threshold_results, new_row], ignore_index=True)
        
        # Compute means and standard deviations
        avg_global_acc_test = np.mean(global_acc_tests)
        std_global_acc_test = np.std(global_acc_tests)
        
        avg_global_pre_test = np.mean(global_pre_tests)
        std_global_pre_test = np.std(global_pre_tests)
        
        avg_global_rec_test = np.mean(global_rec_tests)
        std_global_rec_test = np.std(global_rec_tests)
        
        avg_global_f1s_test = np.mean(global_f1s_tests)
        std_global_f1s_test = np.std(global_f1s_tests)
        
        avg_training_time = np.mean(training_times)
        std_training_time = np.std(training_times)

        # Compute means and standard deviations for weight values
        weighted_avg_global_acc_test = np.mean(weighted_global_acc_tests)
        weighted_std_global_acc_test = np.std(weighted_global_acc_tests)
        
        weighted_avg_global_pre_test = np.mean(weighted_global_pre_tests)
        weighted_std_global_pre_test = np.std(weighted_global_pre_tests)
        
        weighted_avg_global_rec_test = np.mean(weighted_global_rec_tests)
        weighted_std_global_rec_test = np.std(weighted_global_rec_tests)
        
        weighted_avg_global_f1s_test = np.mean(weighted_global_f1s_tests)
        weighted_std_global_f1s_test = np.std(weighted_global_f1s_tests)
        
        # Average metrics history across seeds
        avg_commun_metrics_history = {
            "metrics_centralized": {
                metric: [(round_idx, np.mean([h.metrics_centralized[metric][round_idx][1]
                                                 for h in all_commun_metrics_histories]))
                         for round_idx in range(comms_round + 1)]
                for metric in ["accuracy", "precision", "recall", "f1score"]
            }
        }

        

        if avg_global_acc_test > best_avg_accuracy:
            best_avg_accuracy = avg_global_acc_test
            best_mu_threshold = mu_threshold
            
            # Save all best results
            best_results = {
                'mu_threshold': mu_threshold,
                'metrics': {
                    'accuracy': {'mean': avg_global_acc_test, 'std': std_global_acc_test},
                    'precision': {'mean': avg_global_pre_test, 'std': std_global_pre_test},
                    'recall': {'mean': avg_global_rec_test, 'std': std_global_rec_test},
                    'f1score': {'mean': avg_global_f1s_test, 'std': std_global_f1s_test},
                    'weighted_accuracy': {'mean': weighted_avg_global_acc_test, 'std': weighted_std_global_acc_test},
                    'weighted_precision': {'mean': weighted_avg_global_pre_test, 'std': weighted_std_global_pre_test},
                    'weighted_recall': {'mean': weighted_avg_global_rec_test, 'std': weighted_std_global_rec_test},
                    'weighted_f1score': {'mean': weighted_avg_global_f1s_test, 'std': weighted_std_global_f1s_test},
                    'metrics_per_client_per_seed': metrics_per_client_per_seed,
                    'model_parameters_per_seed': model_parameters_per_seed,
                },
                'training_time': {'mean': avg_training_time, 'std': std_training_time},
                'avg_commun_metrics_history': avg_commun_metrics_history

            }

    
    best_mu_threshold = best_results['mu_threshold']
            
    # Compute means and standard deviations
    avg_global_acc_test = best_results["metrics"]["accuracy"]["mean"]
    std_global_acc_test = best_results["metrics"]["accuracy"]["std"]
    
    avg_global_pre_test = best_results["metrics"]["precision"]["mean"]
    std_global_pre_test = best_results["metrics"]["precision"]["std"]
    
    avg_global_rec_test = best_results["metrics"]["recall"]["mean"]
    std_global_rec_test = best_results["metrics"]["recall"]["std"]
    
    avg_global_f1s_test = best_results["metrics"]["f1score"]["mean"]
    std_global_f1s_test = best_results["metrics"]["f1score"]["std"]
    
    avg_training_time = best_results["training_time"]["mean"]
    std_training_time = best_results["training_time"]["std"]
    
    weighted_avg_global_acc_test = best_results["metrics"]["weighted_accuracy"]["mean"]
    weighted_std_global_acc_test = best_results["metrics"]["weighted_accuracy"]["std"]
    
    weighted_avg_global_pre_test = best_results["metrics"]["weighted_precision"]["mean"]
    weighted_std_global_pre_test = best_results["metrics"]["weighted_precision"]["std"]
    
    weighted_avg_global_rec_test = best_results["metrics"]["weighted_recall"]["mean"]
    weighted_std_global_rec_test = best_results["metrics"]["weighted_recall"]["std"]
    
    weighted_avg_global_f1s_test = best_results["metrics"]["weighted_f1score"]["mean"]
    weighted_std_global_f1s_test = best_results["metrics"]["weighted_f1score"]["std"]
    
    avg_commun_metrics_history = best_results["avg_commun_metrics_history"]
    
    # Print best results
    print("\n\nBEST RESULTS:")
    print(f'Best Mu Threshold: {best_mu_threshold}')
    print(f'Global Accuracy: Mean = {avg_global_acc_test:.4f}, 'f'Std = {std_global_acc_test:.4f}')
    print(f'Global Precision: Mean = {avg_global_pre_test:.4f}, 'f'Std = {std_global_pre_test:.4f}')
    print(f'Global Recall: Mean = {avg_global_rec_test:.4f}, 'f'Std = {std_global_rec_test:.4f}')
    print(f'Global F1-Score: Mean = {avg_global_f1s_test:.4f}, 'f'Std = {std_global_f1s_test:.4f}')
    print("************************************************************************************")
    # print(f'Weighted Accuracy: Mean = {weighted_avg_global_acc_test:.4f}, Std = {weighted_std_global_acc_test:.4f}')
    # print(f'Weighted Precision: Mean = {weighted_avg_global_pre_test:.4f}, Std = {weighted_std_global_pre_test:.4f}')
    # print(f'Weighted Recall: Mean = {weighted_avg_global_rec_test:.4f}, Std = {weighted_std_global_rec_test:.4f}')
    # print(f'Weighted F1-Score: Mean = {weighted_avg_global_f1s_test:.4f}, Std = {weighted_std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')



def train_fedavgm(clients_glob, clients_glob_test, local_nodes_glob, random_state, seeds, epochs, comms_round):

    print(f"\nTraining the {agg_method_used} aggregation algorithm using Flower...")    
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(clients_dict=clients_glob)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_test, list_y_test = from_FedArtML_to_Flower_format(clients_dict=clients_glob_test)
    
    # Define function to pass to each local node (client)
    def client_fn(context: Context) -> fl.client.Client:
        # Define model
        if dataset_used == "acs_income" or dataset_used == "dutch":
            model = LRModel(len(covariates), len(np.unique(y_all)))
        elif dataset_used == "sent140":
            model = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
        elif dataset_used == "celeba":
            model = DNN_celeba(input_shape=X_test[0].shape, num_classes=len(np.unique(y_all)))
    
        # Set optimizer
        optimizer = Adam(learning_rate=0.001)
    
        # Compile model
        model.compile(optimizer=optimizer, loss=loss_inic, metrics=metrics)
    
        # Load data partition of each client ID (cid)
        x_train_cid = np.array(list_x_train[int(context.node_config['partition-id'])], dtype=float)
        y_train_cid = np.array(list_y_train[int(context.node_config['partition-id'])], dtype=float)

        # Load data partition of each client ID (cid)
        x_test_cid = np.array(list_x_test[int(context.node_config['partition-id'])], dtype=float)
        y_test_cid = np.array(list_y_test[int(context.node_config['partition-id'])], dtype=float)

        # Create and return client
        return FlowerClient(model, x_train_cid, y_train_cid, x_test_cid, y_test_cid, epochs, context.node_config['partition-id'])
        
    # Initialize variables to store the best results
    best_server_momentum_threshold = None
    best_avg_accuracy = -1  # Initialize to a very low value
    best_results = None
    
    # Loop through each PSI threshold
    for server_momentum_threshold in tqdm(server_momentum_thresholds,desc="Fine tuning"):
        print(f"\nEvaluating for PSI threshold: {server_momentum_threshold}")

        # Initialize storage for metrics and training times
        global_acc_tests, global_pre_tests, global_rec_tests, global_f1s_tests = [], [], [], []
        weighted_global_acc_tests, weighted_global_pre_tests, weighted_global_rec_tests, weighted_global_f1s_tests = [], [], [], []        
        training_times = []
        all_commun_metrics_histories = []  # For saving history of all communication rounds

        # Initialize dataframe to store results for this server_momentum_threshold
        threshold_results = pd.DataFrame(columns=[
            'agg_method_used', 'server_momentum_threshold', 'seed', 'training_time', 
            'global_acc_test', 'global_pre_test', 'global_rec_test', 'global_f1s_test',
            'weighted_global_acc_test', 'weighted_global_pre_test', 'weighted_global_rec_test', 'weighted_global_f1s_test'
        ])

        metrics_per_client_per_seed = []
        model_parameters_per_seed = [] 
        
        # Train model for each seed
        for seed in tqdm(seeds,desc=f"Looping on seeds for tuning parameter{server_momentum_threshold}"):
            print(f"Running simulation with seed {seed}")
            np.random.seed(seed)
            tf.random.set_seed(seed)

            # Define model
            if dataset_used == "acs_income" or dataset_used == "dutch":
                my_model = LRModel(len(covariates), len(np.unique(y_all)))
            elif dataset_used == "sent140":
                my_model = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
            elif dataset_used == "celeba":
                my_model = DNN_celeba(input_shape=X_test[0].shape, num_classes=len(np.unique(y_all)))
            
            # Create Federated strategy
            strategy = fl.server.strategy.FedAvgM(
                fraction_fit=0.5,
                fraction_evaluate=0.5,
                evaluate_fn=evaluate_LR_CL,
                initial_parameters=fl.common.ndarrays_to_parameters(my_model.get_weights()),
                server_momentum=server_momentum_threshold
            )
            
            # Start training simulation
            start_time = time.time()
            commun_metrics_history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=len(list_x_train),
                config=fl.server.ServerConfig(num_rounds=comms_round),
                strategy=strategy,
                ray_init_args={"num_cpus": CPUs_to_use},                
            )
            training_time = time.time() - start_time
            
            # Append communication metrics history
            all_commun_metrics_histories.append(commun_metrics_history)

            # Retrieve metrics
            global_acc_test = retrieve_global_metrics(commun_metrics_history, "centralized", "accuracy", False)
            global_pre_test = retrieve_global_metrics(commun_metrics_history, "centralized", "precision", False)
            global_rec_test = retrieve_global_metrics(commun_metrics_history, "centralized", "recall", False)
            global_f1s_test = retrieve_global_metrics(commun_metrics_history, "centralized", "f1score", False)

            # Calculate metrics per client (to do weighted average)
            metrics_weighted = evaluate_LR_FL(global_params_retrieved, list_x_test, list_y_test)
            metrics_per_client = metrics_weighted["client_metrics"]
            weighted_averages = metrics_weighted["weighted_averages"]
            
            # Retrieve metrics
            weighted_global_acc_test = weighted_averages['accuracy']
            weighted_global_pre_test = weighted_averages['precision']
            weighted_global_rec_test = weighted_averages['recall']
            weighted_global_f1s_test = weighted_averages['f1score']
        
            # Store results
            global_acc_tests.append(global_acc_test)
            global_pre_tests.append(global_pre_test)
            global_rec_tests.append(global_rec_test)
            global_f1s_tests.append(global_f1s_test)
            weighted_global_acc_tests.append(weighted_global_acc_test)
            weighted_global_pre_tests.append(weighted_global_pre_test)
            weighted_global_rec_tests.append(weighted_global_rec_test)
            weighted_global_f1s_tests.append(weighted_global_f1s_test)             
            training_times.append(training_time)

            metrics_per_client_per_seed.append(dict(metrics_per_client))
            model_parameters_per_seed.append(global_params_retrieved)
        
            new_row = pd.DataFrame([{
                'agg_method_used': agg_method_used,
                'server_momentum_threshold': server_momentum_threshold,
                'seed': seed,
                'training_time': training_time,
                'global_acc_test': global_acc_test,
                'global_pre_test': global_pre_test,
                'global_rec_test': global_rec_test,
                'global_f1s_test': global_f1s_test,
                'weighted_global_acc_test': weighted_global_acc_test,
                'weighted_global_pre_test': weighted_global_pre_test,
                'weighted_global_rec_test': weighted_global_rec_test,
                'weighted_global_f1s_test': weighted_global_f1s_test                   
            }])
            
            threshold_results = pd.concat([threshold_results, new_row], ignore_index=True)

        # Compute means and standard deviations
        avg_global_acc_test = np.mean(global_acc_tests)
        std_global_acc_test = np.std(global_acc_tests)
        
        avg_global_pre_test = np.mean(global_pre_tests)
        std_global_pre_test = np.std(global_pre_tests)
        
        avg_global_rec_test = np.mean(global_rec_tests)
        std_global_rec_test = np.std(global_rec_tests)
        
        avg_global_f1s_test = np.mean(global_f1s_tests)
        std_global_f1s_test = np.std(global_f1s_tests)
        
        avg_training_time = np.mean(training_times)
        std_training_time = np.std(training_times)

        # Compute means and standard deviations for weight values
        weighted_avg_global_acc_test = np.mean(weighted_global_acc_tests)
        weighted_std_global_acc_test = np.std(weighted_global_acc_tests)
        
        weighted_avg_global_pre_test = np.mean(weighted_global_pre_tests)
        weighted_std_global_pre_test = np.std(weighted_global_pre_tests)
        
        weighted_avg_global_rec_test = np.mean(weighted_global_rec_tests)
        weighted_std_global_rec_test = np.std(weighted_global_rec_tests)
        
        weighted_avg_global_f1s_test = np.mean(weighted_global_f1s_tests)
        weighted_std_global_f1s_test = np.std(weighted_global_f1s_tests)
        
        # Average metrics history across seeds
        avg_commun_metrics_history = {
            "metrics_centralized": {
                metric: [(round_idx, np.mean([h.metrics_centralized[metric][round_idx][1]
                                                 for h in all_commun_metrics_histories]))
                         for round_idx in range(comms_round + 1)]
                for metric in ["accuracy", "precision", "recall", "f1score"]
            }
        }

        

        if avg_global_acc_test > best_avg_accuracy:
            best_avg_accuracy = avg_global_acc_test
            best_server_momentum_threshold = server_momentum_threshold
            
            # Save all best results
            best_results = {
                'server_momentum_threshold': server_momentum_threshold,
                'metrics': {
                    'accuracy': {'mean': avg_global_acc_test, 'std': std_global_acc_test},
                    'precision': {'mean': avg_global_pre_test, 'std': std_global_pre_test},
                    'recall': {'mean': avg_global_rec_test, 'std': std_global_rec_test},
                    'f1score': {'mean': avg_global_f1s_test, 'std': std_global_f1s_test},
                    'weighted_accuracy': {'mean': weighted_avg_global_acc_test, 'std': weighted_std_global_acc_test},
                    'weighted_precision': {'mean': weighted_avg_global_pre_test, 'std': weighted_std_global_pre_test},
                    'weighted_recall': {'mean': weighted_avg_global_rec_test, 'std': weighted_std_global_rec_test},
                    'weighted_f1score': {'mean': weighted_avg_global_f1s_test, 'std': weighted_std_global_f1s_test},
                    'metrics_per_client_per_seed': metrics_per_client_per_seed,
                    'model_parameters_per_seed': model_parameters_per_seed,
                },
                'training_time': {'mean': avg_training_time, 'std': std_training_time},
                'avg_commun_metrics_history': avg_commun_metrics_history

            }

    best_server_momentum_threshold = best_results['server_momentum_threshold']
            
    # Compute means and standard deviations
    avg_global_acc_test = best_results["metrics"]["accuracy"]["mean"]
    std_global_acc_test = best_results["metrics"]["accuracy"]["std"]
    
    avg_global_pre_test = best_results["metrics"]["precision"]["mean"]
    std_global_pre_test = best_results["metrics"]["precision"]["std"]
    
    avg_global_rec_test = best_results["metrics"]["recall"]["mean"]
    std_global_rec_test = best_results["metrics"]["recall"]["std"]
    
    avg_global_f1s_test = best_results["metrics"]["f1score"]["mean"]
    std_global_f1s_test = best_results["metrics"]["f1score"]["std"]
    
    avg_training_time = best_results["training_time"]["mean"]
    std_training_time = best_results["training_time"]["std"]
    
    weighted_avg_global_acc_test = best_results["metrics"]["weighted_accuracy"]["mean"]
    weighted_std_global_acc_test = best_results["metrics"]["weighted_accuracy"]["std"]
    
    weighted_avg_global_pre_test = best_results["metrics"]["weighted_precision"]["mean"]
    weighted_std_global_pre_test = best_results["metrics"]["weighted_precision"]["std"]
    
    weighted_avg_global_rec_test = best_results["metrics"]["weighted_recall"]["mean"]
    weighted_std_global_rec_test = best_results["metrics"]["weighted_recall"]["std"]
    
    weighted_avg_global_f1s_test = best_results["metrics"]["weighted_f1score"]["mean"]
    weighted_std_global_f1s_test = best_results["metrics"]["weighted_f1score"]["std"]
    
    avg_commun_metrics_history = best_results["avg_commun_metrics_history"]
    
    # Print best results
    print("\n\nBEST RESULTS:")
    print(f'Best Server Momentum Threshold: {best_server_momentum_threshold}')
    print(f'Global Accuracy: Mean = {avg_global_acc_test:.4f}, 'f'Std = {std_global_acc_test:.4f}')
    print(f'Global Precision: Mean = {avg_global_pre_test:.4f}, 'f'Std = {std_global_pre_test:.4f}')
    print(f'Global Recall: Mean = {avg_global_rec_test:.4f}, 'f'Std = {std_global_rec_test:.4f}')
    print(f'Global F1-Score: Mean = {avg_global_f1s_test:.4f}, 'f'Std = {std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')
    print("************************************************************************************")
    # print(f'Weighted Accuracy: Mean = {weighted_avg_global_acc_test:.4f}, Std = {weighted_std_global_acc_test:.4f}')
    # print(f'Weighted Precision: Mean = {weighted_avg_global_pre_test:.4f}, Std = {weighted_std_global_pre_test:.4f}')
    # print(f'Weighted Recall: Mean = {weighted_avg_global_rec_test:.4f}, Std = {weighted_std_global_rec_test:.4f}')
    # print(f'Weighted F1-Score: Mean = {weighted_avg_global_f1s_test:.4f}, Std = {weighted_std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')




def train_fedadagrad(clients_glob, clients_glob_test, local_nodes_glob, random_state, seeds, epochs, comms_round):

    print(f"\nTraining the {agg_method_used} aggregation algorithm using Flower...")    
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(clients_dict=clients_glob)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_test, list_y_test = from_FedArtML_to_Flower_format(clients_dict=clients_glob_test)
    
    # Define function to pass to each local node (client)
    def client_fn(context: Context) -> fl.client.Client:
        # Define model
        if dataset_used == "acs_income" or dataset_used == "dutch":
            model = LRModel(len(covariates), len(np.unique(y_all)))
        elif dataset_used == "sent140":
            model = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
        elif dataset_used == "celeba":
            model = DNN_celeba(input_shape=X_test[0].shape, num_classes=len(np.unique(y_all)))
    
        # Set optimizer
        optimizer = Adam(learning_rate=0.001)
    
        # Compile model
        model.compile(optimizer=optimizer, loss=loss_inic, metrics=metrics)
    
        # Load data partition of each client ID (cid)
        x_train_cid = np.array(list_x_train[int(context.node_config['partition-id'])], dtype=float)
        y_train_cid = np.array(list_y_train[int(context.node_config['partition-id'])], dtype=float)

        # Load data partition of each client ID (cid)
        x_test_cid = np.array(list_x_test[int(context.node_config['partition-id'])], dtype=float)
        y_test_cid = np.array(list_y_test[int(context.node_config['partition-id'])], dtype=float)

        # Create and return client
        return FlowerClient(model, x_train_cid, y_train_cid, x_test_cid, y_test_cid, epochs, context.node_config['partition-id'])
        
    # Initialize variables to store the best results
    best_tau_threshold = None
    best_avg_accuracy = -1  # Initialize to a very low value
    best_results = None

    i = 0
    # Loop through each PSI threshold
    for tau_threshold in tqdm(tau_thresholds,desc="Fine tuning"):
        print(f"\nEvaluating for PSI threshold: {tau_threshold}")

        # Initialize storage for metrics and training times
        global_acc_tests, global_pre_tests, global_rec_tests, global_f1s_tests = [], [], [], []
        weighted_global_acc_tests, weighted_global_pre_tests, weighted_global_rec_tests, weighted_global_f1s_tests = [], [], [], []        
        training_times = []
        all_commun_metrics_histories = []  # For saving history of all communication rounds

        # Initialize dataframe to store results for this tau_threshold
        threshold_results = pd.DataFrame(columns=[
            'agg_method_used', 'tau_threshold', 'seed', 'training_time', 
            'global_acc_test', 'global_pre_test', 'global_rec_test', 'global_f1s_test',
            'weighted_global_acc_test', 'weighted_global_pre_test', 'weighted_global_rec_test', 'weighted_global_f1s_test'
        ])

        metrics_per_client_per_seed = []
        model_parameters_per_seed = [] 
        
        # Train model for each seed
        for seed in tqdm(seeds,desc=f"Looping on seeds for tuning parameter{tau_threshold}"):
            print(f"Running simulation with seed {seed}")
            np.random.seed(seed)
            tf.random.set_seed(seed)

            # Define model
            if dataset_used == "acs_income" or dataset_used == "dutch":
                my_model = LRModel(len(covariates), len(np.unique(y_all)))
            elif dataset_used == "sent140":
                my_model = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
            elif dataset_used == "celeba":
                my_model = DNN_celeba(input_shape=X_test[0].shape, num_classes=len(np.unique(y_all)))
            
            # Create Federated strategy
            strategy = fl.server.strategy.FedAdagrad(
                fraction_fit=0.5,
                fraction_evaluate=0.5,
                evaluate_fn=evaluate_LR_CL,
            initial_parameters=fl.common.ndarrays_to_parameters(my_model.get_weights()),
            eta=eta_range[i],
            eta_l=eta_l_range[i],
            tau=tau_threshold
            )
            
            # Start training simulation
            start_time = time.time()
            commun_metrics_history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=len(list_x_train),
                config=fl.server.ServerConfig(num_rounds=comms_round),
                strategy=strategy,
                ray_init_args={"num_cpus": CPUs_to_use},                
            )
            training_time = time.time() - start_time
            
            # Append communication metrics history
            all_commun_metrics_histories.append(commun_metrics_history)

            # Retrieve metrics
            global_acc_test = retrieve_global_metrics(commun_metrics_history, "centralized", "accuracy", False)
            global_pre_test = retrieve_global_metrics(commun_metrics_history, "centralized", "precision", False)
            global_rec_test = retrieve_global_metrics(commun_metrics_history, "centralized", "recall", False)
            global_f1s_test = retrieve_global_metrics(commun_metrics_history, "centralized", "f1score", False)

            # Calculate metrics per client (to do weighted average)
            metrics_weighted = evaluate_LR_FL(global_params_retrieved, list_x_test, list_y_test)
            metrics_per_client = metrics_weighted["client_metrics"]
            weighted_averages = metrics_weighted["weighted_averages"]
            
            # Retrieve metrics
            weighted_global_acc_test = weighted_averages['accuracy']
            weighted_global_pre_test = weighted_averages['precision']
            weighted_global_rec_test = weighted_averages['recall']
            weighted_global_f1s_test = weighted_averages['f1score']
        
            # Store results
            global_acc_tests.append(global_acc_test)
            global_pre_tests.append(global_pre_test)
            global_rec_tests.append(global_rec_test)
            global_f1s_tests.append(global_f1s_test)
            weighted_global_acc_tests.append(weighted_global_acc_test)
            weighted_global_pre_tests.append(weighted_global_pre_test)
            weighted_global_rec_tests.append(weighted_global_rec_test)
            weighted_global_f1s_tests.append(weighted_global_f1s_test)             
            training_times.append(training_time)

            metrics_per_client_per_seed.append(dict(metrics_per_client))
            model_parameters_per_seed.append(global_params_retrieved)
        
            new_row = pd.DataFrame([{
                'agg_method_used': agg_method_used,
                'tau_threshold': tau_threshold,
                'seed': seed,
                'training_time': training_time,
                'global_acc_test': global_acc_test,
                'global_pre_test': global_pre_test,
                'global_rec_test': global_rec_test,
                'global_f1s_test': global_f1s_test,
                'weighted_global_acc_test': weighted_global_acc_test,
                'weighted_global_pre_test': weighted_global_pre_test,
                'weighted_global_rec_test': weighted_global_rec_test,
                'weighted_global_f1s_test': weighted_global_f1s_test                   
            }])
            
            threshold_results = pd.concat([threshold_results, new_row], ignore_index=True)


        # Compute means and standard deviations
        avg_global_acc_test = np.mean(global_acc_tests)
        std_global_acc_test = np.std(global_acc_tests)
        
        avg_global_pre_test = np.mean(global_pre_tests)
        std_global_pre_test = np.std(global_pre_tests)
        
        avg_global_rec_test = np.mean(global_rec_tests)
        std_global_rec_test = np.std(global_rec_tests)
        
        avg_global_f1s_test = np.mean(global_f1s_tests)
        std_global_f1s_test = np.std(global_f1s_tests)
        
        avg_training_time = np.mean(training_times)
        std_training_time = np.std(training_times)

        # Compute means and standard deviations for weight values
        weighted_avg_global_acc_test = np.mean(weighted_global_acc_tests)
        weighted_std_global_acc_test = np.std(weighted_global_acc_tests)
        
        weighted_avg_global_pre_test = np.mean(weighted_global_pre_tests)
        weighted_std_global_pre_test = np.std(weighted_global_pre_tests)
        
        weighted_avg_global_rec_test = np.mean(weighted_global_rec_tests)
        weighted_std_global_rec_test = np.std(weighted_global_rec_tests)
        
        weighted_avg_global_f1s_test = np.mean(weighted_global_f1s_tests)
        weighted_std_global_f1s_test = np.std(weighted_global_f1s_tests)

        # Average metrics history across seeds
        avg_commun_metrics_history = {
            "metrics_centralized": {
                metric: [(round_idx, np.mean([h.metrics_centralized[metric][round_idx][1]
                                                 for h in all_commun_metrics_histories]))
                         for round_idx in range(comms_round + 1)]
                for metric in ["accuracy", "precision", "recall", "f1score"]
            }
        }

        

        if avg_global_acc_test > best_avg_accuracy:
            best_avg_accuracy = avg_global_acc_test
            best_tau_threshold = tau_threshold
            
            # Save all best results
            best_results = {
                'tau_threshold': tau_threshold,
                'metrics': {
                    'accuracy': {'mean': avg_global_acc_test, 'std': std_global_acc_test},
                    'precision': {'mean': avg_global_pre_test, 'std': std_global_pre_test},
                    'recall': {'mean': avg_global_rec_test, 'std': std_global_rec_test},
                    'f1score': {'mean': avg_global_f1s_test, 'std': std_global_f1s_test},
                    'weighted_accuracy': {'mean': weighted_avg_global_acc_test, 'std': weighted_std_global_acc_test},
                    'weighted_precision': {'mean': weighted_avg_global_pre_test, 'std': weighted_std_global_pre_test},
                    'weighted_recall': {'mean': weighted_avg_global_rec_test, 'std': weighted_std_global_rec_test},
                    'weighted_f1score': {'mean': weighted_avg_global_f1s_test, 'std': weighted_std_global_f1s_test},
                    'metrics_per_client_per_seed': metrics_per_client_per_seed,
                    'model_parameters_per_seed': model_parameters_per_seed,
                },
                'training_time': {'mean': avg_training_time, 'std': std_training_time},
                'avg_commun_metrics_history': avg_commun_metrics_history

            }
        i+=1
    
    best_tau_threshold = best_results['tau_threshold']
            
    # Compute means and standard deviations
    avg_global_acc_test = best_results["metrics"]["accuracy"]["mean"]
    std_global_acc_test = best_results["metrics"]["accuracy"]["std"]
    
    avg_global_pre_test = best_results["metrics"]["precision"]["mean"]
    std_global_pre_test = best_results["metrics"]["precision"]["std"]
    
    avg_global_rec_test = best_results["metrics"]["recall"]["mean"]
    std_global_rec_test = best_results["metrics"]["recall"]["std"]
    
    avg_global_f1s_test = best_results["metrics"]["f1score"]["mean"]
    std_global_f1s_test = best_results["metrics"]["f1score"]["std"]
    
    avg_training_time = best_results["training_time"]["mean"]
    std_training_time = best_results["training_time"]["std"]
    
    weighted_avg_global_acc_test = best_results["metrics"]["weighted_accuracy"]["mean"]
    weighted_std_global_acc_test = best_results["metrics"]["weighted_accuracy"]["std"]
    
    weighted_avg_global_pre_test = best_results["metrics"]["weighted_precision"]["mean"]
    weighted_std_global_pre_test = best_results["metrics"]["weighted_precision"]["std"]
    
    weighted_avg_global_rec_test = best_results["metrics"]["weighted_recall"]["mean"]
    weighted_std_global_rec_test = best_results["metrics"]["weighted_recall"]["std"]
    
    weighted_avg_global_f1s_test = best_results["metrics"]["weighted_f1score"]["mean"]
    weighted_std_global_f1s_test = best_results["metrics"]["weighted_f1score"]["std"]
    
    avg_commun_metrics_history = best_results["avg_commun_metrics_history"]
    
    # Print best results
    print("\n\nBEST RESULTS:")
    print(f'Best Tau Threshold: {best_tau_threshold}')
    print(f'Global Accuracy: Mean = {avg_global_acc_test:.4f}, 'f'Std = {std_global_acc_test:.4f}')
    print(f'Global Precision: Mean = {avg_global_pre_test:.4f}, 'f'Std = {std_global_pre_test:.4f}')
    print(f'Global Recall: Mean = {avg_global_rec_test:.4f}, 'f'Std = {std_global_rec_test:.4f}')
    print(f'Global F1-Score: Mean = {avg_global_f1s_test:.4f}, 'f'Std = {std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')
    print("************************************************************************************")
    # print(f'Weighted Accuracy: Mean = {weighted_avg_global_acc_test:.4f}, Std = {weighted_std_global_acc_test:.4f}')
    # print(f'Weighted Precision: Mean = {weighted_avg_global_pre_test:.4f}, Std = {weighted_std_global_pre_test:.4f}')
    # print(f'Weighted Recall: Mean = {weighted_avg_global_rec_test:.4f}, Std = {weighted_std_global_rec_test:.4f}')
    # print(f'Weighted F1-Score: Mean = {weighted_avg_global_f1s_test:.4f}, Std = {weighted_std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')





def train_fedyogi(clients_glob, clients_glob_test, local_nodes_glob, random_state, seeds, epochs, comms_round):

    print(f"\nTraining the {agg_method_used} aggregation algorithm using Flower...")    
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(clients_dict=clients_glob)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_test, list_y_test = from_FedArtML_to_Flower_format(clients_dict=clients_glob_test)
    
    # Define function to pass to each local node (client)
    def client_fn(context: Context) -> fl.client.Client:
        # Define model
        if dataset_used == "acs_income" or dataset_used == "dutch":
            model = LRModel(len(covariates), len(np.unique(y_all)))
        elif dataset_used == "sent140":
            model = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
        elif dataset_used == "celeba":
            model = DNN_celeba(input_shape=X_test[0].shape, num_classes=len(np.unique(y_all)))
    
        # Set optimizer
        optimizer = Adam(learning_rate=0.001)
    
        # Compile model
        model.compile(optimizer=optimizer, loss=loss_inic, metrics=metrics)
    
        # Load data partition of each client ID (cid)
        x_train_cid = np.array(list_x_train[int(context.node_config['partition-id'])], dtype=float)
        y_train_cid = np.array(list_y_train[int(context.node_config['partition-id'])], dtype=float)

        # Load data partition of each client ID (cid)
        x_test_cid = np.array(list_x_test[int(context.node_config['partition-id'])], dtype=float)
        y_test_cid = np.array(list_y_test[int(context.node_config['partition-id'])], dtype=float)

        # Create and return client
        return FlowerClient(model, x_train_cid, y_train_cid, x_test_cid, y_test_cid, epochs, context.node_config['partition-id'])
        
    # Initialize variables to store the best results
    best_tau_threshold = None
    best_avg_accuracy = -1  # Initialize to a very low value
    best_results = None

    i = 0
    # Loop through each PSI threshold
    for tau_threshold in tqdm(tau_thresholds,desc="Fine tuning"):
        print(f"\nEvaluating for PSI threshold: {tau_threshold}")

        # Initialize storage for metrics and training times
        global_acc_tests, global_pre_tests, global_rec_tests, global_f1s_tests = [], [], [], []
        weighted_global_acc_tests, weighted_global_pre_tests, weighted_global_rec_tests, weighted_global_f1s_tests = [], [], [], []        
        training_times = []
        all_commun_metrics_histories = []  # For saving history of all communication rounds

        # Initialize dataframe to store results for this tau_threshold
        threshold_results = pd.DataFrame(columns=[
            'agg_method_used', 'tau_threshold', 'seed', 'training_time', 
            'global_acc_test', 'global_pre_test', 'global_rec_test', 'global_f1s_test',
            'weighted_global_acc_test', 'weighted_global_pre_test', 'weighted_global_rec_test', 'weighted_global_f1s_test'
        ])

        metrics_per_client_per_seed = []
        model_parameters_per_seed = [] 
        
        # Train model for each seed
        for seed in tqdm(seeds,desc=f"Looping on seeds for tuning parameter{tau_threshold}"):
            print(f"Running simulation with seed {seed}")
            np.random.seed(seed)
            tf.random.set_seed(seed)

            # Define model
            if dataset_used == "acs_income" or dataset_used == "dutch":
                my_model = LRModel(len(covariates), len(np.unique(y_all)))
            elif dataset_used == "sent140":
                my_model = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
            elif dataset_used == "celeba":
                my_model = DNN_celeba(input_shape=X_test[0].shape, num_classes=len(np.unique(y_all)))

            
            # Create Federated strategy
            strategy = fl.server.strategy.FedYogi(
                fraction_fit=0.5,
                # min_available_clients=int(0.5 * local_nodes_glob),
                # min_fit_clients=min(int(0.5 * local_nodes_glob),CPUs_to_use),
                
                fraction_evaluate=0.5,
                evaluate_fn=evaluate_LR_CL,
            initial_parameters=fl.common.ndarrays_to_parameters(my_model.get_weights()),
            eta=eta_range[i],
            eta_l=eta_l_range[i],
            beta_1=beta_1_range[i],
            beta_2=beta_2_range[i],
            tau=tau_threshold
            )
            
            # Start training simulation
            start_time = time.time()
            commun_metrics_history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=len(list_x_train),
                config=fl.server.ServerConfig(num_rounds=comms_round),
                strategy=strategy,
                ray_init_args={"num_cpus": CPUs_to_use},                
            )
            training_time = time.time() - start_time
            
            # Append communication metrics history
            all_commun_metrics_histories.append(commun_metrics_history)

            # Retrieve metrics
            global_acc_test = retrieve_global_metrics(commun_metrics_history, "centralized", "accuracy", False)
            global_pre_test = retrieve_global_metrics(commun_metrics_history, "centralized", "precision", False)
            global_rec_test = retrieve_global_metrics(commun_metrics_history, "centralized", "recall", False)
            global_f1s_test = retrieve_global_metrics(commun_metrics_history, "centralized", "f1score", False)

            # Calculate metrics per client (to do weighted average)
            metrics_weighted = evaluate_LR_FL(global_params_retrieved, list_x_test, list_y_test)
            metrics_per_client = metrics_weighted["client_metrics"]
            weighted_averages = metrics_weighted["weighted_averages"]
            
            # Retrieve metrics
            weighted_global_acc_test = weighted_averages['accuracy']
            weighted_global_pre_test = weighted_averages['precision']
            weighted_global_rec_test = weighted_averages['recall']
            weighted_global_f1s_test = weighted_averages['f1score']
        
            # Store results
            global_acc_tests.append(global_acc_test)
            global_pre_tests.append(global_pre_test)
            global_rec_tests.append(global_rec_test)
            global_f1s_tests.append(global_f1s_test)
            weighted_global_acc_tests.append(weighted_global_acc_test)
            weighted_global_pre_tests.append(weighted_global_pre_test)
            weighted_global_rec_tests.append(weighted_global_rec_test)
            weighted_global_f1s_tests.append(weighted_global_f1s_test)             
            training_times.append(training_time)

            metrics_per_client_per_seed.append(dict(metrics_per_client))
            model_parameters_per_seed.append(global_params_retrieved)
        
            new_row = pd.DataFrame([{
                'agg_method_used': agg_method_used,
                'tau_threshold': tau_threshold,
                'seed': seed,
                'training_time': training_time,
                'global_acc_test': global_acc_test,
                'global_pre_test': global_pre_test,
                'global_rec_test': global_rec_test,
                'global_f1s_test': global_f1s_test,
                'weighted_global_acc_test': weighted_global_acc_test,
                'weighted_global_pre_test': weighted_global_pre_test,
                'weighted_global_rec_test': weighted_global_rec_test,
                'weighted_global_f1s_test': weighted_global_f1s_test                   
            }])
            
            threshold_results = pd.concat([threshold_results, new_row], ignore_index=True)

        # Compute means and standard deviations
        avg_global_acc_test = np.mean(global_acc_tests)
        std_global_acc_test = np.std(global_acc_tests)
        
        avg_global_pre_test = np.mean(global_pre_tests)
        std_global_pre_test = np.std(global_pre_tests)
        
        avg_global_rec_test = np.mean(global_rec_tests)
        std_global_rec_test = np.std(global_rec_tests)
        
        avg_global_f1s_test = np.mean(global_f1s_tests)
        std_global_f1s_test = np.std(global_f1s_tests)
        
        avg_training_time = np.mean(training_times)
        std_training_time = np.std(training_times)

        # Compute means and standard deviations for weight values
        weighted_avg_global_acc_test = np.mean(weighted_global_acc_tests)
        weighted_std_global_acc_test = np.std(weighted_global_acc_tests)
        
        weighted_avg_global_pre_test = np.mean(weighted_global_pre_tests)
        weighted_std_global_pre_test = np.std(weighted_global_pre_tests)
        
        weighted_avg_global_rec_test = np.mean(weighted_global_rec_tests)
        weighted_std_global_rec_test = np.std(weighted_global_rec_tests)
        
        weighted_avg_global_f1s_test = np.mean(weighted_global_f1s_tests)
        weighted_std_global_f1s_test = np.std(weighted_global_f1s_tests)
        
        # Average metrics history across seeds
        avg_commun_metrics_history = {
            "metrics_centralized": {
                metric: [(round_idx, np.mean([h.metrics_centralized[metric][round_idx][1]
                                                 for h in all_commun_metrics_histories]))
                         for round_idx in range(comms_round + 1)]
                for metric in ["accuracy", "precision", "recall", "f1score"]
            }
        }

        

        if avg_global_acc_test > best_avg_accuracy:
            best_avg_accuracy = avg_global_acc_test
            best_tau_threshold = tau_threshold
            
            # Save all best results
            best_results = {
                'tau_threshold': tau_threshold,
                'metrics': {
                    'accuracy': {'mean': avg_global_acc_test, 'std': std_global_acc_test},
                    'precision': {'mean': avg_global_pre_test, 'std': std_global_pre_test},
                    'recall': {'mean': avg_global_rec_test, 'std': std_global_rec_test},
                    'f1score': {'mean': avg_global_f1s_test, 'std': std_global_f1s_test},
                    'weighted_accuracy': {'mean': weighted_avg_global_acc_test, 'std': weighted_std_global_acc_test},
                    'weighted_precision': {'mean': weighted_avg_global_pre_test, 'std': weighted_std_global_pre_test},
                    'weighted_recall': {'mean': weighted_avg_global_rec_test, 'std': weighted_std_global_rec_test},
                    'weighted_f1score': {'mean': weighted_avg_global_f1s_test, 'std': weighted_std_global_f1s_test},
                    'metrics_per_client_per_seed': metrics_per_client_per_seed,
                    'model_parameters_per_seed': model_parameters_per_seed,
                },
                'training_time': {'mean': avg_training_time, 'std': std_training_time},
                'avg_commun_metrics_history': avg_commun_metrics_history

            }
        i+=1
    
    best_tau_threshold = best_results['tau_threshold']
            
    # Compute means and standard deviations
    avg_global_acc_test = best_results["metrics"]["accuracy"]["mean"]
    std_global_acc_test = best_results["metrics"]["accuracy"]["std"]
    
    avg_global_pre_test = best_results["metrics"]["precision"]["mean"]
    std_global_pre_test = best_results["metrics"]["precision"]["std"]
    
    avg_global_rec_test = best_results["metrics"]["recall"]["mean"]
    std_global_rec_test = best_results["metrics"]["recall"]["std"]
    
    avg_global_f1s_test = best_results["metrics"]["f1score"]["mean"]
    std_global_f1s_test = best_results["metrics"]["f1score"]["std"]
    
    avg_training_time = best_results["training_time"]["mean"]
    std_training_time = best_results["training_time"]["std"]
    
    weighted_avg_global_acc_test = best_results["metrics"]["weighted_accuracy"]["mean"]
    weighted_std_global_acc_test = best_results["metrics"]["weighted_accuracy"]["std"]
    
    weighted_avg_global_pre_test = best_results["metrics"]["weighted_precision"]["mean"]
    weighted_std_global_pre_test = best_results["metrics"]["weighted_precision"]["std"]
    
    weighted_avg_global_rec_test = best_results["metrics"]["weighted_recall"]["mean"]
    weighted_std_global_rec_test = best_results["metrics"]["weighted_recall"]["std"]
    
    weighted_avg_global_f1s_test = best_results["metrics"]["weighted_f1score"]["mean"]
    weighted_std_global_f1s_test = best_results["metrics"]["weighted_f1score"]["std"]
    
    avg_commun_metrics_history = best_results["avg_commun_metrics_history"]
    
    # Print best results
    print("\n\nBEST RESULTS:")
    print(f'Best Tau Threshold: {best_tau_threshold}')
    print(f'Global Accuracy: Mean = {avg_global_acc_test:.4f}, 'f'Std = {std_global_acc_test:.4f}')
    print(f'Global Precision: Mean = {avg_global_pre_test:.4f}, 'f'Std = {std_global_pre_test:.4f}')
    print(f'Global Recall: Mean = {avg_global_rec_test:.4f}, 'f'Std = {std_global_rec_test:.4f}')
    print(f'Global F1-Score: Mean = {avg_global_f1s_test:.4f}, 'f'Std = {std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')
    print("************************************************************************************")
    # print(f'Weighted Accuracy: Mean = {weighted_avg_global_acc_test:.4f}, Std = {weighted_std_global_acc_test:.4f}')
    # print(f'Weighted Precision: Mean = {weighted_avg_global_pre_test:.4f}, Std = {weighted_std_global_pre_test:.4f}')
    # print(f'Weighted Recall: Mean = {weighted_avg_global_rec_test:.4f}, Std = {weighted_std_global_rec_test:.4f}')
    # print(f'Weighted F1-Score: Mean = {weighted_avg_global_f1s_test:.4f}, Std = {weighted_std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')
    

def train_fedadam(clients_glob, clients_glob_test, local_nodes_glob, random_state, seeds, epochs, comms_round):

    print(f"\nTraining the {agg_method_used} aggregation algorithm using Flower...")    
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(clients_dict=clients_glob)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_test, list_y_test = from_FedArtML_to_Flower_format(clients_dict=clients_glob_test)
    
    # Define function to pass to each local node (client)
    def client_fn(context: Context) -> fl.client.Client:
        # Define model
        if dataset_used == "acs_income" or dataset_used == "dutch":
            model = LRModel(len(covariates), len(np.unique(y_all)))
        elif dataset_used == "sent140":
            model = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
        elif dataset_used == "celeba":
            model = DNN_celeba(input_shape=X_test[0].shape, num_classes=len(np.unique(y_all)))
    
        # Set optimizer
        optimizer = Adam(learning_rate=0.001)
    
        # Compile model
        model.compile(optimizer=optimizer, loss=loss_inic, metrics=metrics)
    
        # Load data partition of each client ID (cid)
        x_train_cid = np.array(list_x_train[int(context.node_config['partition-id'])], dtype=float)
        y_train_cid = np.array(list_y_train[int(context.node_config['partition-id'])], dtype=float)

        # Load data partition of each client ID (cid)
        x_test_cid = np.array(list_x_test[int(context.node_config['partition-id'])], dtype=float)
        y_test_cid = np.array(list_y_test[int(context.node_config['partition-id'])], dtype=float)

        # Create and return client
        return FlowerClient(model, x_train_cid, y_train_cid, x_test_cid, y_test_cid, epochs, context.node_config['partition-id'])
        
    # Initialize variables to store the best results
    best_tau_threshold = None
    best_avg_accuracy = -1  # Initialize to a very low value
    best_results = None

    i = 0
    # Loop through each PSI threshold
    for tau_threshold in tqdm(tau_thresholds,desc="Fine tuning"):
        print(f"\nEvaluating for PSI threshold: {tau_threshold}")

        # Initialize storage for metrics and training times
        global_acc_tests, global_pre_tests, global_rec_tests, global_f1s_tests = [], [], [], []
        weighted_global_acc_tests, weighted_global_pre_tests, weighted_global_rec_tests, weighted_global_f1s_tests = [], [], [], []        
        training_times = []
        all_commun_metrics_histories = []  # For saving history of all communication rounds

        # Initialize dataframe to store results for this tau_threshold
        threshold_results = pd.DataFrame(columns=[
            'agg_method_used', 'tau_threshold', 'seed', 'training_time', 
            'global_acc_test', 'global_pre_test', 'global_rec_test', 'global_f1s_test',
            'weighted_global_acc_test', 'weighted_global_pre_test', 'weighted_global_rec_test', 'weighted_global_f1s_test'
        ])

        metrics_per_client_per_seed = []
        model_parameters_per_seed = [] 
        
        # Train model for each seed
        for seed in tqdm(seeds,desc=f"Looping on seeds for tuning parameter{tau_threshold}"):
            print(f"Running simulation with seed {seed}")
            np.random.seed(seed)
            tf.random.set_seed(seed)

            # Define model
            if dataset_used == "acs_income" or dataset_used == "dutch":
                my_model = LRModel(len(covariates), len(np.unique(y_all)))
            elif dataset_used == "sent140":
                my_model = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
            elif dataset_used == "celeba":
                my_model = DNN_celeba(input_shape=X_test[0].shape, num_classes=len(np.unique(y_all)))

            
            # Create Federated strategy
            strategy = fl.server.strategy.FedAdam(
                fraction_fit=0.5,
                # min_available_clients=int(0.5 * local_nodes_glob),
                # min_fit_clients=min(int(0.5 * local_nodes_glob),CPUs_to_use),
                
                fraction_evaluate=0.5,
                evaluate_fn=evaluate_LR_CL,
            initial_parameters=fl.common.ndarrays_to_parameters(my_model.get_weights()),
            eta=eta_range[i],
            eta_l=eta_l_range[i],
            beta_1=beta_1_range[i],
            beta_2=beta_2_range[i],
            tau=tau_threshold
            )
            
            # Start training simulation
            start_time = time.time()
            commun_metrics_history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=len(list_x_train),
                config=fl.server.ServerConfig(num_rounds=comms_round),
                strategy=strategy,
            )
            training_time = time.time() - start_time
            
            # Append communication metrics history
            all_commun_metrics_histories.append(commun_metrics_history)

            # Retrieve metrics
            global_acc_test = retrieve_global_metrics(commun_metrics_history, "centralized", "accuracy", False)
            global_pre_test = retrieve_global_metrics(commun_metrics_history, "centralized", "precision", False)
            global_rec_test = retrieve_global_metrics(commun_metrics_history, "centralized", "recall", False)
            global_f1s_test = retrieve_global_metrics(commun_metrics_history, "centralized", "f1score", False)

            # Calculate metrics per client (to do weighted average)
            metrics_weighted = evaluate_LR_FL(global_params_retrieved, list_x_test, list_y_test)
            metrics_per_client = metrics_weighted["client_metrics"]
            weighted_averages = metrics_weighted["weighted_averages"]
            
            # Retrieve metrics
            weighted_global_acc_test = weighted_averages['accuracy']
            weighted_global_pre_test = weighted_averages['precision']
            weighted_global_rec_test = weighted_averages['recall']
            weighted_global_f1s_test = weighted_averages['f1score']
        
            # Store results
            global_acc_tests.append(global_acc_test)
            global_pre_tests.append(global_pre_test)
            global_rec_tests.append(global_rec_test)
            global_f1s_tests.append(global_f1s_test)
            weighted_global_acc_tests.append(weighted_global_acc_test)
            weighted_global_pre_tests.append(weighted_global_pre_test)
            weighted_global_rec_tests.append(weighted_global_rec_test)
            weighted_global_f1s_tests.append(weighted_global_f1s_test)             
            training_times.append(training_time)

            metrics_per_client_per_seed.append(dict(metrics_per_client))
            model_parameters_per_seed.append(global_params_retrieved)
        
            new_row = pd.DataFrame([{
                'agg_method_used': agg_method_used,
                'tau_threshold': tau_threshold,
                'seed': seed,
                'training_time': training_time,
                'global_acc_test': global_acc_test,
                'global_pre_test': global_pre_test,
                'global_rec_test': global_rec_test,
                'global_f1s_test': global_f1s_test,
                'weighted_global_acc_test': weighted_global_acc_test,
                'weighted_global_pre_test': weighted_global_pre_test,
                'weighted_global_rec_test': weighted_global_rec_test,
                'weighted_global_f1s_test': weighted_global_f1s_test                   
            }])
            
            threshold_results = pd.concat([threshold_results, new_row], ignore_index=True)

        # Compute means and standard deviations
        avg_global_acc_test = np.mean(global_acc_tests)
        std_global_acc_test = np.std(global_acc_tests)
        
        avg_global_pre_test = np.mean(global_pre_tests)
        std_global_pre_test = np.std(global_pre_tests)
        
        avg_global_rec_test = np.mean(global_rec_tests)
        std_global_rec_test = np.std(global_rec_tests)
        
        avg_global_f1s_test = np.mean(global_f1s_tests)
        std_global_f1s_test = np.std(global_f1s_tests)
        
        avg_training_time = np.mean(training_times)
        std_training_time = np.std(training_times)

        # Compute means and standard deviations for weight values
        weighted_avg_global_acc_test = np.mean(weighted_global_acc_tests)
        weighted_std_global_acc_test = np.std(weighted_global_acc_tests)
        
        weighted_avg_global_pre_test = np.mean(weighted_global_pre_tests)
        weighted_std_global_pre_test = np.std(weighted_global_pre_tests)
        
        weighted_avg_global_rec_test = np.mean(weighted_global_rec_tests)
        weighted_std_global_rec_test = np.std(weighted_global_rec_tests)
        
        weighted_avg_global_f1s_test = np.mean(weighted_global_f1s_tests)
        weighted_std_global_f1s_test = np.std(weighted_global_f1s_tests)

        # Average metrics history across seeds
        avg_commun_metrics_history = {
            "metrics_centralized": {
                metric: [(round_idx, np.mean([h.metrics_centralized[metric][round_idx][1]
                                                 for h in all_commun_metrics_histories]))
                         for round_idx in range(comms_round + 1)]
                for metric in ["accuracy", "precision", "recall", "f1score"]
            }
        }

        

        if avg_global_acc_test > best_avg_accuracy:
            best_avg_accuracy = avg_global_acc_test
            best_tau_threshold = tau_threshold
            
            # Save all best results
            best_results = {
                'tau_threshold': tau_threshold,
                'metrics': {
                    'accuracy': {'mean': avg_global_acc_test, 'std': std_global_acc_test},
                    'precision': {'mean': avg_global_pre_test, 'std': std_global_pre_test},
                    'recall': {'mean': avg_global_rec_test, 'std': std_global_rec_test},
                    'f1score': {'mean': avg_global_f1s_test, 'std': std_global_f1s_test},
                    'weighted_accuracy': {'mean': weighted_avg_global_acc_test, 'std': weighted_std_global_acc_test},
                    'weighted_precision': {'mean': weighted_avg_global_pre_test, 'std': weighted_std_global_pre_test},
                    'weighted_recall': {'mean': weighted_avg_global_rec_test, 'std': weighted_std_global_rec_test},
                    'weighted_f1score': {'mean': weighted_avg_global_f1s_test, 'std': weighted_std_global_f1s_test},
                    'metrics_per_client_per_seed': metrics_per_client_per_seed,
                    'model_parameters_per_seed': model_parameters_per_seed,
                },
                'training_time': {'mean': avg_training_time, 'std': std_training_time},
                'avg_commun_metrics_history': avg_commun_metrics_history

            }
        i+=1
    
    best_tau_threshold = best_results['tau_threshold']
            
    # Compute means and standard deviations
    avg_global_acc_test = best_results["metrics"]["accuracy"]["mean"]
    std_global_acc_test = best_results["metrics"]["accuracy"]["std"]
    
    avg_global_pre_test = best_results["metrics"]["precision"]["mean"]
    std_global_pre_test = best_results["metrics"]["precision"]["std"]
    
    avg_global_rec_test = best_results["metrics"]["recall"]["mean"]
    std_global_rec_test = best_results["metrics"]["recall"]["std"]
    
    avg_global_f1s_test = best_results["metrics"]["f1score"]["mean"]
    std_global_f1s_test = best_results["metrics"]["f1score"]["std"]
    
    avg_training_time = best_results["training_time"]["mean"]
    std_training_time = best_results["training_time"]["std"]
    
    weighted_avg_global_acc_test = best_results["metrics"]["weighted_accuracy"]["mean"]
    weighted_std_global_acc_test = best_results["metrics"]["weighted_accuracy"]["std"]
    
    weighted_avg_global_pre_test = best_results["metrics"]["weighted_precision"]["mean"]
    weighted_std_global_pre_test = best_results["metrics"]["weighted_precision"]["std"]
    
    weighted_avg_global_rec_test = best_results["metrics"]["weighted_recall"]["mean"]
    weighted_std_global_rec_test = best_results["metrics"]["weighted_recall"]["std"]
    
    weighted_avg_global_f1s_test = best_results["metrics"]["weighted_f1score"]["mean"]
    weighted_std_global_f1s_test = best_results["metrics"]["weighted_f1score"]["std"]
    
    avg_commun_metrics_history = best_results["avg_commun_metrics_history"]
    
    # Print best results
    print("\n\nBEST RESULTS:")
    print(f'Best Tau Threshold: {best_tau_threshold}')
    print(f'Global Accuracy: Mean = {avg_global_acc_test:.4f}, 'f'Std = {std_global_acc_test:.4f}')
    print(f'Global Precision: Mean = {avg_global_pre_test:.4f}, 'f'Std = {std_global_pre_test:.4f}')
    print(f'Global Recall: Mean = {avg_global_rec_test:.4f}, 'f'Std = {std_global_rec_test:.4f}')
    print(f'Global F1-Score: Mean = {avg_global_f1s_test:.4f}, 'f'Std = {std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')
    print("************************************************************************************")
    # print(f'Weighted Accuracy: Mean = {weighted_avg_global_acc_test:.4f}, Std = {weighted_std_global_acc_test:.4f}')
    # print(f'Weighted Precision: Mean = {weighted_avg_global_pre_test:.4f}, Std = {weighted_std_global_pre_test:.4f}')
    # print(f'Weighted Recall: Mean = {weighted_avg_global_rec_test:.4f}, Std = {weighted_std_global_rec_test:.4f}')
    # print(f'Weighted F1-Score: Mean = {weighted_avg_global_f1s_test:.4f}, Std = {weighted_std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')




def train_poc(clients_glob, clients_glob_test, local_nodes_glob, random_state, seeds, epochs, comms_round):

    K = local_nodes_glob
    global batch_size, y_test_tensor,args
    batch_size = 64
    lr = learn_rate
    min_comm_round = 0
    comm_round = comms_round
    d = d_poc                                                     
    w_list = get_m_list(d)
    m_list = [1]
    
    # configuration
    parser = argparse.ArgumentParser(description="FinalModel")
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--d", type=int, default=d)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--comm_round", type=int, default=comm_round)
    parser.add_argument("--min_comm_round", type=int, default=min_comm_round)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    args = parser.parse_args(args=[])

    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(clients_dict=clients_glob)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_test, list_y_test = from_FedArtML_to_Flower_format(clients_dict=clients_glob_test)
    
    if dataset_used == "celeba":
        # Convert the numpy arrays to PyTorch tensors
        x_test_tensor = torch.tensor(np.transpose(X_test, (0, 3, 1,2)), dtype=torch.float32)
        y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
        
        print("X Test shape:", np.transpose(X_test, (0, 3, 1,2)).shape)
        print("Y Test shape:", Y_test.shape)
        
        # Create a TensorDataset from the test data
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)  # unsqueeze to add a channel dimension
        
        # Create a DataLoader from the TensorDataset
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        clients_train = clients_glob
        
        clients_data_sizes = [len(clients_train[f'Client_{i+1}']) for i in range(args.K)]
        prob_distribution = list(clients_data_sizes / np.sum(clients_data_sizes))
        
        client_datasets = []
        client_names = list(clients_train.keys())
        
        for client_id, client in enumerate(client_names):
            # Get data from each client
            each_client_train = np.array(clients_train[client], dtype=object)
        
            # Extract features for each client
            feat = np.transpose(np.array(each_client_train[:, 0].tolist()), (0, 3, 1,2))
            # Extract labels from each client
            y_tra = np.array(each_client_train[:, 1])
        
            client_dataset = ClientDataset(feat, y_tra)
            client_dataset.id = client_id  # Assign the id to the ClientDataset object
        
            client_datasets.append(client_dataset)
    else:
        # Convert the numpy arrays to PyTorch tensors
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
        
        print("X Test shape:", X_test.shape)
        print("Y Test shape:", Y_test.shape)
        
        # Create a TensorDataset from the test data
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)  # unsqueeze to add a channel dimension
        
        # Create a DataLoader from the TensorDataset
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        clients_train = clients_glob
        # hd = round(distances['without_class_completion']['hellinger'], 2)
        
        clients_data_sizes = [len(clients_train[f'Client_{i+1}']) for i in range(args.K)]
        prob_distribution = list(clients_data_sizes / np.sum(clients_data_sizes))
        
        client_datasets = []
        client_names = list(clients_train.keys())
        
        for client_id, client in enumerate(client_names):
            # Get data from each client
            each_client_train = np.array(clients_train[client], dtype=object)
        
            # Extract features for each client
            feat = np.array(each_client_train[:, 0].tolist())
        
            # Extract labels from each client
            y_tra = np.array(each_client_train[:, 1])
        
            client_dataset = ClientDataset(feat, y_tra)
            client_dataset.id = client_id  # Assign the id to the ClientDataset object
        
            client_datasets.append(client_dataset)

    print(f"\nTraining the {agg_method_used} aggregation algorithm using FedLab...")    

    best_avg_accuracy = -1  # Initialize to a very low value
    
    # for w in w_list:
    for w in w_list:
        for m in m_list:
            C = 0.5
            acc_list = []
            loss_list = []
            comm_size_list = []
            tta_list = []
            clients_list = []
            clusters_list = []
    
            # Initialize storage for metrics and training times
            global_acc_tests, global_pre_tests, global_rec_tests, global_f1s_tests = [], [], [], []
            weighted_global_acc_tests, weighted_global_pre_tests, weighted_global_rec_tests, weighted_global_f1s_tests = [], [], [], []                
            training_times = []
    
            metrics_per_client_per_seed = []
            model_parameters_per_seed = [] 
            
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                torch.cuda.manual_seed_all(seed)
    
                # Start training simulation
                start_time = time.time()
    
                # Define model
                if dataset_used == "acs_income" or dataset_used == "dutch":
                    model = LRModel_pytorch(len(covariates), len(np.unique(y_all)))
                elif dataset_used == "sent140":
                    model = DNN_pytorch(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
                elif dataset_used == "celeba":
                    model = DNN_celeba_pytorch(feat.shape[1:], len(np.unique(y_all)))

                handler = Powerofchoice(
                    model = model,
                    prob_distribution = prob_distribution,
                    global_round = args.comm_round,
                    sample_ratio = C
                )
    
                handler.setup_optim(d = args.d)
    
                # trainer = PowerofchoiceSerialClientTrainer(model, args.K, cuda=True)
                trainer = PowerofchoiceSerialClientTrainer(model, args.K, cuda=False)
    
                trainer.setup_optim(args.epochs, args.batch_size)
                trainer.setup_dataset(client_datasets)
    
                pipeline = PowerofchoicePipeline(handler=handler, trainer=trainer, test_loader=test_loader)
    
                pipeline.main()
    
                training_time = time.time() - start_time
                print("")
                print(f"Training for seed: {seed} completed.")
                print("")
    
                # Retrieve metrics
                global_acc_test = pipeline.get_accuracy()[-1]
                global_pre_test = 0
                global_rec_test = 0
                global_f1s_test = 0
    
                global_params_retrieved = pipeline.handler.model_parameters

                # Calculate metrics per client (to do weighted average)
                if dataset_used == "celeba":
                    metrics_weighted = evaluate_DNN_FL_celeba_pytorch(global_params_retrieved, list_x_test, list_y_test)
                else:
                    metrics_weighted = evaluate_LR_FL_pytorch(global_params_retrieved, list_x_test, list_y_test)
                
                metrics_per_client = metrics_weighted["client_metrics"]
                weighted_averages = metrics_weighted["weighted_averages"]
                
                # # Retrieve metrics
                weighted_global_acc_test = weighted_averages['accuracy']
                weighted_global_pre_test = weighted_averages['precision']
                weighted_global_rec_test = weighted_averages['recall']
                weighted_global_f1s_test = weighted_averages['f1score']
    
                
                # Store results
                global_acc_tests.append(global_acc_test)
                global_pre_tests.append(global_pre_test)
                global_rec_tests.append(global_rec_test)
                global_f1s_tests.append(global_f1s_test)
                weighted_global_acc_tests.append(weighted_global_acc_test)
                weighted_global_pre_tests.append(weighted_global_pre_test)
                weighted_global_rec_tests.append(weighted_global_rec_test)
                weighted_global_f1s_tests.append(weighted_global_f1s_test)             
                training_times.append(training_time)
    
                metrics_per_client_per_seed.append(dict(metrics_per_client))
                model_parameters_per_seed.append(global_params_retrieved)
    
                acc_list.append(pipeline.get_accuracy())
                loss_list.append(pipeline.get_loss())
                comm_size_list.append(pipeline.get_communication_size())
                tta_list.append(pipeline.get_time_list())
                clients_list.append(pipeline.get_selected_clients())
                clusters_list.append(pipeline.get_selected_clusters())
    
            # Average metrics history across seeds
            avg_commun_metrics_history = {'metrics_centralized':{'accuracy':list(enumerate(np.mean(acc_list,0)))}}
            
            # Compute means and standard deviations
            avg_global_acc_test = np.mean(global_acc_tests)
            std_global_acc_test = np.std(global_acc_tests)
            
            avg_global_pre_test = np.mean(global_pre_tests)
            std_global_pre_test = np.std(global_pre_tests)
            
            avg_global_rec_test = np.mean(global_rec_tests)
            std_global_rec_test = np.std(global_rec_tests)
            
            avg_global_f1s_test = np.mean(global_f1s_tests)
            std_global_f1s_test = np.std(global_f1s_tests)
            
            avg_training_time = np.mean(training_times)
            std_training_time = np.std(training_times)
                
            # Compute means and standard deviations for weight values
            weighted_avg_global_acc_test = np.mean(weighted_global_acc_tests)
            weighted_std_global_acc_test = np.std(weighted_global_acc_tests)
            
            weighted_avg_global_pre_test = np.mean(weighted_global_pre_tests)
            weighted_std_global_pre_test = np.std(weighted_global_pre_tests)
            
            weighted_avg_global_rec_test = np.mean(weighted_global_rec_tests)
            weighted_std_global_rec_test = np.std(weighted_global_rec_tests)
            
            weighted_avg_global_f1s_test = np.mean(weighted_global_f1s_tests)
            weighted_std_global_f1s_test = np.std(weighted_global_f1s_tests)
                
            
            print("===================")
            print(f"Completed m: {m}")
            print("===================")
    
    
        if avg_global_acc_test > best_avg_accuracy:
            best_avg_accuracy = avg_global_acc_test
            best_w_threshold = w
            
            # Save all best results
            best_results = {
                'w_threshold': w,
                'metrics': {
                    'accuracy': {'mean': avg_global_acc_test, 'std': std_global_acc_test},
                    'precision': {'mean': avg_global_pre_test, 'std': std_global_pre_test},
                    'recall': {'mean': avg_global_rec_test, 'std': std_global_rec_test},
                    'f1score': {'mean': avg_global_f1s_test, 'std': std_global_f1s_test},
                    'weighted_accuracy': {'mean': weighted_avg_global_acc_test, 'std': weighted_std_global_acc_test},
                    'weighted_precision': {'mean': weighted_avg_global_pre_test, 'std': weighted_std_global_pre_test},
                    'weighted_recall': {'mean': weighted_avg_global_rec_test, 'std': weighted_std_global_rec_test},
                    'weighted_f1score': {'mean': weighted_avg_global_f1s_test, 'std': weighted_std_global_f1s_test},
                    'metrics_per_client_per_seed': metrics_per_client_per_seed,
                    'model_parameters_per_seed': model_parameters_per_seed,
                },
                'training_time': {'mean': avg_training_time, 'std': std_training_time},
                'avg_commun_metrics_history': avg_commun_metrics_history
            }
    
        
        print("++++++++++++++++++++")
        print(f"Completed w: {w}")
        print("++++++++++++++++++++")
    
      
    best_w_threshold = best_results['w_threshold']
            
    # Compute means and standard deviations
    avg_global_acc_test = best_results["metrics"]["accuracy"]["mean"]
    std_global_acc_test = best_results["metrics"]["accuracy"]["std"]
    
    avg_global_pre_test = best_results["metrics"]["precision"]["mean"]
    std_global_pre_test = best_results["metrics"]["precision"]["std"]
    
    avg_global_rec_test = best_results["metrics"]["recall"]["mean"]
    std_global_rec_test = best_results["metrics"]["recall"]["std"]
    
    avg_global_f1s_test = best_results["metrics"]["f1score"]["mean"]
    std_global_f1s_test = best_results["metrics"]["f1score"]["std"]
    
    avg_training_time = best_results["training_time"]["mean"]
    std_training_time = best_results["training_time"]["std"]
    
    weighted_avg_global_acc_test = best_results["metrics"]["weighted_accuracy"]["mean"]
    weighted_std_global_acc_test = best_results["metrics"]["weighted_accuracy"]["std"]
    
    weighted_avg_global_pre_test = best_results["metrics"]["weighted_precision"]["mean"]
    weighted_std_global_pre_test = best_results["metrics"]["weighted_precision"]["std"]
    
    weighted_avg_global_rec_test = best_results["metrics"]["weighted_recall"]["mean"]
    weighted_std_global_rec_test = best_results["metrics"]["weighted_recall"]["std"]
    
    weighted_avg_global_f1s_test = best_results["metrics"]["weighted_f1score"]["mean"]
    weighted_std_global_f1s_test = best_results["metrics"]["weighted_f1score"]["std"]
    
    avg_commun_metrics_history = best_results["avg_commun_metrics_history"]
    
    # Print best results
    print("\n\nBEST RESULTS:")
    print(f'Best W Threshold: {best_w_threshold}')
    print(f'Global Accuracy: Mean = {avg_global_acc_test:.4f}, 'f'Std = {std_global_acc_test:.4f}')
    # print(f'Global Precision: Mean = {avg_global_pre_test:.4f}, 'f'Std = {std_global_pre_test:.4f}')
    # print(f'Global Recall: Mean = {avg_global_rec_test:.4f}, 'f'Std = {std_global_rec_test:.4f}')
    # print(f'Global F1-Score: Mean = {avg_global_f1s_test:.4f}, 'f'Std = {std_global_f1s_test:.4f}')
    print("************************************************************************************")
    # print(f'Weighted Accuracy: Mean = {weighted_avg_global_acc_test:.4f}, Std = {weighted_std_global_acc_test:.4f}')
    # print(f'Weighted Precision: Mean = {weighted_avg_global_pre_test:.4f}, Std = {weighted_std_global_pre_test:.4f}')
    # print(f'Weighted Recall: Mean = {weighted_avg_global_rec_test:.4f}, Std = {weighted_std_global_rec_test:.4f}')
    # print(f'Weighted F1-Score: Mean = {weighted_avg_global_f1s_test:.4f}, Std = {weighted_std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')



def train_haccs(clients_glob, clients_glob_test, local_nodes_glob, random_state, seeds, epochs, comms_round):

    K = local_nodes_glob
    global batch_size, y_test_tensor,args,comm_round
    batch_size = 64
    lr = learn_rate
    min_comm_round = 0
    comm_round = comms_round
    d = d_poc                                                     
    w_list = get_m_list(d)
    m_list = [1]
    
    # configuration
    parser = argparse.ArgumentParser(description="FinalModel")
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--d", type=int, default=d)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--comm_round", type=int, default=comm_round)
    parser.add_argument("--min_comm_round", type=int, default=min_comm_round)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    args = parser.parse_args(args=[])

    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(clients_dict=clients_glob)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_test, list_y_test = from_FedArtML_to_Flower_format(clients_dict=clients_glob_test)
    

    if dataset_used == "celeba":
        # Convert the numpy arrays to PyTorch tensors
        x_test_tensor = torch.tensor(np.transpose(X_test, (0, 3, 1,2)), dtype=torch.float32)
        y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
        
        print("X Test shape:", np.transpose(X_test, (0, 3, 1,2)).shape)
        print("Y Test shape:", Y_test.shape)
        
        # Create a TensorDataset from the test data
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)  # unsqueeze to add a channel dimension
        
        # Create a DataLoader from the TensorDataset
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        clients_train = clients_glob
        
        clients_data_sizes = [len(clients_train[f'Client_{i+1}']) for i in range(args.K)]
        prob_distribution = list(clients_data_sizes / np.sum(clients_data_sizes))
        
        client_datasets = []
        client_names = list(clients_train.keys())
        
        for client_id, client in enumerate(client_names):
            # Get data from each client
            each_client_train = np.array(clients_train[client], dtype=object)
        
            # Extract features for each client
            feat = np.transpose(np.array(each_client_train[:, 0].tolist()), (0, 3, 1,2))
            # Extract labels from each client
            y_tra = np.array(each_client_train[:, 1])
        
            client_dataset = ClientDataset(feat, y_tra)
            client_dataset.id = client_id  # Assign the id to the ClientDataset object
        
            client_datasets.append(client_dataset)
    else:
        # Convert the numpy arrays to PyTorch tensors
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
        
        print("X Test shape:", X_test.shape)
        print("Y Test shape:", Y_test.shape)
        
        # Create a TensorDataset from the test data
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)  # unsqueeze to add a channel dimension
        
        # Create a DataLoader from the TensorDataset
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        clients_train = clients_glob
        # hd = round(distances['without_class_completion']['hellinger'], 2)
        
        clients_data_sizes = [len(clients_train[f'Client_{i+1}']) for i in range(args.K)]
        prob_distribution = list(clients_data_sizes / np.sum(clients_data_sizes))
        
        client_datasets = []
        client_names = list(clients_train.keys())
        
        for client_id, client in enumerate(client_names):
            # Get data from each client
            each_client_train = np.array(clients_train[client], dtype=object)
        
            # Extract features for each client
            feat = np.array(each_client_train[:, 0].tolist())
        
            # Extract labels from each client
            y_tra = np.array(each_client_train[:, 1])
        
            client_dataset = ClientDataset(feat, y_tra)
            client_dataset.id = client_id  # Assign the id to the ClientDataset object
        
            client_datasets.append(client_dataset)


    print(f"\nClustering clients for the {agg_method_used} aggregation algorithm...")    
    
    label_distributions = []
    max_label = max([dataset.labels.max() for dataset in client_datasets])  # Find the max label across all datasets
    
    for client_dataset in client_datasets:
        labels = client_dataset.labels
        # Compute histogram with bins from 0 to max_label + 1
        hist, bins = np.histogram(labels, bins=np.arange(max_label + 2) - 0.5)
        hist = hist.astype(float) / hist.sum()
        label_distributions.append(hist)
    
    label_distance_matrix = compute_pairwise_hellinger(label_distributions)
    
    mean_label_hd = np.mean(label_distance_matrix)
    
    print(f"Mean Hellinger Distance across all clients' label distributions: {mean_label_hd}")
    print("")
   
    distance_matrix = label_distance_matrix

    best_dict = get_scores_and_labels_OPTICS(distance_matrix)
    
    warnings.filterwarnings("ignore")
    
    optics = OPTICS(
        metric='precomputed',
        min_samples=best_dict['best_min_samples'],
        max_eps=best_dict['best_max_eps'],
        cluster_method=best_dict['best_cluster_method'],
        xi=best_dict['best_xi']
    )
    
    labels = optics.fit_predict(distance_matrix)
    
    non_noise_mask = labels != -1
    X_sub = distance_matrix[non_noise_mask][:, non_noise_mask]
    
    score = silhouette_score(X_sub, labels[non_noise_mask], metric='precomputed')
    
    clusters_list = labels
    
    cluster_dict = {}
    # Start new cluster IDs from the max existing ID + 1
    new_cluster_id = max(np.unique(clusters_list)) + 1
    
    total_amount_data = np.sum([len(client) for client in client_datasets])
    
    for cluster_index, obj in zip(clusters_list, client_datasets):
        if cluster_index == -1:
            index = str(new_cluster_id)
            new_cluster_id += 1
        else:
            index = str(cluster_index)
    
        client_data = len(obj) / total_amount_data
        if index not in cluster_dict:
            cluster_dict[index] = []
        cluster_dict[index].append((obj.id, client_data))
    
    # Adjust the cluster_range to exclude -1 and include the new clusters
    cluster_range = range(min(filter(lambda x: x != -1, np.unique(clusters_list))), new_cluster_id)
    
    cluster_distribution = [len(cluster_dict.get(str(i), [])) for i in cluster_range]
    
    best_dict['cluster_dict'] = cluster_dict
    
    print(f"Clients distribution among clusters: {cluster_distribution}")
    print("")
    print(f"Check that all clients have been clustered: {sum(cluster_distribution) == args.K}")

    print(f"\nTraining the {agg_method_used} aggregation algorithm using FedLab...")    
    
    rho_val = rho_haccs
    cluster_sampling = ['client', 'data', 'uniform']
    chosen_sampling = cluster_sampling[1]
    
    k_clust_selected = w_list[:-1][0]
      
    best_avg_accuracy = -1  # Initialize to a very low value
    
    for w in w_list[:-1]:
        for m in m_list:
            C = 0.5
            acc_list = []
            loss_list = []
            comm_size_list = []
            tta_list = []
            clients_list = []
            clusters_list = []
    
            # Initialize storage for metrics and training times
            global_acc_tests, global_pre_tests, global_rec_tests, global_f1s_tests = [], [], [], []
            weighted_global_acc_tests, weighted_global_pre_tests, weighted_global_rec_tests, weighted_global_f1s_tests = [], [], [], []                
            training_times = []
    
            metrics_per_client_per_seed = []
            model_parameters_per_seed = [] 
            
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                torch.cuda.manual_seed_all(seed)
    
                # Start training simulation
                start_time = time.time()
    
                # Define model
                if dataset_used == "acs_income" or dataset_used == "dutch":
                    model = LRModel_pytorch(len(covariates), len(np.unique(y_all)))
                elif dataset_used == "sent140":
                    model = DNN_pytorch(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
                elif dataset_used == "celeba":
                    model = DNN_celeba_pytorch(feat.shape[1:], len(np.unique(y_all)))
                
    
                handler = HACCS(
                    model = model,
                    global_round = args.comm_round,
                    sample_ratio = C
                )
    
                handler.setup_optim(d = args.d)
                handler.setup_clusters(cluster_dict = best_dict['cluster_dict'], cluster_sampling = chosen_sampling, w = k_clust_selected, m = m, rho = rho_val)
    
                # trainer = HACCSSerialClientTrainer(model, args.K, cuda=True)
                trainer = HACCSSerialClientTrainer(model, args.K, cuda=False)
    
                trainer.setup_optim(args.epochs, args.batch_size)
                trainer.setup_dataset(client_datasets)
    
                pipeline = HACCSPipeline(handler=handler, trainer=trainer, test_loader=test_loader)
    
                pipeline.main(args.min_comm_round)
    
                training_time = time.time() - start_time
                print("")
                print(f"Training for seed: {seed} completed.")
                print("")
    
   
                # Retrieve metrics
                global_acc_test = pipeline.get_accuracy()[-1]
                global_pre_test = 0
                global_rec_test = 0
                global_f1s_test = 0
    
                global_params_retrieved = pipeline.handler.model_parameters
                
                # Calculate metrics per client (to do weighted average)
                if dataset_used == "celeba":
                    metrics_weighted = evaluate_DNN_FL_celeba_pytorch(global_params_retrieved, list_x_test, list_y_test)
                else:
                    metrics_weighted = evaluate_LR_FL_pytorch(global_params_retrieved, list_x_test, list_y_test)

                metrics_per_client = metrics_weighted["client_metrics"]
                weighted_averages = metrics_weighted["weighted_averages"]
                
                # # Retrieve metrics
                weighted_global_acc_test = weighted_averages['accuracy']
                weighted_global_pre_test = weighted_averages['precision']
                weighted_global_rec_test = weighted_averages['recall']
                weighted_global_f1s_test = weighted_averages['f1score']
    
                
                # Store results
                global_acc_tests.append(global_acc_test)
                global_pre_tests.append(global_pre_test)
                global_rec_tests.append(global_rec_test)
                global_f1s_tests.append(global_f1s_test)
                weighted_global_acc_tests.append(weighted_global_acc_test)
                weighted_global_pre_tests.append(weighted_global_pre_test)
                weighted_global_rec_tests.append(weighted_global_rec_test)
                weighted_global_f1s_tests.append(weighted_global_f1s_test)             
                training_times.append(training_time)
    
                metrics_per_client_per_seed.append(dict(metrics_per_client))
                model_parameters_per_seed.append(global_params_retrieved)
    
                acc_list.append(pipeline.get_accuracy())
                loss_list.append(pipeline.get_loss())
                comm_size_list.append(pipeline.get_communication_size())
                tta_list.append(pipeline.get_time_list())
                clients_list.append(pipeline.get_selected_clients())
                clusters_list.append(pipeline.get_selected_clusters())
    
            # Average metrics history across seeds
            avg_commun_metrics_history = {'metrics_centralized':{'accuracy':list(enumerate(np.mean(acc_list,0)))}}
            
            # Compute means and standard deviations
            avg_global_acc_test = np.mean(global_acc_tests)
            std_global_acc_test = np.std(global_acc_tests)
            
            avg_global_pre_test = np.mean(global_pre_tests)
            std_global_pre_test = np.std(global_pre_tests)
            
            avg_global_rec_test = np.mean(global_rec_tests)
            std_global_rec_test = np.std(global_rec_tests)
            
            avg_global_f1s_test = np.mean(global_f1s_tests)
            std_global_f1s_test = np.std(global_f1s_tests)
            
            avg_training_time = np.mean(training_times)
            std_training_time = np.std(training_times)
                
            # Compute means and standard deviations for weight values
            weighted_avg_global_acc_test = np.mean(weighted_global_acc_tests)
            weighted_std_global_acc_test = np.std(weighted_global_acc_tests)
            
            weighted_avg_global_pre_test = np.mean(weighted_global_pre_tests)
            weighted_std_global_pre_test = np.std(weighted_global_pre_tests)
            
            weighted_avg_global_rec_test = np.mean(weighted_global_rec_tests)
            weighted_std_global_rec_test = np.std(weighted_global_rec_tests)
            
            weighted_avg_global_f1s_test = np.mean(weighted_global_f1s_tests)
            weighted_std_global_f1s_test = np.std(weighted_global_f1s_tests)
                
            
            print("===================")
            print(f"Completed m: {m}")
            print("===================")
    
        if avg_global_acc_test > best_avg_accuracy:
            best_avg_accuracy = avg_global_acc_test
            best_w_threshold = w
            
            # Save all best results
            best_results = {
                'w_threshold': w,
                'metrics': {
                    'accuracy': {'mean': avg_global_acc_test, 'std': std_global_acc_test},
                    'precision': {'mean': avg_global_pre_test, 'std': std_global_pre_test},
                    'recall': {'mean': avg_global_rec_test, 'std': std_global_rec_test},
                    'f1score': {'mean': avg_global_f1s_test, 'std': std_global_f1s_test},
                    'weighted_accuracy': {'mean': weighted_avg_global_acc_test, 'std': weighted_std_global_acc_test},
                    'weighted_precision': {'mean': weighted_avg_global_pre_test, 'std': weighted_std_global_pre_test},
                    'weighted_recall': {'mean': weighted_avg_global_rec_test, 'std': weighted_std_global_rec_test},
                    'weighted_f1score': {'mean': weighted_avg_global_f1s_test, 'std': weighted_std_global_f1s_test},
                    'metrics_per_client_per_seed': metrics_per_client_per_seed,
                    'model_parameters_per_seed': model_parameters_per_seed,
                },
                'training_time': {'mean': avg_training_time, 'std': std_training_time},
                'avg_commun_metrics_history': avg_commun_metrics_history
            }
    
       
        print("++++++++++++++++++++")
        print(f"Completed w: {w}")
        print("++++++++++++++++++++")
    
      
    best_w_threshold = best_results['w_threshold']
            
    # Compute means and standard deviations
    avg_global_acc_test = best_results["metrics"]["accuracy"]["mean"]
    std_global_acc_test = best_results["metrics"]["accuracy"]["std"]
    
    avg_global_pre_test = best_results["metrics"]["precision"]["mean"]
    std_global_pre_test = best_results["metrics"]["precision"]["std"]
    
    avg_global_rec_test = best_results["metrics"]["recall"]["mean"]
    std_global_rec_test = best_results["metrics"]["recall"]["std"]
    
    avg_global_f1s_test = best_results["metrics"]["f1score"]["mean"]
    std_global_f1s_test = best_results["metrics"]["f1score"]["std"]
    
    avg_training_time = best_results["training_time"]["mean"]
    std_training_time = best_results["training_time"]["std"]
    
    weighted_avg_global_acc_test = best_results["metrics"]["weighted_accuracy"]["mean"]
    weighted_std_global_acc_test = best_results["metrics"]["weighted_accuracy"]["std"]
    
    weighted_avg_global_pre_test = best_results["metrics"]["weighted_precision"]["mean"]
    weighted_std_global_pre_test = best_results["metrics"]["weighted_precision"]["std"]
    
    weighted_avg_global_rec_test = best_results["metrics"]["weighted_recall"]["mean"]
    weighted_std_global_rec_test = best_results["metrics"]["weighted_recall"]["std"]
    
    weighted_avg_global_f1s_test = best_results["metrics"]["weighted_f1score"]["mean"]
    weighted_std_global_f1s_test = best_results["metrics"]["weighted_f1score"]["std"]
    
    avg_commun_metrics_history = best_results["avg_commun_metrics_history"]
    
    # Print best results
    print("\n\nBEST RESULTS:")
    print(f'Best W Threshold: {best_w_threshold}')
    print(f'Global Accuracy: Mean = {avg_global_acc_test:.4f}, 'f'Std = {std_global_acc_test:.4f}')
    # print(f'Global Precision: Mean = {avg_global_pre_test:.4f}, 'f'Std = {std_global_pre_test:.4f}')
    # print(f'Global Recall: Mean = {avg_global_rec_test:.4f}, 'f'Std = {std_global_rec_test:.4f}')
    # print(f'Global F1-Score: Mean = {avg_global_f1s_test:.4f}, 'f'Std = {std_global_f1s_test:.4f}')
    print("************************************************************************************")
    # print(f'Weighted Accuracy: Mean = {weighted_avg_global_acc_test:.4f}, Std = {weighted_std_global_acc_test:.4f}')
    # print(f'Weighted Precision: Mean = {weighted_avg_global_pre_test:.4f}, Std = {weighted_std_global_pre_test:.4f}')
    # print(f'Weighted Recall: Mean = {weighted_avg_global_rec_test:.4f}, Std = {weighted_std_global_rec_test:.4f}')
    # print(f'Weighted F1-Score: Mean = {weighted_avg_global_f1s_test:.4f}, Std = {weighted_std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')



def train_fedcls(clients_glob, clients_glob_test, local_nodes_glob, random_state, seeds, epochs, comms_round):

    K = local_nodes_glob
    global batch_size, y_test_tensor,args
    batch_size = 64
    lr = learn_rate
    min_comm_round = 0
    comm_round = comms_round
    d = d_poc                                                     
    w_list = get_m_list(d)
    m_list = [1]
    
    # configuration
    parser = argparse.ArgumentParser(description="FinalModel")
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--d", type=int, default=d)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--comm_round", type=int, default=comm_round)
    parser.add_argument("--min_comm_round", type=int, default=min_comm_round)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    args = parser.parse_args(args=[])

    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(clients_dict=clients_glob)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_test, list_y_test = from_FedArtML_to_Flower_format(clients_dict=clients_glob_test)
    

    if dataset_used == "celeba":
        # Convert the numpy arrays to PyTorch tensors
        x_test_tensor = torch.tensor(np.transpose(X_test, (0, 3, 1,2)), dtype=torch.float32)
        y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
        
        print("X Test shape:", np.transpose(X_test, (0, 3, 1,2)).shape)
        print("Y Test shape:", Y_test.shape)
        
        # Create a TensorDataset from the test data
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)  # unsqueeze to add a channel dimension
        
        # Create a DataLoader from the TensorDataset
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        clients_train = clients_glob
        
        clients_data_sizes = [len(clients_train[f'Client_{i+1}']) for i in range(args.K)]
        prob_distribution = list(clients_data_sizes / np.sum(clients_data_sizes))
        
        client_datasets = []
        client_names = list(clients_train.keys())
        
        for client_id, client in enumerate(client_names):
            # Get data from each client
            each_client_train = np.array(clients_train[client], dtype=object)
        
            # Extract features for each client
            feat = np.transpose(np.array(each_client_train[:, 0].tolist()), (0, 3, 1,2))
            # Extract labels from each client
            y_tra = np.array(each_client_train[:, 1])
        
            client_dataset = ClientDataset(feat, y_tra)
            client_dataset.id = client_id  # Assign the id to the ClientDataset object
        
            client_datasets.append(client_dataset)
    else:
        # Convert the numpy arrays to PyTorch tensors
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
        
        print("X Test shape:", X_test.shape)
        print("Y Test shape:", Y_test.shape)
        
        # Create a TensorDataset from the test data
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)  # unsqueeze to add a channel dimension
        
        # Create a DataLoader from the TensorDataset
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        clients_train = clients_glob
        # hd = round(distances['without_class_completion']['hellinger'], 2)
        
        clients_data_sizes = [len(clients_train[f'Client_{i+1}']) for i in range(args.K)]
        prob_distribution = list(clients_data_sizes / np.sum(clients_data_sizes))
        
        client_datasets = []
        client_names = list(clients_train.keys())
        
        for client_id, client in enumerate(client_names):
            # Get data from each client
            each_client_train = np.array(clients_train[client], dtype=object)
        
            # Extract features for each client
            feat = np.array(each_client_train[:, 0].tolist())
        
            # Extract labels from each client
            y_tra = np.array(each_client_train[:, 1])
        
            client_dataset = ClientDataset(feat, y_tra)
            client_dataset.id = client_id  # Assign the id to the ClientDataset object
        
            client_datasets.append(client_dataset)


    print(f"\nClustering clients for the {agg_method_used} aggregation algorithm...")    

    label_per_client = []
    for client_dataset in client_datasets:
        label_per_client.append(client_dataset.labels)
    
    # Define a distance threshold (tune based on desired granularity)
    threshold = thl_fedcls  # Adjust threshold to control similarity tolerance
    
    clusters, cluster_labels = clients_clustering(label_per_client, threshold)
    print(clusters)
    # Fix the problem when all the clients belong to one cluster
    if len(clusters) == 0:
        random.seed(random_state)
        nodes = list(range(local_nodes_glob))
        random.shuffle(nodes)
        
        one_part = int(local_nodes_glob * 0.2)
        complement = local_nodes_glob - one_part
        clusters = [nodes[:one_part]]
        cluster_labels = [0] * one_part + [-1] * complement
        
    print("Clusters:", clusters)
    # print("Number of clusters:",sum([len(i) for i in clusters]))
    print("Number of clusters:",len(clusters))
    print("Cluster Labels:", cluster_labels)
    
    clusters_list = cluster_labels
    
    cluster_dict = {}
    # Start new cluster IDs from the max existing ID + 1
    new_cluster_id = max(np.unique(clusters_list)) + 1
    
    total_amount_data = np.sum([len(client) for client in client_datasets])
    
    for cluster_index, obj in zip(clusters_list, client_datasets):
        if cluster_index == -1:
            index = str(new_cluster_id)
            new_cluster_id += 1
        else:
            index = str(cluster_index)
    
        client_data = len(obj) / total_amount_data
        if index not in cluster_dict:
            cluster_dict[index] = []
        cluster_dict[index].append((obj.id, client_data))
    
    # Adjust the cluster_range to exclude -1 and include the new clusters
    cluster_range = range(min(filter(lambda x: x != -1, np.unique(clusters_list))), new_cluster_id)
    
    cluster_distribution = [len(cluster_dict.get(str(i), [])) for i in cluster_range]

    best_dict = {}
    best_dict['cluster_dict'] = cluster_dict
    
    print(f"Clients distribution among clusters: {cluster_distribution}")
    print("")
    print(f"Check that all clients have been clustered: {sum(cluster_distribution) == args.K}")
    
    print(f"\nTraining the {agg_method_used} aggregation algorithm using FedLab...")    

    cluster_sampling = ['client', 'data', 'uniform']
    chosen_sampling = cluster_sampling[1]

    best_avg_accuracy = -1  # Initialize to a very low value
    
    for w in w_list[:-1]:        
        for m in m_list:
            C = 0.5
            acc_list = []
            loss_list = []
            comm_size_list = []
            tta_list = []
            clients_list = []
            clusters_list = []
    
            # Initialize storage for metrics and training times
            global_acc_tests, global_pre_tests, global_rec_tests, global_f1s_tests = [], [], [], []
            weighted_global_acc_tests, weighted_global_pre_tests, weighted_global_rec_tests, weighted_global_f1s_tests = [], [], [], []                
            training_times = []
    
            metrics_per_client_per_seed = []
            model_parameters_per_seed = [] 
            
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                torch.cuda.manual_seed_all(seed)
    
                # Start training simulation
                start_time = time.time()
    
                # Define model
                if dataset_used == "acs_income" or dataset_used == "dutch":
                    model = LRModel_pytorch(len(covariates), len(np.unique(y_all)))
                elif dataset_used == "sent140":
                    model = DNN_pytorch(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
                elif dataset_used == "celeba":
                    model = DNN_celeba_pytorch(feat.shape[1:], len(np.unique(y_all)))
                
    
                handler = FedCLS(
                    model = model,
                    global_round = args.comm_round,
                    sample_ratio = C
                )
    
                handler.setup_optim(d = args.d)
                handler.setup_clusters(cluster_dict = best_dict['cluster_dict'], cluster_sampling = chosen_sampling, w = w, m = m)
    
                # trainer = ClusteredPowerofchoiceSerialClientTrainer(model, args.K, cuda=True)
                trainer = ClusteredPowerofchoiceSerialClientTrainer(model, args.K, cuda=False)
    
                trainer.setup_optim(args.epochs, args.batch_size)
                trainer.setup_dataset(client_datasets)
    
                pipeline = FedCLSPipeline(handler=handler, trainer=trainer, test_loader=test_loader)
    
                pipeline.main(args.min_comm_round)
    
                training_time = time.time() - start_time
                print("")
                print(f"Training for seed: {seed} completed.")
                print("")
    
    
                # Retrieve metrics
                global_acc_test = pipeline.get_accuracy()[-1]
                global_pre_test = 0
                global_rec_test = 0
                global_f1s_test = 0
    
                global_params_retrieved = pipeline.handler.model_parameters

                # Calculate metrics per client (to do weighted average)
                if dataset_used == "celeba":
                    metrics_weighted = evaluate_DNN_FL_celeba_pytorch(global_params_retrieved, list_x_test, list_y_test)
                else:
                    metrics_weighted = evaluate_LR_FL_pytorch(global_params_retrieved, list_x_test, list_y_test)

                metrics_per_client = metrics_weighted["client_metrics"]
                weighted_averages = metrics_weighted["weighted_averages"]
                
                # # Retrieve metrics
                weighted_global_acc_test = weighted_averages['accuracy']
                weighted_global_pre_test = weighted_averages['precision']
                weighted_global_rec_test = weighted_averages['recall']
                weighted_global_f1s_test = weighted_averages['f1score']
    
                
                # Store results
                global_acc_tests.append(global_acc_test)
                global_pre_tests.append(global_pre_test)
                global_rec_tests.append(global_rec_test)
                global_f1s_tests.append(global_f1s_test)
                weighted_global_acc_tests.append(weighted_global_acc_test)
                weighted_global_pre_tests.append(weighted_global_pre_test)
                weighted_global_rec_tests.append(weighted_global_rec_test)
                weighted_global_f1s_tests.append(weighted_global_f1s_test)             
                training_times.append(training_time)
    
                metrics_per_client_per_seed.append(dict(metrics_per_client))
                model_parameters_per_seed.append(global_params_retrieved)
    
                acc_list.append(pipeline.get_accuracy())
                loss_list.append(pipeline.get_loss())
                comm_size_list.append(pipeline.get_communication_size())
                tta_list.append(pipeline.get_time_list())
                clients_list.append(pipeline.get_selected_clients())
                clusters_list.append(pipeline.get_selected_clusters())
    
            # Average metrics history across seeds
            avg_commun_metrics_history = {'metrics_centralized':{'accuracy':list(enumerate(np.mean(acc_list,0)))}}
            
            # Compute means and standard deviations
            avg_global_acc_test = np.mean(global_acc_tests)
            std_global_acc_test = np.std(global_acc_tests)
            
            avg_global_pre_test = np.mean(global_pre_tests)
            std_global_pre_test = np.std(global_pre_tests)
            
            avg_global_rec_test = np.mean(global_rec_tests)
            std_global_rec_test = np.std(global_rec_tests)
            
            avg_global_f1s_test = np.mean(global_f1s_tests)
            std_global_f1s_test = np.std(global_f1s_tests)
            
            avg_training_time = np.mean(training_times)
            std_training_time = np.std(training_times)
                
            # Compute means and standard deviations for weight values
            weighted_avg_global_acc_test = np.mean(weighted_global_acc_tests)
            weighted_std_global_acc_test = np.std(weighted_global_acc_tests)
            
            weighted_avg_global_pre_test = np.mean(weighted_global_pre_tests)
            weighted_std_global_pre_test = np.std(weighted_global_pre_tests)
            
            weighted_avg_global_rec_test = np.mean(weighted_global_rec_tests)
            weighted_std_global_rec_test = np.std(weighted_global_rec_tests)
            
            weighted_avg_global_f1s_test = np.mean(weighted_global_f1s_tests)
            weighted_std_global_f1s_test = np.std(weighted_global_f1s_tests)
                
            
            print("===================")
            print(f"Completed m: {m}")
            print("===================")
    
    
        if avg_global_acc_test > best_avg_accuracy:
            best_avg_accuracy = avg_global_acc_test
            best_w_threshold = w
            
            # Save all best results
            best_results = {
                'w_threshold': w,
                'metrics': {
                    'accuracy': {'mean': avg_global_acc_test, 'std': std_global_acc_test},
                    'precision': {'mean': avg_global_pre_test, 'std': std_global_pre_test},
                    'recall': {'mean': avg_global_rec_test, 'std': std_global_rec_test},
                    'f1score': {'mean': avg_global_f1s_test, 'std': std_global_f1s_test},
                    'weighted_accuracy': {'mean': weighted_avg_global_acc_test, 'std': weighted_std_global_acc_test},
                    'weighted_precision': {'mean': weighted_avg_global_pre_test, 'std': weighted_std_global_pre_test},
                    'weighted_recall': {'mean': weighted_avg_global_rec_test, 'std': weighted_std_global_rec_test},
                    'weighted_f1score': {'mean': weighted_avg_global_f1s_test, 'std': weighted_std_global_f1s_test},
                    'metrics_per_client_per_seed': metrics_per_client_per_seed,
                    'model_parameters_per_seed': model_parameters_per_seed,
                },
                'training_time': {'mean': avg_training_time, 'std': std_training_time},
                'avg_commun_metrics_history': avg_commun_metrics_history
            }
    
       
        print("++++++++++++++++++++")
        print(f"Completed w: {w}")
        print("++++++++++++++++++++")
    
      
    best_w_threshold = best_results['w_threshold']
            
    # Compute means and standard deviations
    avg_global_acc_test = best_results["metrics"]["accuracy"]["mean"]
    std_global_acc_test = best_results["metrics"]["accuracy"]["std"]
    
    avg_global_pre_test = best_results["metrics"]["precision"]["mean"]
    std_global_pre_test = best_results["metrics"]["precision"]["std"]
    
    avg_global_rec_test = best_results["metrics"]["recall"]["mean"]
    std_global_rec_test = best_results["metrics"]["recall"]["std"]
    
    avg_global_f1s_test = best_results["metrics"]["f1score"]["mean"]
    std_global_f1s_test = best_results["metrics"]["f1score"]["std"]
    
    avg_training_time = best_results["training_time"]["mean"]
    std_training_time = best_results["training_time"]["std"]
    
    weighted_avg_global_acc_test = best_results["metrics"]["weighted_accuracy"]["mean"]
    weighted_std_global_acc_test = best_results["metrics"]["weighted_accuracy"]["std"]
    
    weighted_avg_global_pre_test = best_results["metrics"]["weighted_precision"]["mean"]
    weighted_std_global_pre_test = best_results["metrics"]["weighted_precision"]["std"]
    
    weighted_avg_global_rec_test = best_results["metrics"]["weighted_recall"]["mean"]
    weighted_std_global_rec_test = best_results["metrics"]["weighted_recall"]["std"]
    
    weighted_avg_global_f1s_test = best_results["metrics"]["weighted_f1score"]["mean"]
    weighted_std_global_f1s_test = best_results["metrics"]["weighted_f1score"]["std"]
    
    avg_commun_metrics_history = best_results["avg_commun_metrics_history"]
    
    # Print best results
    print("\n\nBEST RESULTS:")
    print(f'Best W Threshold: {best_w_threshold}')
    print(f'Global Accuracy: Mean = {avg_global_acc_test:.4f}, 'f'Std = {std_global_acc_test:.4f}')
    # print(f'Global Precision: Mean = {avg_global_pre_test:.4f}, 'f'Std = {std_global_pre_test:.4f}')
    # print(f'Global Recall: Mean = {avg_global_rec_test:.4f}, 'f'Std = {std_global_rec_test:.4f}')
    # print(f'Global F1-Score: Mean = {avg_global_f1s_test:.4f}, 'f'Std = {std_global_f1s_test:.4f}')
    print("************************************************************************************")
    # print(f'Weighted Accuracy: Mean = {weighted_avg_global_acc_test:.4f}, Std = {weighted_std_global_acc_test:.4f}')
    # print(f'Weighted Precision: Mean = {weighted_avg_global_pre_test:.4f}, Std = {weighted_std_global_pre_test:.4f}')
    # print(f'Weighted Recall: Mean = {weighted_avg_global_rec_test:.4f}, Std = {weighted_std_global_rec_test:.4f}')
    # print(f'Weighted F1-Score: Mean = {weighted_avg_global_f1s_test:.4f}, Std = {weighted_std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')




def train_cfl(clients_glob, clients_glob_test, local_nodes_glob, random_state, seeds, epochs, comms_round):

    K = local_nodes_glob
    global batch_size, y_test_tensor,args
    batch_size = 64
    lr = learn_rate
    min_comm_round = 0
    comm_round = comms_round
    d = d_poc                                                     
    w_list = get_m_list(d)
    m_list = [1]
    
    # configuration
    parser = argparse.ArgumentParser(description="FinalModel")
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--d", type=int, default=d)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--comm_round", type=int, default=comm_round)
    parser.add_argument("--min_comm_round", type=int, default=min_comm_round)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    args = parser.parse_args(args=[])

    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(clients_dict=clients_glob)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_test, list_y_test = from_FedArtML_to_Flower_format(clients_dict=clients_glob_test)
    

    if dataset_used == "celeba":
        # Convert the numpy arrays to PyTorch tensors
        x_test_tensor = torch.tensor(np.transpose(X_test, (0, 3, 1,2)), dtype=torch.float32)
        y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
        
        print("X Test shape:", np.transpose(X_test, (0, 3, 1,2)).shape)
        print("Y Test shape:", Y_test.shape)
        
        # Create a TensorDataset from the test data
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)  # unsqueeze to add a channel dimension
        
        # Create a DataLoader from the TensorDataset
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        clients_train = clients_glob
        
        clients_data_sizes = [len(clients_train[f'Client_{i+1}']) for i in range(args.K)]
        prob_distribution = list(clients_data_sizes / np.sum(clients_data_sizes))
        
        client_datasets = []
        client_names = list(clients_train.keys())
        
        for client_id, client in enumerate(client_names):
            # Get data from each client
            each_client_train = np.array(clients_train[client], dtype=object)
        
            # Extract features for each client
            feat = np.transpose(np.array(each_client_train[:, 0].tolist()), (0, 3, 1,2))
            # Extract labels from each client
            y_tra = np.array(each_client_train[:, 1])
        
            client_dataset = ClientDataset(feat, y_tra)
            client_dataset.id = client_id  # Assign the id to the ClientDataset object
        
            client_datasets.append(client_dataset)
    else:
        # Convert the numpy arrays to PyTorch tensors
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
        
        print("X Test shape:", X_test.shape)
        print("Y Test shape:", Y_test.shape)
        
        # Create a TensorDataset from the test data
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)  # unsqueeze to add a channel dimension
        
        # Create a DataLoader from the TensorDataset
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        clients_train = clients_glob
        # hd = round(distances['without_class_completion']['hellinger'], 2)
        
        clients_data_sizes = [len(clients_train[f'Client_{i+1}']) for i in range(args.K)]
        prob_distribution = list(clients_data_sizes / np.sum(clients_data_sizes))
        
        client_datasets = []
        client_names = list(clients_train.keys())
        
        for client_id, client in enumerate(client_names):
            # Get data from each client
            each_client_train = np.array(clients_train[client], dtype=object)
        
            # Extract features for each client
            feat = np.array(each_client_train[:, 0].tolist())
        
            # Extract labels from each client
            y_tra = np.array(each_client_train[:, 1])
        
            client_dataset = ClientDataset(feat, y_tra)
            client_dataset.id = client_id  # Assign the id to the ClientDataset object
        
            client_datasets.append(client_dataset)


    print(f"\nTraining the {agg_method_used} aggregation algorithm using FedLab...")    

    # Define hyperparameters
    epsilon1_thresholds = eps_1_range
    epsilon2_range = eps_2_range
    gamma_max_range = gam_max_range

    best_avg_accuracy = -1  # Initialize to a very low value
    
    i = 0
    for eps1 in epsilon1_thresholds:        
        C = 0.5
        acc_list = []
        loss_list = []
        comm_size_list = []
        tta_list = []
        clients_list = []
        clusters_list = []

        # Initialize storage for metrics and training times
        global_acc_tests, global_pre_tests, global_rec_tests, global_f1s_tests = [], [], [], []
        weighted_global_acc_tests, weighted_global_pre_tests, weighted_global_rec_tests, weighted_global_f1s_tests = [], [], [], []                
        training_times = []

        metrics_per_client_per_seed = []
        model_parameters_per_seed = [] 
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed_all(seed)

            # Start training simulation
            start_time = time.time()

            # Define model
            if dataset_used == "acs_income" or dataset_used == "dutch":
                model = LRModel_pytorch(len(covariates), len(np.unique(y_all)))
            elif dataset_used == "sent140":
                model = DNN_pytorch(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
            elif dataset_used == "celeba":
                model = DNN_celeba_pytorch(feat.shape[1:], len(np.unique(y_all)))
            

            # Pipeline initialization remains the same
            cfl_handler = CFLServerHandler(
                model,
                total_rounds=args.comm_round,
                sample_ratio=C,
                epsilon1=eps1,
                epsilon2=epsilon2_range[i],
                gamma_max=gamma_max_range[i]
            )

            cfl_trainer = CFLSerialClientTrainer(model, num_clients=args.K, cuda=False)
            cfl_trainer.setup_dataset(client_datasets)
            cfl_trainer.setup_optim(epochs=args.epochs, batch_size=args.batch_size, lr=lr)
            
            pipeline = CFLPipeline(cfl_handler, cfl_trainer, test_loader)
            final_clusters = pipeline.main()    

            training_time = time.time() - start_time
            print("")
            print(f"Training for seed: {seed} completed.")
            print("")

            # Retrieve metrics
            # global_acc_test = pipeline.get_accuracy()[-1]
            global_acc_test = max(pipeline.get_accuracy())
            global_pre_test = 0
            global_rec_test = 0
            global_f1s_test = 0

            global_params_retrieved = pipeline.root_handler.model_parameters

            # Calculate metrics per client (to do weighted average)
            if dataset_used == "celeba":
                metrics_weighted = evaluate_DNN_FL_celeba_pytorch(global_params_retrieved, list_x_test, list_y_test)
            else:
                metrics_weighted = evaluate_LR_FL_pytorch(global_params_retrieved, list_x_test, list_y_test)
 
            metrics_per_client = metrics_weighted["client_metrics"]
            weighted_averages = metrics_weighted["weighted_averages"]
            
            # # Retrieve metrics
            weighted_global_acc_test = weighted_averages['accuracy']
            weighted_global_pre_test = weighted_averages['precision']
            weighted_global_rec_test = weighted_averages['recall']
            weighted_global_f1s_test = weighted_averages['f1score']

            
            # Store results
            global_acc_tests.append(global_acc_test)
            global_pre_tests.append(global_pre_test)
            global_rec_tests.append(global_rec_test)
            global_f1s_tests.append(global_f1s_test)
            weighted_global_acc_tests.append(weighted_global_acc_test)
            weighted_global_pre_tests.append(weighted_global_pre_test)
            weighted_global_rec_tests.append(weighted_global_rec_test)
            weighted_global_f1s_tests.append(weighted_global_f1s_test)             
            training_times.append(training_time)

            metrics_per_client_per_seed.append(dict(metrics_per_client))
            model_parameters_per_seed.append(global_params_retrieved)

            acc_list.append(pipeline.get_accuracy())

        # Average metrics history across seeds
        avg_commun_metrics_history = {'metrics_centralized':{'accuracy':list(enumerate(np.mean(acc_list,0)))}}
        
        # Compute means and standard deviations
        avg_global_acc_test = np.mean(global_acc_tests)
        std_global_acc_test = np.std(global_acc_tests)
        
        avg_global_pre_test = np.mean(global_pre_tests)
        std_global_pre_test = np.std(global_pre_tests)
        
        avg_global_rec_test = np.mean(global_rec_tests)
        std_global_rec_test = np.std(global_rec_tests)
        
        avg_global_f1s_test = np.mean(global_f1s_tests)
        std_global_f1s_test = np.std(global_f1s_tests)
        
        avg_training_time = np.mean(training_times)
        std_training_time = np.std(training_times)
            
        # Compute means and standard deviations for weight values
        weighted_avg_global_acc_test = np.mean(weighted_global_acc_tests)
        weighted_std_global_acc_test = np.std(weighted_global_acc_tests)
        
        weighted_avg_global_pre_test = np.mean(weighted_global_pre_tests)
        weighted_std_global_pre_test = np.std(weighted_global_pre_tests)
        
        weighted_avg_global_rec_test = np.mean(weighted_global_rec_tests)
        weighted_std_global_rec_test = np.std(weighted_global_rec_tests)
        
        weighted_avg_global_f1s_test = np.mean(weighted_global_f1s_tests)
        weighted_std_global_f1s_test = np.std(weighted_global_f1s_tests)
            

        if avg_global_acc_test > best_avg_accuracy:
            best_avg_accuracy = avg_global_acc_test
            best_eps1_threshold = eps1
            
            # Save all best results
            best_results = {
                'eps1_threshold': eps1,
                'metrics': {
                    'accuracy': {'mean': avg_global_acc_test, 'std': std_global_acc_test},
                    'precision': {'mean': avg_global_pre_test, 'std': std_global_pre_test},
                    'recall': {'mean': avg_global_rec_test, 'std': std_global_rec_test},
                    'f1score': {'mean': avg_global_f1s_test, 'std': std_global_f1s_test},
                    'weighted_accuracy': {'mean': weighted_avg_global_acc_test, 'std': weighted_std_global_acc_test},
                    'weighted_precision': {'mean': weighted_avg_global_pre_test, 'std': weighted_std_global_pre_test},
                    'weighted_recall': {'mean': weighted_avg_global_rec_test, 'std': weighted_std_global_rec_test},
                    'weighted_f1score': {'mean': weighted_avg_global_f1s_test, 'std': weighted_std_global_f1s_test},
                    'metrics_per_client_per_seed': metrics_per_client_per_seed,
                    'model_parameters_per_seed': model_parameters_per_seed,
                },
                'training_time': {'mean': avg_training_time, 'std': std_training_time},
                'avg_commun_metrics_history': avg_commun_metrics_history
            }

        i+=1
        
        print("===================")
        print(f"Completed epsilon 1: {eps1}")
        print("===================")
    
      
    best_eps1_threshold = best_results['eps1_threshold']
            
    # Compute means and standard deviations
    avg_global_acc_test = best_results["metrics"]["accuracy"]["mean"]
    std_global_acc_test = best_results["metrics"]["accuracy"]["std"]
    
    avg_global_pre_test = best_results["metrics"]["precision"]["mean"]
    std_global_pre_test = best_results["metrics"]["precision"]["std"]
    
    avg_global_rec_test = best_results["metrics"]["recall"]["mean"]
    std_global_rec_test = best_results["metrics"]["recall"]["std"]
    
    avg_global_f1s_test = best_results["metrics"]["f1score"]["mean"]
    std_global_f1s_test = best_results["metrics"]["f1score"]["std"]
    
    avg_training_time = best_results["training_time"]["mean"]
    std_training_time = best_results["training_time"]["std"]
    
    weighted_avg_global_acc_test = best_results["metrics"]["weighted_accuracy"]["mean"]
    weighted_std_global_acc_test = best_results["metrics"]["weighted_accuracy"]["std"]
    
    weighted_avg_global_pre_test = best_results["metrics"]["weighted_precision"]["mean"]
    weighted_std_global_pre_test = best_results["metrics"]["weighted_precision"]["std"]
    
    weighted_avg_global_rec_test = best_results["metrics"]["weighted_recall"]["mean"]
    weighted_std_global_rec_test = best_results["metrics"]["weighted_recall"]["std"]
    
    weighted_avg_global_f1s_test = best_results["metrics"]["weighted_f1score"]["mean"]
    weighted_std_global_f1s_test = best_results["metrics"]["weighted_f1score"]["std"]
    
    avg_commun_metrics_history = best_results["avg_commun_metrics_history"]
    
    # Print best results
    print("\n\nBEST RESULTS:")
    print(f'Best W Threshold: {best_eps1_threshold}')
    print(f'Global Accuracy: Mean = {avg_global_acc_test:.4f}, 'f'Std = {std_global_acc_test:.4f}')
    # print(f'Global Precision: Mean = {avg_global_pre_test:.4f}, 'f'Std = {std_global_pre_test:.4f}')
    # print(f'Global Recall: Mean = {avg_global_rec_test:.4f}, 'f'Std = {std_global_rec_test:.4f}')
    # print(f'Global F1-Score: Mean = {avg_global_f1s_test:.4f}, 'f'Std = {std_global_f1s_test:.4f}')
    print("************************************************************************************")
    # print(f'Weighted Accuracy: Mean = {weighted_avg_global_acc_test:.4f}, Std = {weighted_std_global_acc_test:.4f}')
    # print(f'Weighted Precision: Mean = {weighted_avg_global_pre_test:.4f}, Std = {weighted_std_global_pre_test:.4f}')
    # print(f'Weighted Recall: Mean = {weighted_avg_global_rec_test:.4f}, Std = {weighted_std_global_rec_test:.4f}')
    # print(f'Weighted F1-Score: Mean = {weighted_avg_global_f1s_test:.4f}, Std = {weighted_std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')
    




def train_fedsoft(clients_glob, clients_glob_test, local_nodes_glob, random_state, seeds, epochs, comms_round):

    K = local_nodes_glob
    global batch_size, y_test_tensor,args
    batch_size = 64
    lr = learn_rate
    min_comm_round = 0
    comm_round = comms_round
    d = d_poc                                                     
    w_list = get_m_list(d)
    m_list = [1]
    
    # configuration
    parser = argparse.ArgumentParser(description="FinalModel")
    parser.add_argument("--K", type=int, default=K)
    parser.add_argument("--d", type=int, default=d)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--comm_round", type=int, default=comm_round)
    parser.add_argument("--min_comm_round", type=int, default=min_comm_round)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)
    args = parser.parse_args(args=[])

    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(clients_dict=clients_glob)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_test, list_y_test = from_FedArtML_to_Flower_format(clients_dict=clients_glob_test)
    

    if dataset_used == "celeba":
        # Convert the numpy arrays to PyTorch tensors
        x_test_tensor = torch.tensor(np.transpose(X_test, (0, 3, 1,2)), dtype=torch.float32)
        y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
        
        print("X Test shape:", np.transpose(X_test, (0, 3, 1,2)).shape)
        print("Y Test shape:", Y_test.shape)
        
        # Create a TensorDataset from the test data
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)  # unsqueeze to add a channel dimension
        
        # Create a DataLoader from the TensorDataset
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        clients_train = clients_glob
        
        clients_data_sizes = [len(clients_train[f'Client_{i+1}']) for i in range(args.K)]
        prob_distribution = list(clients_data_sizes / np.sum(clients_data_sizes))
        
        client_datasets = []
        client_names = list(clients_train.keys())
        
        for client_id, client in enumerate(client_names):
            # Get data from each client
            each_client_train = np.array(clients_train[client], dtype=object)
        
            # Extract features for each client
            feat = np.transpose(np.array(each_client_train[:, 0].tolist()), (0, 3, 1,2))
            # Extract labels from each client
            y_tra = np.array(each_client_train[:, 1])
        
            client_dataset = ClientDataset(feat, y_tra)
            client_dataset.id = client_id  # Assign the id to the ClientDataset object
        
            client_datasets.append(client_dataset)
    else:
        # Convert the numpy arrays to PyTorch tensors
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
        
        print("X Test shape:", X_test.shape)
        print("Y Test shape:", Y_test.shape)
        
        # Create a TensorDataset from the test data
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)  # unsqueeze to add a channel dimension
        
        # Create a DataLoader from the TensorDataset
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        clients_train = clients_glob
        # hd = round(distances['without_class_completion']['hellinger'], 2)
        
        clients_data_sizes = [len(clients_train[f'Client_{i+1}']) for i in range(args.K)]
        prob_distribution = list(clients_data_sizes / np.sum(clients_data_sizes))
        
        client_datasets = []
        client_names = list(clients_train.keys())
        
        for client_id, client in enumerate(client_names):
            # Get data from each client
            each_client_train = np.array(clients_train[client], dtype=object)
        
            # Extract features for each client
            feat = np.array(each_client_train[:, 0].tolist())
        
            # Extract labels from each client
            y_tra = np.array(each_client_train[:, 1])
        
            client_dataset = ClientDataset(feat, y_tra)
            client_dataset.id = client_id  # Assign the id to the ClientDataset object
        
            client_datasets.append(client_dataset)


    print(f"\nTraining the {agg_method_used} aggregation algorithm using FedLab...")    

    # Define hyperparameters
    num_clusters_thresholds = n_clust_range
    
    best_avg_accuracy = -1  # Initialize to a very low value

    i = 0
    for num_clusters in num_clusters_thresholds:        
        C = 0.5
        acc_list = []
        loss_list = []
        comm_size_list = []
        tta_list = []
        clients_list = []
        clusters_list = []

        # Initialize storage for metrics and training times
        global_acc_tests, global_pre_tests, global_rec_tests, global_f1s_tests = [], [], [], []
        weighted_global_acc_tests, weighted_global_pre_tests, weighted_global_rec_tests, weighted_global_f1s_tests = [], [], [], []                
        training_times = []

        metrics_per_client_per_seed = []
        model_parameters_per_seed = [] 
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed_all(seed)

            # Start training simulation
            start_time = time.time()

            # Define model
            if dataset_used == "acs_income" or dataset_used == "dutch":
                model = LRModel_pytorch(len(covariates), len(np.unique(y_all)))
            elif dataset_used == "sent140":
                model = DNN_pytorch(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
            elif dataset_used == "celeba":
                model = DNN_celeba_pytorch(feat.shape[1:], len(np.unique(y_all)))
                    

            handler = FedSoftServerHandler(
                model=model,
                num_clusters=num_clusters,
                total_rounds=args.comm_round,
                sample_ratio=C,
                num_clients=args.K  # This is now handled in child class
            )

            trainer = FedSoftClientTrainer(
                model=model,
                num_clusters=num_clusters,
                num_clients=args.K,  # This should match handler's num_clients
                cuda=False
            )
            
            trainer.setup_dataset(client_datasets)
            trainer.setup_optim(epochs=args.epochs, batch_size=args.batch_size, lr=lr)
            
            pipeline = FedSoftPipeline(handler, trainer, test_loader)
            pipeline.main()

            training_time = time.time() - start_time
            print("")
            print(f"Training for seed: {seed} completed.")
            print("")

            # Retrieve metrics
            # global_acc_test = pipeline.get_accuracy()[-1]
            global_acc_test = max(pipeline.get_accuracy())
            global_pre_test = 0
            global_rec_test = 0
            global_f1s_test = 0

            global_params_retrieved = pipeline.handler.model_parameters

            # Calculate metrics per client (to do weighted average)
            if dataset_used == "celeba":
                metrics_weighted = evaluate_DNN_FL_celeba_pytorch(global_params_retrieved, list_x_test, list_y_test)
            else:
                metrics_weighted = evaluate_LR_FL_pytorch(global_params_retrieved, list_x_test, list_y_test)

            metrics_per_client = metrics_weighted["client_metrics"]
            weighted_averages = metrics_weighted["weighted_averages"]
            
            # # Retrieve metrics
            weighted_global_acc_test = weighted_averages['accuracy']
            weighted_global_pre_test = weighted_averages['precision']
            weighted_global_rec_test = weighted_averages['recall']
            weighted_global_f1s_test = weighted_averages['f1score']

            
            # Store results
            global_acc_tests.append(global_acc_test)
            global_pre_tests.append(global_pre_test)
            global_rec_tests.append(global_rec_test)
            global_f1s_tests.append(global_f1s_test)
            weighted_global_acc_tests.append(weighted_global_acc_test)
            weighted_global_pre_tests.append(weighted_global_pre_test)
            weighted_global_rec_tests.append(weighted_global_rec_test)
            weighted_global_f1s_tests.append(weighted_global_f1s_test)             
            training_times.append(training_time)

            metrics_per_client_per_seed.append(dict(metrics_per_client))
            model_parameters_per_seed.append(global_params_retrieved)

            acc_list.append(pipeline.get_accuracy())

        # Average metrics history across seeds
        avg_commun_metrics_history = {'metrics_centralized':{'accuracy':list(enumerate(np.mean(acc_list,0)))}}
        
        # Compute means and standard deviations
        avg_global_acc_test = np.mean(global_acc_tests)
        std_global_acc_test = np.std(global_acc_tests)
        
        avg_global_pre_test = np.mean(global_pre_tests)
        std_global_pre_test = np.std(global_pre_tests)
        
        avg_global_rec_test = np.mean(global_rec_tests)
        std_global_rec_test = np.std(global_rec_tests)
        
        avg_global_f1s_test = np.mean(global_f1s_tests)
        std_global_f1s_test = np.std(global_f1s_tests)
        
        avg_training_time = np.mean(training_times)
        std_training_time = np.std(training_times)
            
        # Compute means and standard deviations for weight values
        weighted_avg_global_acc_test = np.mean(weighted_global_acc_tests)
        weighted_std_global_acc_test = np.std(weighted_global_acc_tests)
        
        weighted_avg_global_pre_test = np.mean(weighted_global_pre_tests)
        weighted_std_global_pre_test = np.std(weighted_global_pre_tests)
        
        weighted_avg_global_rec_test = np.mean(weighted_global_rec_tests)
        weighted_std_global_rec_test = np.std(weighted_global_rec_tests)
        
        weighted_avg_global_f1s_test = np.mean(weighted_global_f1s_tests)
        weighted_std_global_f1s_test = np.std(weighted_global_f1s_tests)
            

        if avg_global_acc_test > best_avg_accuracy:
            best_avg_accuracy = avg_global_acc_test
            best_num_clusters_threshold = num_clusters
            
            # Save all best results
            best_results = {
                'num_clusters_threshold': num_clusters,
                'metrics': {
                    'accuracy': {'mean': avg_global_acc_test, 'std': std_global_acc_test},
                    'precision': {'mean': avg_global_pre_test, 'std': std_global_pre_test},
                    'recall': {'mean': avg_global_rec_test, 'std': std_global_rec_test},
                    'f1score': {'mean': avg_global_f1s_test, 'std': std_global_f1s_test},
                    'weighted_accuracy': {'mean': weighted_avg_global_acc_test, 'std': weighted_std_global_acc_test},
                    'weighted_precision': {'mean': weighted_avg_global_pre_test, 'std': weighted_std_global_pre_test},
                    'weighted_recall': {'mean': weighted_avg_global_rec_test, 'std': weighted_std_global_rec_test},
                    'weighted_f1score': {'mean': weighted_avg_global_f1s_test, 'std': weighted_std_global_f1s_test},
                    'metrics_per_client_per_seed': metrics_per_client_per_seed,
                    'model_parameters_per_seed': model_parameters_per_seed,
                },
                'training_time': {'mean': avg_training_time, 'std': std_training_time},
                'avg_commun_metrics_history': avg_commun_metrics_history
            }

        i+=1
        
        print("===================")
        print(f"Completed num_clusters 1: {num_clusters}")
        print("===================")
    
      
    best_num_clusters_threshold = best_results['num_clusters_threshold']
            
    # Compute means and standard deviations
    avg_global_acc_test = best_results["metrics"]["accuracy"]["mean"]
    std_global_acc_test = best_results["metrics"]["accuracy"]["std"]
    
    avg_global_pre_test = best_results["metrics"]["precision"]["mean"]
    std_global_pre_test = best_results["metrics"]["precision"]["std"]
    
    avg_global_rec_test = best_results["metrics"]["recall"]["mean"]
    std_global_rec_test = best_results["metrics"]["recall"]["std"]
    
    avg_global_f1s_test = best_results["metrics"]["f1score"]["mean"]
    std_global_f1s_test = best_results["metrics"]["f1score"]["std"]
    
    avg_training_time = best_results["training_time"]["mean"]
    std_training_time = best_results["training_time"]["std"]
    
    weighted_avg_global_acc_test = best_results["metrics"]["weighted_accuracy"]["mean"]
    weighted_std_global_acc_test = best_results["metrics"]["weighted_accuracy"]["std"]
    
    weighted_avg_global_pre_test = best_results["metrics"]["weighted_precision"]["mean"]
    weighted_std_global_pre_test = best_results["metrics"]["weighted_precision"]["std"]
    
    weighted_avg_global_rec_test = best_results["metrics"]["weighted_recall"]["mean"]
    weighted_std_global_rec_test = best_results["metrics"]["weighted_recall"]["std"]
    
    weighted_avg_global_f1s_test = best_results["metrics"]["weighted_f1score"]["mean"]
    weighted_std_global_f1s_test = best_results["metrics"]["weighted_f1score"]["std"]
    
    avg_commun_metrics_history = best_results["avg_commun_metrics_history"]
    
    # Print best results
    print("\n\nBEST RESULTS:")
    print(f'Best W Threshold: {best_num_clusters_threshold}')
    print(f'Global Accuracy: Mean = {avg_global_acc_test:.4f}, 'f'Std = {std_global_acc_test:.4f}')
    # print(f'Global Precision: Mean = {avg_global_pre_test:.4f}, 'f'Std = {std_global_pre_test:.4f}')
    # print(f'Global Recall: Mean = {avg_global_rec_test:.4f}, 'f'Std = {std_global_rec_test:.4f}')
    # print(f'Global F1-Score: Mean = {avg_global_f1s_test:.4f}, 'f'Std = {std_global_f1s_test:.4f}')
    print("************************************************************************************")
    # print(f'Weighted Accuracy: Mean = {weighted_avg_global_acc_test:.4f}, Std = {weighted_std_global_acc_test:.4f}')
    # print(f'Weighted Precision: Mean = {weighted_avg_global_pre_test:.4f}, Std = {weighted_std_global_pre_test:.4f}')
    # print(f'Weighted Recall: Mean = {weighted_avg_global_rec_test:.4f}, Std = {weighted_std_global_rec_test:.4f}')
    # print(f'Weighted F1-Score: Mean = {weighted_avg_global_f1s_test:.4f}, Std = {weighted_std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')


def calculate_psi_label_skew(clients_glob, epsilon=1e-6):
    """
    Calculate the Population Stability Index (PSI) for each client concerning global frequencies
    and the Weighted Population Stability Index (WPSI_L).

    Parameters:
    - clients_glob: Dictionary where each key is a client identifier, and the value is a list of tuples with label information.
    - epsilon: Small value added to observed and expected frequencies to avoid division by zero or log(0) issues.
    
    Returns:
    - psi_per_client: Dictionary with PSI values for each client.
    - psi_per_class: Dictionary with PSI values for each label in each client.
    - global_freq: Dictionary with global frequencies for each label across all clients.
    - group_tables: Dictionary with group tables for each client, containing label counts and relative frequencies.
    - wpsi_l: Weighted average of PSI values, weighted by the number of examples in each client.
    """

    # Initialize a dictionary to accumulate frequencies for all labels from all clients
    global_freq_accum = {}
    client_example_counts = {}  # To store the number of examples per client

    # First pass to calculate global frequencies
    for key, value in clients_glob.items():
        labels_check = [val[1] for val in value]
        client_example_counts[key] = len(labels_check)
        
        labels_check_df = pd.DataFrame(labels_check, columns=["label"])
        group = labels_check_df['label'].value_counts().to_dict()
        for label, count in group.items():
            global_freq_accum[label] = global_freq_accum.get(label, 0) + count

    # Calculate global frequencies
    total_count = sum(global_freq_accum.values())
    global_freq = {label: count / total_count for label, count in global_freq_accum.items()}
    all_labels = set(global_freq_accum.keys())

    # Initialize result dictionaries
    psi_per_client = {}
    psi_per_class = {}
    group_tables = {}

    # Second pass to calculate PSI values
    for key, value in clients_glob.items():
        labels_check = [val[1] for val in value]
        labels_check_df = pd.DataFrame(labels_check, columns=["label"]).reset_index()
        
        # Calculate client's label distribution
        group = labels_check_df.groupby(['label']).count().reset_index()
        group['particip'] = group['index'].values / len(labels_check)
        group = group.set_index('label').reindex(all_labels, fill_value=0).reset_index()
        
        group_tables[key] = group
        observed_freq = group.set_index('label')['particip'].to_dict()

        # Calculate PSI per class
        psi_per_class[key] = {}
        psi_total = 0
        
        for label in all_labels:
            observed = observed_freq.get(label, 0) + epsilon
            expected = global_freq.get(label, 0) + epsilon
            psi_contribution = (observed - expected) * np.log(observed / expected)
            
            psi_per_class[key][label] = psi_contribution
            psi_total += psi_contribution
        
        psi_per_client[key] = psi_total

    # Calculate weighted PSI
    total_examples = sum(client_example_counts.values())
    wpsi_l = sum(psi_per_client[client] * count for client, count in client_example_counts.items()) / total_examples

    return psi_per_client, psi_per_class, global_freq, group_tables, wpsi_l


def calculate_global_frequencies(group_tables):
    global_sum = defaultdict(int)
    
    for client, df in group_tables.items():
        # Group by label and sum indexes
        label_sums = df.groupby('label')['index'].sum()
        
        # Accumulate results
        for label, value in label_sums.items():
            # Convert boolean labels to 0/1 if needed
            key = label
            global_sum[key] += value
            
    return dict(global_sum)




def calculate_psi_label_skew_test(clients_glob, global_frequencies, epsilon=1e-6):
    """
    Calculate the Population Stability Index (PSI) for each client using precomputed global frequencies
    and the Weighted Population Stability Index (WPSI_L).

    Parameters:
    - clients_glob: Dictionary where each key is a client identifier, and the value is a list of tuples with label information
    - global_frequencies: Dictionary with precomputed global label frequencies (summed index counts)
    - epsilon: Small value to avoid division by zero
    
    Returns:
    - psi_per_client: Dictionary with PSI values for each client
    - psi_per_class: Dictionary with PSI values per label per client
    - global_freq: Normalized global frequency distribution
    - group_tables: Client label distributions
    - wpsi_l: Weighted average PSI
    """
    # Calculate normalized global frequencies
    total_global = sum(global_frequencies.values())
    global_freq = {k: v/total_global for k, v in global_frequencies.items()}
    all_labels = set(global_frequencies.keys())

    # Initialize containers
    psi_per_client = {}
    psi_per_class = {}
    group_tables = {}
    client_example_counts = {}

    # Single pass processing
    for client, data in clients_glob.items():
        # Extract labels and count examples
        labels = [item[1] for item in data]
        client_example_counts[client] = len(labels)
        
        # Create client distribution
        label_df = pd.DataFrame(labels, columns=['label'])
        group = label_df.groupby('label').size().reset_index(name='index')
        group['particip'] = group['index'] / len(labels)
        
        # Ensure all labels are represented
        group = group.set_index('label').reindex(all_labels, fill_value=0).reset_index()
        group_tables[client] = group
        
        # Calculate PSI components
        psi_per_class[client] = {}
        client_psi = 0
        
        for _, row in group.iterrows():
            label = row['label']
            observed = row['particip'] + epsilon
            expected = global_freq.get(label, 0) + epsilon
            
            psi_contribution = (observed - expected) * np.log(observed/expected)
            psi_per_class[client][label] = psi_contribution
            client_psi += psi_contribution
            
        psi_per_client[client] = client_psi

    # Calculate weighted average PSI
    total_examples = sum(client_example_counts.values())
    wpsi_l = sum(psi * client_example_counts[client] 
                for client, psi in psi_per_client.items()) / total_examples

    return psi_per_client, psi_per_class, global_freq, group_tables, wpsi_l

    
def train_clust_psi_pfl(clients_glob, clients_glob_test, local_nodes_glob, random_state, seeds, epochs, comms_round):
    print("Calculating PSI and clustering clients...")

   
    # Calculate PSI for clients
    psi_per_client_l, psi_per_client_per_class_l,global_freq_l, group_tables_l, wpsi_l = calculate_psi_label_skew(clients_glob)
    print("PSI per client:", psi_per_client_l)
    print("Global frequencies:", global_freq_l)
    print("Weighted PSI (WPSI_L):", wpsi_l)

    # Claculate global frequencies
    global_frequencies = calculate_global_frequencies(group_tables_l)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(clients_dict=clients_glob)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_test, list_y_test = from_FedArtML_to_Flower_format(clients_dict=clients_glob_test)
    
    # Extract per-class PSI values into numerical array
    psi_class_array = np.array([list(client.values()) for client in psi_per_client_per_class_l.values()])
    
    # Reshape total PSI values to column vector
    psi_total_array = np.array(list(psi_per_client_l.values())).reshape(-1, 1)
    
    # Get number of examples per client
    num_examples_per_client = [len(client) for client in list_y_train]
    
    # Combine into feature matrix
    psi_data = np.hstack((psi_total_array, psi_class_array))

    # Create scaler instance
    scaler = StandardScaler()
    # Fit and transform data
    psi_data = scaler.fit_transform(psi_data)

    
    silhouette_scores = []
    best_k = 1  # Start with 1 cluster as default
    max_score = -1
    k_range = range(1, local_nodes_glob)
    # k_range = range(1, 3)
    
    for k in k_range:
        if k == 1:
            score = 0  # Silhouette undefined for k=1, set baseline
            kmeans = 0
            clusters_km = np.array([0] * local_nodes_glob)
        else:
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            clusters_km = kmeans.fit_predict(psi_data)
            
            # Check if clustering actually produced multiple clusters
            unique_clusters_km = np.unique(clusters_km)
            if len(unique_clusters_km) == 1:
                score = 0  # Treat single-cluster results as invalid
            else:
                score = silhouette_score(psi_data, clusters_km)
        
        silhouette_scores.append(score)
        if score > max_score:
            best_kmeans = kmeans
            best_clusters_km = clusters_km
            max_score, best_k = score, k
            best_clusters_km = clusters_km
    
    # Handle case where all scores are 0 (true single cluster)
    if max_score <= 0:
        best_k = 1
    
    print(f"Number of recommended clusters: {best_k}")
    print(f"Max silhouette score: {max_score:.2f}")
    
    # Initialize cluster indices dictionary
    cluster_indices = {cluster_id: [] for cluster_id in range(best_k)}
    
    # Group client indices by their cluster assignment
    for client_idx, cluster_id in enumerate(best_clusters_km):
        cluster_indices[cluster_id].append(client_idx)
    
    # Create clustered data partitions
    clustered_x_train = [
        np.concatenate([list_x_train[i] for i in client_indices], axis=0)
        for cluster_id, client_indices in cluster_indices.items()
    ]
    
    clustered_y_train = [
        np.concatenate([list_y_train[i] for i in client_indices], axis=0)
        for cluster_id, client_indices in cluster_indices.items()
    ]

    # Calculate PSI per test client
    psi_per_client_l_test, psi_per_client_per_class_l_test, global_freq_l_test, group_tables_l_test, wpsi_l_test = calculate_psi_label_skew_test(
        clients_glob=clients_glob_test,
        global_frequencies=global_frequencies
    )
    
    print("PSI per client test:", psi_per_client_l_test)
    print("Global frequencies test:", global_freq_l_test)
    print("Weighted PSI (WPSI_L) test:", wpsi_l_test)
    
    # Reshape total PSI values to column vector
    psi_total_array_test = np.array(list(psi_per_client_l_test.values())).reshape(-1, 1)
    
    # Extract per-class PSI values into numerical array
    psi_class_array_test = np.array([list(client.values()) for client in psi_per_client_per_class_l_test.values()])
    
    # Get number of examples per client in test
    num_examples_per_client_test = [len(client) for client in list_y_test]
    
    # Combine into feature matrix tets
    psi_data_test = np.hstack((psi_total_array_test, psi_class_array_test))
    # psi_data_test = psi_class_array_test
    
    psi_data_test = scaler.transform(psi_data_test)
    
    best_clusters_km_test = best_kmeans.predict(psi_data)
    
    # Initialize cluster indices dictionary
    cluster_indices_test = {cluster_id: [] for cluster_id in range(best_k)}
    
    # Group client indices by their cluster assignment
    for client_idx, cluster_id in enumerate(best_clusters_km_test):
        cluster_indices_test[cluster_id].append(client_idx)
    
    clustered_x_test = [
        np.concatenate([list_x_test[i] for i in client_indices], axis=0)
        for cluster_id, client_indices in cluster_indices_test.items()
    ]
    
    clustered_y_test = [
        np.concatenate([list_y_test[i] for i in client_indices], axis=0)
        for cluster_id, client_indices in cluster_indices_test.items()
    ]
    
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_train, list_y_train = from_FedArtML_to_Flower_format(clients_dict=clients_glob)
    
    # Convert from SplitAsFederatedData function output (FedArtML) to Flower (list) format
    list_x_test, list_y_test = from_FedArtML_to_Flower_format(clients_dict=clients_glob_test)
    
    model_parameters_per_seed = []    
   
    # Define function to pass to each local node (client)
    def client_fn(context: Context) -> fl.client.Client:
        # Define model
        if dataset_used == "acs_income" or dataset_used == "dutch":
            model = LRModel(len(covariates), len(np.unique(y_all)))
        elif dataset_used == "sent140":
            model = DNN(MAX_SEQUENCE_LENGTH, len(np.unique(y_all)), N_HIDDEN, embedding_matrix)
    
        # Set optimizer
        optimizer = Adam(learning_rate=learn_rate)
    
        # Compile model
        model.compile(optimizer=optimizer, loss=loss_inic, metrics=metrics)
    
        # Load train data partition of each client ID (cid)
        x_train_cid = np.array(list_x_train_selected[int(context.node_config['partition-id'])], dtype=float)
        y_train_cid = np.array(list_y_train_selected[int(context.node_config['partition-id'])], dtype=float)
        val, count = np.unique(y_train_cid, return_counts=True)
        print("XXXXXXXX Val:",val,",Count:",count)

        # Load test data partition of each client ID (cid)
        x_test_cid = np.array(list_x_test[int(context.node_config['partition-id'])], dtype=float)
        y_test_cid = np.array(list_y_test[int(context.node_config['partition-id'])], dtype=float)

        # Create and return client
        return FlowerClient(model, x_train_cid, y_train_cid, x_test_cid, y_test_cid, epochs, context.node_config['partition-id'])

    # Initialize variables to store the best results
    best_psi_threshold = None
    best_avg_accuracy = -1  # Initialize to a very low value
    best_results = None

    # Initialize storage for metrics and training times
    global_acc_tests, global_pre_tests, global_rec_tests, global_f1s_tests = [], [], [], []
    weighted_global_acc_tests, weighted_global_pre_tests, weighted_global_rec_tests, weighted_global_f1s_tests = [], [], [], []
    training_times = []
    all_commun_metrics_histories = []  # For saving history of all communication rounds
    commun_metrics_histories = []
    metrics_per_client_per_seed = []

    # Initialize dataframe to store results for this psi_threshold
    cluster_results = pd.DataFrame(columns=[
        'agg_method_used', 'WPSI_L', 'cluster', 'seed', 'cluster_num_samples', 'cluster_num_clients', 'training_time', 
        'global_acc_test', 'global_pre_test', 'global_rec_test', 'global_f1s_test',
        'weighted_global_acc_test', 'weighted_global_pre_test', 'weighted_global_rec_test', 'weighted_global_f1s_test'
    ])
    
    # Train model for each seed
    for seed in tqdm(seeds,desc=f"Looping on seeds\n"):
        print(f"\nRunning simulation with seed {seed}\n")
        np.random.seed(seed)
        tf.random.set_seed(seed)    

        # Initialize storage for metrics and training times
        global_acc_tests_per_clust, global_pre_tests_per_clust, global_rec_tests_per_clust, global_f1s_tests_per_clust = [], [], [], []
        weighted_global_acc_tests_per_clust, weighted_global_pre_tests_per_clust, weighted_global_rec_tests_per_clust, weighted_global_f1s_tests_per_clust = [], [], [], []
        training_times_per_clust = []
        all_commun_metrics_histories_per_clust = []  # For saving history of all communication rounds

        metrics_per_client_per_seed_per_clust = []
        model_parameters_per_seed_per_clust = [] 

        test_sizes = []
    
        # Loop through each PSI threshold
        for cluster_id in tqdm(range(best_k),desc=f"Looping  on clusters for seed {seed}"):
            print(f"Training Cluster: {cluster_id}")
            
            # Get client indices for this cluster
            client_indices = [i for i, c_id in enumerate(best_clusters_km) if c_id == cluster_id]
            
            # Select train cluster data
            list_x_train_selected = [list_x_train[i] for i in client_indices]
            list_y_train_selected = [list_y_train[i] for i in client_indices]

            # Get test client indices for this cluster
            client_indices_test = [i for i, c_id in enumerate(best_clusters_km_test) if c_id == cluster_id]
            
            # Select test cluster data
            list_x_test_selected = [list_x_test[i] for i in client_indices_test]
            list_y_test_selected = [list_y_test[i] for i in client_indices_test]
            
            # Debugging output
            print(f"Filtered x_train: {len(list_x_train_selected)} clients")
            print(f"Filtered y_train: {len(list_y_train_selected)} clients")
            print(f"Filtered x_test: {len(list_x_test_selected)} clients")
            print(f"Filtered y_test: {len(list_y_test_selected)} clients")
            n_clients_in_cluster = len(list_x_train_selected)
            
            # Replicate single-client clusters to 4 copies
            if len(list_x_train_selected) <= 2:
                list_x_train_selected *= 4  #s Creates 4 identical copies
                list_y_train_selected *= 4
                print(f"Replicated single client to {len(list_x_train_selected)} clients")

            
            # Create centralized X_test_clust and Y_test _clust to evaluate model
            global X_test_clust, Y_test_clust            
            X_test_clust = np.concatenate(list_x_test_selected, axis=0)
            Y_test_clust = np.concatenate(
                [y.astype(bool) for y in list_y_test_selected],  # Explicit type conversion
                axis=0
            )
    
            print(f"X_test_clust: {X_test_clust.shape}")
            print(f"Y_test_clust: {Y_test_clust.shape}")
    
            # Create Federated strategy
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=0.5,
                # min_available_clients=1,
                # min_fit_clients=1,
                fraction_evaluate=0.5,
                evaluate_fn=evaluate_LR_CL_clust
            )

            # Start training simulation
            start_time = time.time()
            commun_metrics_history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=len(list_x_train_selected),
                config=fl.server.ServerConfig(num_rounds=comms_round),
                strategy=strategy,
                ray_init_args={"num_cpus": CPUs_to_use},
            )
            training_time_per_clust = time.time() - start_time
            
            # Append communication metrics history
            all_commun_metrics_histories.append(commun_metrics_history)

            # Retrieve metrics
            global_acc_test_per_clust = retrieve_global_metrics(commun_metrics_history, "centralized", "accuracy", False)
            global_pre_test_per_clust = retrieve_global_metrics(commun_metrics_history, "centralized", "precision", False)
            global_rec_test_per_clust = retrieve_global_metrics(commun_metrics_history, "centralized", "recall", False)
            global_f1s_test_per_clust = retrieve_global_metrics(commun_metrics_history, "centralized", "f1score", False)
            
            # Calculate metrics per client (to do weighted average)
            metrics_weighted_per_clust = evaluate_LR_FL(global_params_retrieved, list_x_test_selected, list_y_test_selected)
            metrics_per_client_per_clust = metrics_weighted_per_clust["client_metrics"]
            print("metrics_per_client_per_clust:",metrics_per_client_per_clust)
            # Change names of clients
            metrics_per_client_per_clust = {
                cluster_indices[cluster_id][i]: value for i, (key, value) in enumerate(metrics_per_client_per_clust.items())
            }            

            print("\n\n\n metrics_per_client_per_clust:",metrics_per_client_per_clust)
            
            weighted_averages_per_clust = metrics_weighted_per_clust["weighted_averages"]
            
            # Retrieve metrics
            weighted_global_acc_test_per_clust = weighted_averages_per_clust['accuracy']
            weighted_global_pre_test_per_clust = weighted_averages_per_clust['precision']
            weighted_global_rec_test_per_clust = weighted_averages_per_clust['recall']
            weighted_global_f1s_test_per_clust = weighted_averages_per_clust['f1score']

            # Store results
            global_acc_tests_per_clust.append(global_acc_test_per_clust)
            global_pre_tests_per_clust.append(global_pre_test_per_clust)
            global_rec_tests_per_clust.append(global_rec_test_per_clust)
            global_f1s_tests_per_clust.append(global_f1s_test_per_clust)
            weighted_global_acc_tests_per_clust.append(weighted_global_acc_test_per_clust)
            weighted_global_pre_tests_per_clust.append(weighted_global_pre_test_per_clust)
            weighted_global_rec_tests_per_clust.append(weighted_global_rec_test_per_clust)
            weighted_global_f1s_tests_per_clust.append(weighted_global_f1s_test_per_clust)   
            training_times_per_clust.append(training_time_per_clust)

            metrics_per_client_per_seed_per_clust.append(dict(metrics_per_client_per_clust))
            model_parameters_per_seed_per_clust.append(global_params_retrieved)

            # Collect test set sizes for each cluster
            test_sizes.append(X_test_clust.shape[0])

            new_row = pd.DataFrame([{
                'agg_method_used': agg_method_used,
                'WPSI_L':wpsi_l,
                'cluster': cluster_id,
                'seed': seed,
                'cluster_num_samples': X_test_clust.shape[0],
                'cluster_num_clients': n_clients_in_cluster,
                'training_time': training_time_per_clust,
                'global_acc_test': global_acc_test_per_clust,
                'global_pre_test': global_pre_test_per_clust,
                'global_rec_test': global_rec_test_per_clust,
                'global_f1s_test': global_f1s_test_per_clust,
                'weighted_global_acc_test': weighted_global_acc_test_per_clust,
                'weighted_global_pre_test': weighted_global_pre_test_per_clust,
                'weighted_global_rec_test': weighted_global_rec_test_per_clust,
                'weighted_global_f1s_test': weighted_global_f1s_test_per_clust                
            }])
            
            cluster_results = pd.concat([cluster_results, new_row], ignore_index=True)        

        new_dict = {client_id: metric
                    for cluster in metrics_per_client_per_seed_per_clust
                    for client_id, metric in cluster.items()}

        metrics_per_client_per_seed.append(new_dict)
        
        # Build extracted metrics
        extracted_metrics_history = {
            metric: [h.metrics_centralized.get(metric, []) 
                    for h in all_commun_metrics_histories] 
            for metric in ["accuracy", "f1score", "precision", "recall"]
        }
        
        # Weighted metric calculation
        def calculate_weighted_metrics(extracted_metrics, test_sizes):
            total_weight = sum(test_sizes)
            return {
                metric: [(clusters[0][rid][0],  # Preserve original round number
                        sum(c[rid][1]*w for c,w in zip(clusters,test_sizes))/total_weight)
                        for rid in range(len(clusters[0]))]
                for metric, clusters in extracted_metrics.items()
            }
        
        commun_metrics_history = calculate_weighted_metrics(extracted_metrics_history, test_sizes)            
            
        # Calculate total test size across all clusters
        total_test_size = sum(test_sizes)
        
        # Global metrics (average of per-cluster centralized metrics)
        global_acc_test = np.average(global_acc_tests_per_clust, weights=test_sizes)
        global_pre_test = np.average(global_pre_tests_per_clust, weights=test_sizes)
        global_rec_test = np.average(global_rec_tests_per_clust, weights=test_sizes)
        global_f1s_test = np.average(global_f1s_tests_per_clust, weights=test_sizes)
        
        # Weighted global metrics (average of per-cluster client-weighted metrics)
        weighted_global_acc_test = np.average(weighted_global_acc_tests_per_clust, weights=test_sizes)
        weighted_global_pre_test = np.average(weighted_global_pre_tests_per_clust, weights=test_sizes)
        weighted_global_rec_test = np.average(weighted_global_rec_tests_per_clust, weights=test_sizes)
        weighted_global_f1s_test = np.average(weighted_global_f1s_tests_per_clust, weights=test_sizes)
        training_time = sum(training_times_per_clust)

        
        # Append resul of seeds
        global_acc_tests.append(global_acc_test)
        global_pre_tests.append(global_pre_test)
        global_rec_tests.append(global_rec_test)
        global_f1s_tests.append(global_f1s_test)

        weighted_global_acc_tests.append(weighted_global_acc_test)
        weighted_global_pre_tests.append(weighted_global_pre_test)
        weighted_global_rec_tests.append(weighted_global_rec_test)
        weighted_global_f1s_tests.append(weighted_global_f1s_test)
        training_times.append(training_time)
        commun_metrics_histories.append(commun_metrics_history)        
        
        
    # Compute means and standard deviations
    avg_global_acc_test = np.mean(global_acc_tests)
    std_global_acc_test = np.std(global_acc_tests)
    
    avg_global_pre_test = np.mean(global_pre_tests)
    std_global_pre_test = np.std(global_pre_tests)
    
    avg_global_rec_test = np.mean(global_rec_tests)
    std_global_rec_test = np.std(global_rec_tests)
    
    avg_global_f1s_test = np.mean(global_f1s_tests)
    std_global_f1s_test = np.std(global_f1s_tests)
    
    avg_training_time = np.mean(training_times)
    std_training_time = np.std(training_times)

    # Compute means and standard deviations for weight values
    weighted_avg_global_acc_test = np.mean(weighted_global_acc_tests)
    weighted_std_global_acc_test = np.std(weighted_global_acc_tests)
    
    weighted_avg_global_pre_test = np.mean(weighted_global_pre_tests)
    weighted_std_global_pre_test = np.std(weighted_global_pre_tests)
    
    weighted_avg_global_rec_test = np.mean(weighted_global_rec_tests)
    weighted_std_global_rec_test = np.std(weighted_global_rec_tests)
    
    weighted_avg_global_f1s_test = np.mean(weighted_global_f1s_tests)
    weighted_std_global_f1s_test = np.std(weighted_global_f1s_tests)

    # Average metrics history across seeds
    def calculate_averaged_metrics(commun_metrics_histories):
        """Average metrics across multiple communication histories."""
        avg_commun_metrics_history = {
            "metrics_centralized": {
                metric: [
                    (round_num, np.mean([h[metric][i][1] for h in commun_metrics_histories]))
                    for i, (round_num, _) in enumerate(commun_metrics_histories[0][metric])
                ]
                for metric in ["accuracy", "precision", "recall", "f1score"]
            }
        }
        return avg_commun_metrics_history
    
    # Usage
    avg_commun_metrics_history = calculate_averaged_metrics(commun_metrics_histories)

    # Save all best results
    best_results = {
        'metrics': {
            'accuracy': {'mean': avg_global_acc_test, 'std': std_global_acc_test},
            'precision': {'mean': avg_global_pre_test, 'std': std_global_pre_test},
            'recall': {'mean': avg_global_rec_test, 'std': std_global_rec_test},
            'f1score': {'mean': avg_global_f1s_test, 'std': std_global_f1s_test},
            'weighted_accuracy': {'mean': weighted_avg_global_acc_test, 'std': weighted_std_global_acc_test},
            'weighted_precision': {'mean': weighted_avg_global_pre_test, 'std': weighted_std_global_pre_test},
            'weighted_recall': {'mean': weighted_avg_global_rec_test, 'std': weighted_std_global_rec_test},
            'weighted_f1score': {'mean': weighted_avg_global_f1s_test, 'std': weighted_std_global_f1s_test},
            'metrics_per_client_per_seed': metrics_per_client_per_seed,
            'model_parameters_per_seed': model_parameters_per_seed,
        },
        'training_time': {'mean': avg_training_time, 'std': std_training_time},
        'avg_commun_metrics_history': avg_commun_metrics_history,
        
    }
    
    
    # Compute means and standard deviations
    avg_global_acc_test = best_results["metrics"]["accuracy"]["mean"]
    std_global_acc_test = best_results["metrics"]["accuracy"]["std"]
    
    avg_global_pre_test = best_results["metrics"]["precision"]["mean"]
    std_global_pre_test = best_results["metrics"]["precision"]["std"]
    
    avg_global_rec_test = best_results["metrics"]["recall"]["mean"]
    std_global_rec_test = best_results["metrics"]["recall"]["std"]
    
    avg_global_f1s_test = best_results["metrics"]["f1score"]["mean"]
    std_global_f1s_test = best_results["metrics"]["f1score"]["std"]
    
    avg_training_time = best_results["training_time"]["mean"]
    std_training_time = best_results["training_time"]["std"]
    
    weighted_avg_global_acc_test = best_results["metrics"]["weighted_accuracy"]["mean"]
    weighted_std_global_acc_test = best_results["metrics"]["weighted_accuracy"]["std"]
    
    weighted_avg_global_pre_test = best_results["metrics"]["weighted_precision"]["mean"]
    weighted_std_global_pre_test = best_results["metrics"]["weighted_precision"]["std"]
    
    weighted_avg_global_rec_test = best_results["metrics"]["weighted_recall"]["mean"]
    weighted_std_global_rec_test = best_results["metrics"]["weighted_recall"]["std"]
    
    weighted_avg_global_f1s_test = best_results["metrics"]["weighted_f1score"]["mean"]
    weighted_std_global_f1s_test = best_results["metrics"]["weighted_f1score"]["std"]
    
    avg_commun_metrics_history = best_results["avg_commun_metrics_history"]
    
    # Print best results
    print("\n\nBEST RESULTS:")
    
    print(f'Global Accuracy: Mean = {avg_global_acc_test:.4f}, 'f'Std = {std_global_acc_test:.4f}')
    print(f'Global Precision: Mean = {avg_global_pre_test:.4f}, 'f'Std = {std_global_pre_test:.4f}')
    print(f'Global Recall: Mean = {avg_global_rec_test:.4f}, 'f'Std = {std_global_rec_test:.4f}')
    print(f'Global F1-Score: Mean = {avg_global_f1s_test:.4f}, 'f'Std = {std_global_f1s_test:.4f}')
    print("************************************************************************************")
    # print(f'Weighted Accuracy: Mean = {weighted_avg_global_acc_test:.4f}, Std = {weighted_std_global_acc_test:.4f}')
    # print(f'Weighted Precision: Mean = {weighted_avg_global_pre_test:.4f}, Std = {weighted_std_global_pre_test:.4f}')
    # print(f'Weighted Recall: Mean = {weighted_avg_global_rec_test:.4f}, Std = {weighted_std_global_rec_test:.4f}')
    # print(f'Weighted F1-Score: Mean = {weighted_avg_global_f1s_test:.4f}, Std = {weighted_std_global_f1s_test:.4f}')
    print(f'Training Time: Mean = {avg_training_time:.2f} seconds, Std = {std_training_time:.2f} seconds')
    










def train_aggregation_methods(clients_glob,clients_glob_test,aggregation_method,local_nodes_glob,random_state, seeds, epochs, comms_round,psi_thresholds):
    if aggregation_method == "psi_pfl":
        train_psi_pfl(clients_glob,clients_glob_test,local_nodes_glob,random_state, seeds, epochs,comms_round,psi_thresholds)
    if aggregation_method == "fedavg":
        train_fedavg(clients_glob,clients_glob_test,local_nodes_glob,random_state, seeds, epochs,comms_round)
    if aggregation_method == "fedprox":
        train_fedprox(clients_glob,clients_glob_test,local_nodes_glob,random_state, seeds, epochs,comms_round)
    if aggregation_method == "fedavgm":
        train_fedavgm(clients_glob,clients_glob_test,local_nodes_glob,random_state, seeds, epochs,comms_round)        
    if aggregation_method == "fedadagrad":
        train_fedadagrad(clients_glob,clients_glob_test,local_nodes_glob,random_state, seeds, epochs,comms_round)
    if aggregation_method == "fedyogi":
        train_fedyogi(clients_glob,clients_glob_test,local_nodes_glob,random_state, seeds, epochs,comms_round)
    if aggregation_method == "fedadam":
        train_fedadam(clients_glob,clients_glob_test,local_nodes_glob,random_state, seeds, epochs,comms_round)
    if aggregation_method == "poc":
        train_poc(clients_glob,clients_glob_test,local_nodes_glob,random_state, seeds, epochs,comms_round)
    if aggregation_method == "haccs":
        train_haccs(clients_glob,clients_glob_test,local_nodes_glob,random_state, seeds, epochs,comms_round)        
    if aggregation_method == "fedcls":
        train_fedcls(clients_glob,clients_glob_test,local_nodes_glob,random_state, seeds, epochs,comms_round)                
    if aggregation_method == "cfl":
        train_cfl(clients_glob,clients_glob_test,local_nodes_glob,random_state, seeds, epochs,comms_round)
    if aggregation_method == "fedsoft":
        train_fedsoft(clients_glob,clients_glob_test,local_nodes_glob,random_state, seeds, epochs,comms_round)        
    elif aggregation_method == "clust_psi_pfl":
        train_clust_psi_pfl(clients_glob,clients_glob_test,local_nodes_glob,random_state, seeds, epochs,comms_round)
    


def split_clients_data(curr_clients_glob, test_size=0.2,curr_random_state=42):
    "Returns the random data split (train and test federated, train and test centralized)"    
    # Step 1 & 2: Split each client into train and test sets
    train_clients = {}
    test_clients = {}

    for client, data in curr_clients_glob.items():
        features, labels = zip(*data)  # Separate features and labels
        features = np.array(features)
        labels = np.array(labels)

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=curr_random_state)

        # Store back into respective dictionaries
        train_clients[client] = [(X_train[i], y_train[i]) for i in range(len(X_train))]
        test_clients[client] = [(X_test[i], y_test[i]) for i in range(len(X_test))]

    # Step 3: Merge all training data into two numpy arrays
    train_features = []
    train_labels = []
    for data in train_clients.values():
        for features, label in data:
            train_features.append(features)
            train_labels.append(label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    # Step 4: Merge all test data into two numpy arrays
    test_features = []
    test_labels = []
    for data in test_clients.values():
        for features, label in data:
            test_features.append(features)
            test_labels.append(label)

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    return train_clients, test_clients, train_features, train_labels, test_features, test_labels

def split_dataset_by_percent(train_images, train_labels, similarity_value, num_clients, prefix_cli='Client', random_state=42):
    "Returns the data split for the similarity method and the distances generated"
    # Combine indices and labels for shuffling
    train_indices = list(range(len(train_labels)))

    np.random.seed(random_state)
    random.seed(random_state)
    # Shuffle the data
    random.shuffle(train_indices)

    # Split indices into IID and Non-IID portions
    len_train_iid = round(similarity_value * len(train_indices))

    train_iid_indices = train_indices[:len_train_iid]
    train_niid_indices = train_indices[len_train_iid:]

    # Sort Non-IID indices by labels to simulate Non-IID splits
    train_niid_indices.sort(key=lambda idx: train_labels[idx])

    # Calculate splits for each user
    delta_train_iid = len(train_iid_indices) // num_clients
    delta_train_niid = len(train_niid_indices) // num_clients

    dataset_split = {}
    pctg_distr = []

    for user in range(num_clients):
        # Allocate portions of IID and Non-IID data to each user
        train_user_indices = train_iid_indices[user * delta_train_iid:(user + 1) * delta_train_iid] + \
                             train_niid_indices[user * delta_train_niid:(user + 1) * delta_train_niid]

        # Store the split in NumPy format
        train_user_images = train_images[train_user_indices]
        train_user_labels = train_labels[train_user_indices]

        # Combine train_user_images and train_user_labels as requested
        combined_data = [(image, label) for image, label in zip(train_user_images, train_user_labels)]

        dataset_split[f"{prefix_cli}_{user + 1}"] = combined_data

        # Calculate label distributions for metrics
        df_aux = pd.DataFrame(train_user_labels, columns=['label']).label.value_counts().reset_index()
        df_aux.columns = ['index', 'label']
        df_node = pd.DataFrame(np.unique(train_labels), columns=['index'])
        df_node = df_node.merge(df_aux, how='left', left_on='index', right_on='index').replace(np.nan, 0)
        df_node['perc'] = df_node.label / sum(df_node.label)

        pctg_distr.append(list(df_node.perc))
    print(pctg_distr)
    # Calculate distances
    JS_dist = jensen_shannon_distance(pctg_distr)
    H_dist = hellinger_distance(pctg_distr)
    # H_dist = np.nan
    emd_dist = earth_movers_distance(pctg_distr)

    distances = {'without_class_completion': {
        'jensen-shannon': JS_dist,
        'hellinger': H_dist,
        'earth-movers': emd_dist
    }}

    return {'without_class_completion': dataset_split}, distances

def create_federated_data(X_all, y_all,partiton_protocol_lab,local_nodes_glob,non_iid_parameter,random_state):
    """Returns federated data (partitioned into clients)"""
    print("\nData partition initiated...")
    if partiton_protocol_lab=="dirichlet":
        # Instantiate a SplitAsFederatedData object
        my_federater = SplitAsFederatedData(random_state = random_state)
        
        # Get federated dataset from centralized dataset
        clients_glob_dic, list_ids_sampled_dic, miss_class_per_node, distances = my_federater.create_clients(image_list = X_all, label_list = y_all,
                                                                     num_clients = local_nodes_glob, prefix_cli='Client',
                                                                 method = "dirichlet", alpha = non_iid_parameter)
        
        clients_glob = clients_glob_dic['without_class_completion']

        print(f"The partition protocol used to split the data was {partiton_protocol_lab} with alpha={non_iid_parameter} and {local_nodes_glob} clients.")
        JSD_glob = distances['without_class_completion']['jensen-shannon']
        HD_glob = distances['without_class_completion']['hellinger']
        print("Hellinger distance:", HD_glob)
        EMD_glob = distances['without_class_completion']['earth-movers']
    elif partiton_protocol_lab == "similarity":
        clients_glob_dic, distances = split_dataset_by_percent(X_all, y_all, similarity_value=non_iid_parameter, num_clients=local_nodes_glob, random_state=random_state)
        clients_glob = clients_glob_dic['without_class_completion']

        print(f"The partition protocol used to split the data was {partiton_protocol_lab} with S={non_iid_parameter} and {local_nodes_glob} clients.")

        JSD_glob = distances['without_class_completion']['jensen-shannon']
        HD_glob = distances['without_class_completion']['hellinger']
        print("Hellinger distance:", HD_glob)
        EMD_glob = distances['without_class_completion']['earth-movers']
    else:
        print(f"{partiton_protocol_lab} method not implemented")

    return clients_glob



def load_preprocess_acs_income():
    global covariates
    # Define covariates (attributes) names
    covariates = ['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P']        
    
    # List of all U.S. states and Puerto Rico abbreviations
    all_states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
                  "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
                  "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
                  "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
                  "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "PR"]
    
    # Initialize empty lists to store features and labels for each state
    X_all = []
    y_all = []
    
    # Iterate over each state
    for state in all_states:
        data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey='person', root_dir=get_default_path() + "/data/acs_income")
        acs_data = data_source.get_data(states=[state], download=True)
        X, y, _ = ACSIncome.df_to_numpy(acs_data)
        
        print(f"{state} - X shape: {X.shape}, Y shape: {y.shape}")
        
        # Append data for posterior concatenation
        X_all.append(X)
        y_all.append(y)
    
    # Concatenate all data along the first axis
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Print final shapes after concatenation
    print("Total X (features) shape for centralized data:", X_all.shape)
    print("Total Y (label) shape for centralized data:", y_all.shape)    

    # Define scaler
    scaler = RobustScaler()
    # Transform data
    transformer = scaler.fit(X_all)
    X_all = transformer.transform(X_all)

    return X_all, y_all



def load_preprocess_dutch():
    global covariates

    selected_data = pd.read_csv(get_default_path() + '/data/dutch/dutch.csv')
    
    # Separate covariates
    X_all = selected_data[selected_data.columns[selected_data.columns!='occupation']]
    
    # Define covariates names
    covariates = X_all.columns
    
    # Encode sex
    label_encoder = LabelEncoder()
    X_all['sex']= label_encoder.fit_transform(X_all['sex']) 
    
    # Keep numpy arrays
    X_all = X_all.values
    y_all = selected_data['occupation'].values

    # Print final shapes after concatenation
    print("Total X (features) shape for centralized data:", X_all.shape)
    print("Total Y (label) shape for centralized data:", y_all.shape)    

    # Define scaler
    scaler = RobustScaler()
    # Transform data
    transformer = scaler.fit(X_all)
    X_all = transformer.transform(X_all)

    return X_all, y_all



def load_preprocess_sent140():

    global MAX_SEQUENCE_LENGTH, N_HIDDEN, embedding_matrix
    
    # Load GloVe embeddings (unchanged)
    GLOVE_EMB = get_default_path() + '/data/sent140/glove_6B/glove.6B.300d.txt'
    EMBEDDING_DIM = 300
    # EMBEDDING_DIM = 20
    
    # Data splitting and tokenization
    MAX_NB_WORDS = 100000
    # MAX_NB_WORDS = 500
    MAX_SEQUENCE_LENGTH = 25  # Reduced sequence length for faster processing
    
    # Define batch size
    BATCH_SIZE = 2048 * 2
    BATCH_SIZE = 2048 // 2
    BATCH_SIZE = 256
    N_HIDDEN = 10
    # Load and preprocess data
    df = pd.read_csv(get_default_path() + "/data/sent140/training.1600000.processed.noemoticon.csv", encoding='latin', header=None)

    # Extract the random sample
    df = df.sample(frac=0.2, random_state=random_state)

    df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
    df = df.drop(['id', 'date', 'query', 'user_id'], axis=1)
    lab_to_sentiment = {0: "Negative", 4: "Positive"}
    df.sentiment = df.sentiment.apply(lambda x: lab_to_sentiment[x])
    
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    
    def preprocess(text, stem=False):
        text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
        tokens = [stemmer.stem(token) if stem else token for token in text.split() if token not in stop_words]
        return " ".join(tokens)
    
    df.text = df.text.apply(lambda x: preprocess(x))
    
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(df.text)
    
    X_all = pad_sequences(tokenizer.texts_to_sequences(df.text), maxlen=MAX_SEQUENCE_LENGTH)
    
    encoder = LabelEncoder()
    y_all = encoder.fit_transform(df.sentiment.to_list()).reshape(-1, )
    
    embeddings_index = {}
    with open(GLOVE_EMB, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word, coefs = values[0], np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return X_all, y_all





def load_preprocess_celeba():

    class ImageDataset:
        def __init__(self, img_dir, attr_file, transform=None):
            self.img_dir = img_dir
            self.transform = transform
    
            # Read the attributes file using pandas
            df = pd.read_csv(attr_file, delim_whitespace=True, skiprows=1)
    
            # Extract image IDs and smile labels
            self.image_ids = df.index.values  # First column contains image IDs
            self.smile_labels = df['Smiling'].astype('float32').values  # Column named 'Smiling'
    
            # Convert -1 labels to 0 for binary classification
            self.smile_labels[self.smile_labels == -1] = 0
    
        def __len__(self):
            return len(self.image_ids)
    
        def __getitem__(self, idx):
            img_name = os.path.join(self.img_dir, f"{self.image_ids[idx]}")
            image = Image.open(img_name).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            label = self.smile_labels[idx]
            return np.array(image), label

    # Define transformations (resize images to 32x32 and normalize)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Paths to images and attribute file
    img_dir = get_default_path() + "/data/celeba/img_align_celeba"
    attr_file = get_default_path() + "/data/celeba/list_attr_celeba.txt"

    # Create Dataset
    dataset = ImageDataset(img_dir=img_dir, attr_file=attr_file, transform=transform)

    # Initialize numpy arrays for images and labels
    num_samples = len(dataset)
    image_shape = (32, 32, 3)  # Height x Width x Channels (channel-last format)

    # Create a DataLoader for efficient data loading
    batch_size = 64  # Adjust batch size based on your system's memory
    num_workers = 4  # Number of parallel workers (adjust based on your CPU)

    # Initialize DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize numpy arrays for images and labels
    X_all = np.zeros((len(dataset), *image_shape), dtype=np.float32)
    y_all = np.zeros(len(dataset), dtype=np.float32)

    # Populate numpy arrays using the DataLoader
    for i, (images, labels) in enumerate(dataloader):
        start_idx = i * batch_size
        end_idx = start_idx + images.size(0)
        
        # Convert images to NumPy and rearrange dimensions from (B, C, H, W) to (B, H, W, C)
        X_all[start_idx:end_idx] = images.permute(0, 2, 3, 1).numpy()
        y_all[start_idx:end_idx] = labels.numpy()
    
    # Print final shapes after concatenation
    print("Total X shape:", X_all.shape)
    print("Total Y shape:", y_all.shape)    

    return X_all, y_all


def load_preprocess_data(dataset):
    """Returns centralized data preprocessed"""
    print("\nLoad and preprocess centralized data initiated...")    

    if dataset == "acs_income":    
        X_all, y_all = load_preprocess_acs_income()
    if dataset == "dutch":    
        X_all, y_all = load_preprocess_dutch()
    if dataset == "sent140":    
        X_all, y_all = load_preprocess_sent140()
    if dataset == "celeba":    
        X_all, y_all = load_preprocess_celeba()                
    return X_all, y_all



def get_default_path():
    """Returns the directory where the script is located"""
    return os.path.dirname(os.path.abspath(sys.argv[0]))
    
def main():
    # Set up argument parser_main
    parser_main = argparse.ArgumentParser(description='Execute aggregation methods.')
    
    # Add arguments
    parser_main.add_argument('--root-path', type=str, default=get_default_path(), help='The root path for the application')
    parser_main.add_argument('--dataset', type=str, default='acs_income', help='Dataset to use for training')
    parser_main.add_argument('--partitioner', type=str, default='dirichlet', help='Partition protocol to split (federate) data')
    parser_main.add_argument('--non-iid-param', type=float, default=0.7, help='Non-IID parameter (alpha for Dirichlet) to split data')
    parser_main.add_argument('--num-clients', type=int, default=10, help='Number of clients')
    parser_main.add_argument('--rand-state', type=int, default=42, help='Random state for reproducibility')
    parser_main.add_argument('--agg-method', type=str, default="psi_pfl", help='Aggregation algorithm to train')
    parser_main.add_argument('--seeds-list', type=str, default="0,1,2,3,42", help='List of random seeds to use for training (comma separated)')
    parser_main.add_argument('--local-epochs', type=int, default=2, help='Number of local epochs (per client)')
    parser_main.add_argument('--comm-rounds', type=int, default=10, help='Number of communication rounds in FL training')   
    parser_main.add_argument('--psi-ths-list', type=str, default="10,25,50", help='List of percentiles for psi thresholds (tau) (comma separated)')
    parser_main.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser_main.add_argument('--mu-ths-list', type=str, default="0.01", help='List of mu thresholds for fedprox (comma separated)')
    parser_main.add_argument('--momentum-ths-list', type=str, default="0.7", help='List of momentum thresholds for fedavgm (comma separated)')
    parser_main.add_argument('--tau-ths-list', type=str, default="0.1", help='List of tau thresholds for fedadagrad, fedyogi and fedadam (comma separated)')
    parser_main.add_argument('--eta-ths-list', type=str, default="0.3162", help='List of eta thresholds for fedadagrad, fedyogi and fedadam (comma separated)')
    parser_main.add_argument('--eta-l-ths-list', type=str, default="1", help='List of eta_l thresholds for fedadagrad, fedyogi and fedadam (comma separated)')
    parser_main.add_argument('--beta-1-ths-list', type=str, default="0.9", help='List of beta1 thresholds for fedyogi and fedadam (comma separated)')
    parser_main.add_argument('--beta-2-ths-list', type=str, default="0.99", help='List of beta2 thresholds for fedyogi and fedadam (comma separated)')    
    parser_main.add_argument('--d-poc', type=int, default=10, help='Number of clients to select (hyperparameter) for power-of-choice (poc)')
    parser_main.add_argument('--rho-haccs', type=float, default=0.95, help='Rho threshold (for haccs)')    
    parser_main.add_argument('--thl-fedcls', type=float, default=0.1, help='Similarity threshold (for fedcls)')    
    parser_main.add_argument('--eps-1-ths-list', type=str, default="0.00001", help='List of epsilon1 thresholds for cfl (comma separated)')        
    parser_main.add_argument('--eps-2-ths-list', type=str, default="0.1", help='List of epsilon2 thresholds for cfl (comma separated)')            
    parser_main.add_argument('--gamma-max-ths-list', type=str, default="0.5", help='List of gamma max thresholds for cfl (comma separated)')            
    parser_main.add_argument('--n-clust-ths-list', type=str, default="5", help='List of number of clusterss thresholds for fedsoft (comma separated)')
    
    # Parse arguments
    args_main = parser_main.parse_args()
    
    global dataset_used,y_all, random_state, loss_inic, metrics, agg_method_used, X_test, Y_test, learn_rate, mu_thresholds, server_momentum_thresholds, tau_thresholds, eta_range, eta_l_range, beta_1_range, beta_2_range, d_poc, rho_haccs, thl_fedcls, eps_1_range, eps_2_range, gam_max_range, n_clust_range

    # Define random state for all the simulation
    random_state = args_main.rand_state
    
    # Define loss
    loss_inic = SparseCategoricalCrossentropy()
    
    # Define metric to check
    metrics = [SparseCategoricalAccuracy()] 

    # Define learning rate
    learn_rate = args_main.lr
    
    # Define datased method employed
    dataset_used = args_main.dataset

    # Define aggregation method employed
    agg_method_used = args_main.agg_method

    # Load centralized data
    X_all, y_all = load_preprocess_data(dataset=args_main.dataset)

    # Partition (federate) data
    clients_glob = create_federated_data(X_all, y_all, args_main.partitioner,args_main.num_clients,args_main.non_iid_param,args_main.rand_state)

    # Obtain train (federated and centralized) and test (federated and centralized) data
    clients_glob, clients_glob_test, X_train, Y_train, X_test, Y_test = split_clients_data(curr_clients_glob=clients_glob,curr_random_state=args_main.rand_state)
    
    # Print shapes for verification
    print("Train (centralized) features shape:", X_train.shape)
    print("Train (centralized) labels shape:", Y_train.shape)
    print("Test (centralized) features shape:", X_test.shape)
    print("Test (centralized) labels shape:", Y_test.shape)

    # Get list from text of seeds comma separated
    seeds = args_main.seeds_list.split(",")
    seeds = [int(seed) for seed in seeds]

    # Get list from text of psi thresholds (tau)
    psi_thresholds = args_main.psi_ths_list.split(",")
    psi_thresholds = [float(th) for th in psi_thresholds]

    # Get list from text of mu thresholds (for fedprox)    
    mu_thresholds = args_main.mu_ths_list.split(",")
    mu_thresholds = [float(th) for th in mu_thresholds]

    # Get list from text of momentum thresholds (for fedavgm)    
    server_momentum_thresholds = args_main.momentum_ths_list.split(",")
    server_momentum_thresholds = [float(th) for th in server_momentum_thresholds]

    # Get list from text of hyperparameters thresholds (for fedadagrad, fedyogi and fedadam)
    tau_thresholds = args_main.tau_ths_list.split(",")
    tau_thresholds = [float(th) for th in tau_thresholds]
    eta_range = args_main.eta_ths_list.split(",")
    eta_range = [float(th) for th in eta_range]
    eta_l_range = args_main.eta_l_ths_list.split(",")
    eta_l_range = [float(th) for th in eta_l_range]
    beta_1_range = args_main.beta_1_ths_list.split(",")
    beta_1_range = [float(th) for th in beta_1_range]
    beta_2_range = args_main.beta_2_ths_list.split(",")
    beta_2_range = [float(th) for th in beta_2_range]

    # Get number of clients to select (for power-of-choice (poc))
    d_poc = args_main.d_poc

    # Get rho threshold (for haccs)    
    rho_haccs = args_main.rho_haccs
    
    # Get similarity threshold (for fedcls)
    thl_fedcls = args_main.thl_fedcls

    # Get list from text of hyperparameters thresholds (for cfl)    
    eps_1_range = args_main.eps_1_ths_list.split(",")
    eps_1_range = [float(th) for th in eps_1_range]
    eps_2_range = args_main.eps_2_ths_list.split(",")
    eps_2_range = [float(th) for th in eps_2_range]
    gam_max_range = args_main.gamma_max_ths_list.split(",")
    gam_max_range = [float(th) for th in gam_max_range]

    # Get list from text of hyperparameters thresholds (for fedsoft)    
    n_clust_range = args_main.n_clust_ths_list.split(",")
    n_clust_range = [int(th) for th in n_clust_range]

    # Train aggregation method selected
    train_aggregation_methods(clients_glob, clients_glob_test, agg_method_used, args_main.num_clients,args_main.rand_state, seeds, args_main.local_epochs, args_main.comm_rounds,psi_thresholds)

    print("Model training finished.")

if __name__ == "__main__":
    main()
