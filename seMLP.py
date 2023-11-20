import torch.nn
from GeneticAlgorithm import GeneticAlgorithm
from MLP import MLP
import numpy as np
from tqdm import tqdm


class seMLP:
    def __init__(self, train_loader, test_loader, in_features, train_epoch=100, lr=0.001, ga_path="", population=30, winning_percentage=0.1, layer_range=(2, 6),
                 node_range=(32, 255), mutation_rate=0.3, store_path="./GA", save_freq=3, weight_decay=0.001):
        if ga_path == "":
            self.GA = GeneticAlgorithm(population, winning_percentage, layer_range, node_range,
                                       mutation_rate, store_path, save_freq)
        else:
            self.GA = GeneticAlgorithm.load(ga_path)
        self.train_epoch = train_epoch
        self.lr = lr
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.in_features = in_features
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.weight_decay = weight_decay

    def train_one_epoch(self, model, epoch, criterion, optimizer):
        model.train()
        # print(f'Epoch: {epoch + 1}')
        losses = 0.0

        for batch_index, batch in enumerate(self.train_loader):
            # x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
            # output = model(x_batch)
            # loss = criterion(output, y_batch)
            # losses += loss.item()
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            data, target = batch[0].to(self.device), batch[1].to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses += loss
        # print(f"Epoch {epoch}, Loss: {losses:.3f}\n")

        #     if batch_index % 100 == 99:  # print every 100 batches
        #         avg_loss_across_batches = running_loss / 100
        #         print(f'Batch {batch_index+1}, Loss: {avg_loss_across_batches:.3f}\n')
        #         running_loss = 0.0

    def validate_one_epoch(self, model, criterion):
        """

        :param mlp:
        :param criterion:
        :return: the RMSE of the testing samples
        """
        model.eval()
        losses = 0.0
        # correct = 0
        for batch_index, batch in enumerate(self.test_loader):
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)
            with torch.no_grad():
                output = model(x_batch)
                loss = criterion(output, y_batch)
                losses += loss.item()

        return np.sqrt(losses)

    def train_one_model(self, in_features, out_features, chromosome):
        mlp = MLP(in_features, out_features, chromosome)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(self.train_epoch):
            self.train_one_epoch(mlp, epoch, criterion, optimizer)
            test_rmse = self.validate_one_epoch(mlp, criterion)
            # print(f"Test RMSE: {test_rmse}")
            # print("*" * 20 + "\n")
        return mlp

    def forward(self, in_features, out_features):
        for i in range(self.GA.population):
            chromosome = self.GA.chromosomes[i]
            mlp = MLP(in_features, out_features, chromosome.get_layers())
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(mlp.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            test_rmse = 0
            for epoch in range(self.train_epoch):
                self.train_one_epoch(mlp, epoch, criterion, optimizer)
                test_rmse = self.validate_one_epoch(mlp, criterion)
                # print(f"Test RMSE: {test_rmse}")
                # print("*"*20+"\n")

            chromosome.update_RMSE(test_rmse)
        self.GA.step()

    def run(self, ga_epoch, curr_ga_epoch=0):
        for i in tqdm(range(curr_ga_epoch, ga_epoch)):
            self.forward(self.in_features, 1)
        self.GA.save_history()


