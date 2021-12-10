from Component import *
from Sample import *
import time


class GPT_GIN2(torch.nn.Module):
    def __init__(self, gnn_input_dim, gnn_hidden_dim, gnn_output_dim, device):
        super(GPT_GIN2, self).__init__()

        self.device = device
        self.gnn = GNN(gnn_input_dim, gnn_hidden_dim, gnn_output_dim).to(device)

        self.key_encoder = GNN(gnn_input_dim, gnn_hidden_dim, gnn_output_dim).to(device)
        self.temperature = 0.05

        self.classifier_gnn = GNN(gnn_input_dim, gnn_hidden_dim, gnn_output_dim).to(device)
        self.classifier = Node_Classifier(gnn_output_dim, cat_dim=1).to(device)

        self.auto_regressive_gnn = GNN(gnn_input_dim, gnn_hidden_dim, gnn_output_dim).to(device)
        self.c_signal = torch.nn.Parameter(torch.Tensor(gnn_input_dim)).to(device)
        self.b_signal = torch.nn.Parameter(torch.Tensor(gnn_input_dim)).to(device)
        self.auto_regressive = Auto_Regressive(gnn_output_dim).to(device)

    def forward(self, x, edge_index_t):
        return self.gnn.forward(x, edge_index_t)

    def fine_tune_node_classification_per_epoch(self, pair_wise_list, all_graph_features_list,
                                                all_graph_edge_index_list, optimizer,
                                                batch_size=32):
        losses_for_epoch = 0

        temp_list = pair_wise_list.copy()
        random.shuffle(temp_list)
        iteration_num = int(len(pair_wise_list) / batch_size)
        for iteration in range(iteration_num):

            y = self.classifier.forward(
                self.classifier_gnn.forward(torch.FloatTensor(all_graph_features_list).to(self.device),
                                            torch.tensor(all_graph_edge_index_list,
                                                         dtype=torch.long).t().to(self.device)))

            losses_for_batch = torch.tensor([0], dtype=torch.float).to(self.device)

            for pair_node in temp_list[iteration * batch_size: (iteration + 1) * batch_size]:
                tensor_1 = y[pair_node[0]]
                tensor_2 = y[pair_node[1]]

                output_tensor = torch.nn.Sigmoid().forward(tensor_1 - tensor_2).to(self.device)
                target_tensor = torch.FloatTensor([1]).to(self.device)
                losses_for_batch += torch.nn.BCELoss().forward(output_tensor, target_tensor).to(self.device)

            losses_for_epoch += float(losses_for_batch)
            print("batch_loss: ", losses_for_batch)
            optimizer.zero_grad()
            losses_for_batch.backward()
            optimizer.step()

        return losses_for_epoch

    def fine_tune_auto_regressive_per_epoch(self, all_graph_features_list, all_graph_edge_index_list, optimizer,
                                            partition_solution_list, candidate_list, target_node_list, batch_size=32):

        self.train()

        losses_for_epoch = 0
        idx_list = [i for i in range(len(partition_solution_list))]
        random.shuffle(idx_list)

        iteration_num = int(len(partition_solution_list) / batch_size)
        edge_index_tensor = torch.tensor(all_graph_edge_index_list, dtype=torch.long).t().to(self.device)
        for iteration in range(iteration_num):

            losses_for_batch = torch.tensor([0], dtype=torch.float).to(self.device)

            for idx in idx_list[iteration * batch_size: (iteration + 1) * batch_size]:

                x = torch.FloatTensor(all_graph_features_list).to(self.device)
                for node_in in partition_solution_list[idx]:
                    x[node_in] += self.c_signal
                for node_out in candidate_list[idx]:
                    x[node_out] += self.b_signal

                y = self.auto_regressive_gnn(x, edge_index_tensor)

                candidate_tensor_list = []
                for node_can in candidate_list[idx]:
                    candidate_tensor_list.append(y[node_can])

                partition_tensor_list = []
                for node_pas in partition_solution_list[idx]:
                    partition_tensor_list.append(y[node_pas])

                _, result = self.auto_regressive.forward(candidate_tensor_list, partition_tensor_list)

                if target_node_list[idx] == -1:
                    target_idx = len(candidate_list[idx])
                else:
                    target_idx = candidate_list[idx].index(target_node_list[idx])

                temp_list = []
                for node_re in result:
                    temp_list.append(node_re.squeeze())
                temp_tensor = torch.stack(temp_list).unsqueeze(dim=0)
                target_tensor = torch.LongTensor([target_idx]).to(self.device)

                losses_for_step = torch.nn.CrossEntropyLoss().forward(temp_tensor, target_tensor)
                # print(losses_for_step)
                losses_for_batch += losses_for_step

            losses_for_batch /= batch_size
            losses_for_epoch += float(losses_for_batch)
            if iteration % 2 == 0:
                print("batch_loss: ", losses_for_batch)
            optimizer.zero_grad()
            losses_for_batch.backward()
            optimizer.step()

        losses_for_epoch /= iteration_num
        return losses_for_epoch
