from Layer import *
import torch.nn.functional as F
import math
import torch


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, drop_out_rate=0.5):
        super(GNN, self).__init__()

        self.conv1 = GINLayer(input_dim=input_dim, hidden_dim=hidden_dim)
        self.conv2 = GINLayer(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.conv3 = GINLayer(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.output_linear = Linear(int(hidden_dim * 3), output_dim)

        self.drop_out_rate = drop_out_rate

    def forward(self, x, edge_index_t):
        x = x.clone()
        x1 = self.conv1.forward(x, edge_index_t)
        x2 = self.conv2.forward(x1, edge_index_t)
        x3 = self.conv3.forward(x2, edge_index_t)
        output = F.dropout(self.output_linear(torch.cat((x1, x2, x3), 1)), p=self.drop_out_rate, training=self.training)
        return output


class Node_Classifier(torch.nn.Module):
    def __init__(self, gnn_output_dim, cat_dim=1):
        super(Node_Classifier, self).__init__()

        hidden_dim = int(math.sqrt(gnn_output_dim))
        self.linear1 = Linear(gnn_output_dim, hidden_dim)
        self.batch_norm1 = BatchNorm1d(hidden_dim)
        self.relu1 = ReLU()
        self.linear2 = Linear(hidden_dim, cat_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear2(self.relu1(self.batch_norm1(self.linear1.forward(x)))))


class Auto_Regressive(torch.nn.Module):
    def __init__(self, gnn_output_dim):
        super(Auto_Regressive, self).__init__()

        h_d = int(gnn_output_dim / 2)
        self.l1 = Linear(gnn_output_dim, h_d)
        self.r1 = ReLU()
        self.b1 = BatchNorm1d(h_d)

        self.lp = Linear(gnn_output_dim, h_d)
        self.rp = ReLU()
        self.bp = BatchNorm1d(h_d)

        self.l2 = Linear(h_d, 1)

    def forward(self, c_list, p_list):
        # x1 = torch.stack(x_list)
        # stop_attention = F.softmax(self.attention_linear.forward(x1), dim=0)
        # product = stop_attention * x1
        # stop_tensor = product.sum(dim=0)
        # x_list.append(stop_tensor)
        # x2 = torch.stack(x_list)
        # pro_list = F.softmax(self.linear.forward(x2), dim=0)
        # x3 = self.linear.forward(x2)

        ps = torch.stack(p_list)
        ps = self.rp(self.lp(ps)).mean(dim=0)

        cs = torch.stack(c_list)
        cs = self.r1(self.l1(cs))
        cs = torch.cat((cs, ps.unsqueeze(dim=0)), dim=0)
        pro_list = F.softmax(self.l2.forward(cs), dim=0)
        x3 = self.l2.forward(cs)
        return pro_list, x3


class Auto_Regressive2(torch.nn.Module):
    def __init__(self, gnn_output_dim):
        super(Auto_Regressive2, self).__init__()

        self.attention_linear = Linear(gnn_output_dim, 1)
        self.linear = Linear(gnn_output_dim, 1)
