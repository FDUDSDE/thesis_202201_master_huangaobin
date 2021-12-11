import torch.utils.data as Data
import numpy as np
import torch
from Sample import *


class Instances_data(Data.Dataset):
    def __init__(self):
        self.instances_pair_list = np.load("..//data//DBLP//instances//instances_pair_list.npy", allow_pickle=True)
        self.instances_edge_index_list = np.load("..//data//DBLP//instances//instances_pair_edge_index_list.npy",
                                                 allow_pickle=True)
        self.instances_fea_list = np.load("..//data//DBLP//instances//instances_pair_features_matrix_list.npy",
                                          allow_pickle=True)
        self.instances_pair_edge_num_list = np.load("..//data//DBLP//instances//instances_pair_edge_num_list.npy",
                                                    allow_pickle=True)
        self.instances_pair_node_num_list = np.load("..//data//DBLP//instances//instances_pair_node_num_list.npy",
                                                    allow_pickle=True)

        self.instances_pair_edge_index_tensor_list = []
        self.instances_pair_fea_tensor_list = []
        for x in range(len(self.instances_pair_list)):
            temp_list_edge = []
            temp_list_fea = []
            for y in range(2):
                temp_list_edge.append(torch.tensor(self.instances_edge_index_list[x][y], dtype=torch.long))
                temp_list_fea.append(torch.FloatTensor(self.instances_fea_list[x][y]))
            self.instances_pair_edge_index_tensor_list.append(temp_list_edge)
            self.instances_pair_fea_tensor_list.append(temp_list_fea)

    def __getitem__(self, idx):
        assert idx < len(self.instances_pair_list)
        return self.instances_pair_edge_index_tensor_list[idx], self.instances_pair_fea_tensor_list[idx], \
               self.instances_pair_edge_num_list[idx], self.instances_pair_node_num_list[idx]

    def __len__(self):
        return len(self.instances_pair_list)


# test_loader = Data.DataLoader(dataset=Instances_data(), batch_size=5, shuffle=True)
# for step, batch in enumerate(test_loader):
#     print(len(batch))
#     print(len(batch[0]))
#     print(len(batch[0][0]))
#     print(len(batch[0][0][0]))
#     print(batch[0][0][4][: int(batch[2][4][0])+1])
#     print(batch[1][0][4][: int(batch[3][4][0])+1])
#     print("_____________________________")
