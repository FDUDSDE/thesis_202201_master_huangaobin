from Model import *
from Data import *


def pre_train(model, epochs=10, batch_size=128, lr=0.01):
    model.train()

    instances_pair_list = np.load("..//data//DBLP//instances//instances_pair_list.npy", allow_pickle=True)
    instances_pair_fea_list = np.load("..//data//DBLP//instances//instances_pair_features_matrix_list.npy",
                                      allow_pickle=True)
    instances_pair_edge_index_list = np.load("..//data//DBLP//instances//instances_pair_edge_index_list.npy",
                                             allow_pickle=True)
    instances_pair_edge_num_list = np.load("..//data//DBLP//instances//instances_pair_edge_num_list.npy",
                                           allow_pickle=True)
    instances_pair_node_num_list = np.load("..//data//DBLP//instances//instances_pair_node_num_list.npy",
                                           allow_pickle=True)

    model.instance_moco_queue_initialization(instance_pair_list=instances_pair_list,
                                             instance_pair_fea_list=instances_pair_fea_list,
                                             instance_pair_edge_index_list=instances_pair_edge_index_list,
                                             instance_pair_edge_num_list=instances_pair_edge_num_list,
                                             instance_pair_node_num_list=instances_pair_node_num_list)

    print("pre_train start")
    losses_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_loader = Data.DataLoader(dataset=Instances_data(), batch_size=batch_size, shuffle=True, num_workers=0,
                                  pin_memory=True)
    for epoch in range(1, epochs):
        losses_per_epoch = model.pre_train_instance_discrimination_per_epoch(instances_dataloader=data_loader,
                                                                             optimizer=optimizer,
                                                                             batch_size=batch_size)
        print("---------------")
        losses_list.append(float(losses_per_epoch))
        print("epoch:", epoch, "loss:", losses_per_epoch)
        torch.save(model.state_dict(), "..//save//pre_trained_model_" + "in_epoch:" + str(epoch) + ".npy")
        print("---------------")
    np.save("..//save//pre_train_loss.npy", losses_list)
    torch.save(model.state_dict(), "..//save//pre_trained_model.npy")
    print("pre_train finish")


def fine_tune(model, epochs=10, batch_size=32, lr=0.01):
    model.train()
    model.classifier_gnn.load_state_dict(model.gnn.state_dict())
    model.auto_regressive_gnn.load_state_dict(model.gnn.state_dict())

    all_communities_list = np.load("..//data//DBLP//community//communities_list.npy", allow_pickle=True)
    all_communities_boundary_list = np.load("..//data//DBLP//community//communities_boundaries_list.npy",
                                            allow_pickle=True)
    sample_idx_list = random.sample([idx for idx in range(len(all_communities_list))], 500)
    communities_list = []
    communities_boundary_list = []
    for idx in sample_idx_list:
        communities_list.append(all_communities_list[idx])
        communities_boundary_list.append(all_communities_boundary_list[idx])

    all_graph_fea_list = np.load("..//data//DBLP//all_graph_fea_list.npy", allow_pickle=True)
    all_graph_edge_index_list = np.load("..//data//DBLP//all_graph_edge_index_list.npy", allow_pickle=True)
    node_adj_dict = np.load("..//data//DBLP//node_adj_dict.npy", allow_pickle=True).item()

    pair_wise_sorted_list = pair_wise_process(communities_list, communities_boundary_list)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("fine_tune1 start")
    losses_list_for_classifier = []
    losses_list_for_auto_regressive = []
    for epoch in range(epochs):
        losses_per_epoch = model.fine_tune_node_classification_per_epoch(pair_wise_list=pair_wise_sorted_list,
                                                                         all_graph_features_list=all_graph_fea_list,
                                                                         all_graph_edge_index_list=all_graph_edge_index_list,
                                                                         optimizer=optimizer, batch_size=batch_size)
        print("---------------")
        losses_list_for_classifier.append(float(losses_per_epoch))
        print("epoch:", epoch, "loss:", losses_per_epoch)
        print("---------------")

    np.save("..//save//fine_tune_classifier_loss.npy", losses_list_for_classifier)
    torch.save(model.state_dict(), "..//save//fine_tune_after_classification_model.npy")
    print("fine_tune1 finish")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("fine_tune2 start")
    for epoch in range(epochs):
        losses_per_epoch = model.fine_tune_auto_regressive_per_epoch(communities_list=communities_list,
                                                                     node_adj_dict=node_adj_dict,
                                                                     all_graph_features_list=all_graph_fea_list,
                                                                     all_graph_edge_index_list=all_graph_edge_index_list,
                                                                     optimizer=optimizer, batch_size=batch_size)
        print("---------------")
        losses_list_for_auto_regressive.append(float(losses_per_epoch))
        print("epoch:", epoch, "loss:", losses_per_epoch)
        print("---------------")

    np.save("..//save//fine_tune_auto_regressive_loss.npy", losses_list_for_auto_regressive)
    torch.save(model.state_dict(), "..//save//fine_tune_after_auto_regressive_model.npy")
    print("fine_tune2 finish")
