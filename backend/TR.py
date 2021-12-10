from Model import *
from Train import *
import time
import numpy as np
import collections


def fine_tune_tr(model, dataset, graph_name, train_size=500, epochs=10, batch_size=32, lr=0.01, pairs=1024, ada=1000,
                 cat="NC"):
    #   群组数据读取
    all_communities_list = np.load(dataset + "/communities_list.npy", allow_pickle=True)
    all_communities_boundary_list = np.load(dataset + "/communities_boundaries_list.npy", allow_pickle=True)

    #   训练数据采样装填
    sample_idx_list = [i for i in range(len(all_communities_list))]
    sample_idx_list = random.sample(sample_idx_list, train_size)

    communities_list = []
    communities_boundary_list = []
    for idx in sample_idx_list:
        communities_list.append(all_communities_list[idx])
        communities_boundary_list.append(all_communities_boundary_list[idx])

    #   图数据读取
    all_graph_fea_list = np.load(dataset + "/all_graph_fea_list.npy", allow_pickle=True)
    all_graph_edge_index_list = np.load(dataset + "/all_graph_edge_index_list.npy", allow_pickle=True)
    node_adj_dict = np.load(dataset + "/node_adj_dict.npy", allow_pickle=True).item()

    #   种子选择器训练
    if cat == "NC":

        pair_wise_list_p = pair_wise_process(communities_list, communities_boundary_list)
        optimizer = torch.optim.Adam([{'params': model.classifier_gnn.parameters(), 'lr': lr * 0.2},
                                      {'params': model.classifier.parameters(), 'lr': lr}], lr=lr)

        print("ss_train start")
        for epoch in range(epochs):
            if pairs < len(pair_wise_list_p):
                pair_wise_list = random.sample(pair_wise_list_p, pairs)
            else:
                pair_wise_list = pair_wise_list_p

            losses_per_epoch = model.fine_tune_node_classification_per_epoch(pair_wise_list=pair_wise_list,
                                                                             all_graph_features_list=all_graph_fea_list,
                                                                             all_graph_edge_index_list=all_graph_edge_index_list,
                                                                             optimizer=optimizer, batch_size=batch_size)

            print("---------------")
            print("epoch:", epoch, "loss:", losses_per_epoch)
            print("---------------")

        model_name = '{}_ss{}.npy'.format(graph_name, str(random.randint(1, 10000)))
        torch.save(model.state_dict(), "./models/seed_selectors/models/{}".format(model_name))
        print("ss_train finish")
        return model_name.split('.')[0]

    #   社区生成器训练
    if cat == "AR":

        dagger_queue_partition_solution = []
        dagger_queue_candidate = []
        dagger_queue_target_node = []
        dagger_size = ada
        dagger_pointer = 0

        print("cg_train start")

        for epoch in range(epochs):
            time_s = time.time()
            if epoch < 5:
                optimizer = torch.optim.Adam([{'params': model.c_signal, 'lr': lr},
                                              {'params': model.b_signal, 'lr': lr},
                                              {'params': model.auto_regressive.parameters(), 'lr': lr}], lr=lr)
            else:
                optimizer = torch.optim.Adam([{'params': model.auto_regressive_gnn.parameters(), 'lr': lr * 0.1},
                                              {'params': model.c_signal, 'lr': lr},
                                              {'params': model.b_signal, 'lr': lr},
                                              {'params': model.auto_regressive.parameters(), 'lr': lr}], lr=lr)

            temp_partition_solution_list, temp_candidate_list, temp_target_node_list = AR_data_generate(
                communities_list=communities_list, node_adj_dict=node_adj_dict)

            partition_solution_list = temp_partition_solution_list + dagger_queue_partition_solution
            candidate_list = temp_candidate_list + dagger_queue_candidate
            target_node_list = temp_target_node_list + dagger_queue_target_node

            losses_per_epoch = model.fine_tune_auto_regressive_per_epoch(all_graph_features_list=all_graph_fea_list,
                                                                         all_graph_edge_index_list=all_graph_edge_index_list,
                                                                         optimizer=optimizer,
                                                                         partition_solution_list=partition_solution_list,
                                                                         candidate_list=candidate_list,
                                                                         target_node_list=target_node_list,
                                                                         batch_size=batch_size)

            print("---------------")
            print("num_of_data:", len(partition_solution_list))
            print("epoch:", epoch, "loss:", losses_per_epoch)

            if epoch % 10 == 0 and epoch > 0:
                print("@@@daggerring@@@")
                rec_partition_solution_list, rec_candidate_list, rec_target_node_list = dagger(model,
                                                                                               communities_list,
                                                                                               node_adj_dict,
                                                                                               all_graph_fea_list,
                                                                                               all_graph_edge_index_list)

                for idx in range(len(rec_partition_solution_list)):
                    if len(rec_candidate_list[idx]) > 0:
                        if len(dagger_queue_partition_solution) < dagger_size:
                            dagger_queue_partition_solution.append(rec_partition_solution_list[idx].copy())
                            dagger_queue_candidate.append(rec_candidate_list[idx].copy())
                            dagger_queue_target_node.append(rec_target_node_list[idx])

                        else:
                            dagger_queue_partition_solution[dagger_pointer] = rec_partition_solution_list[idx].copy()
                            dagger_queue_candidate[dagger_pointer] = rec_candidate_list[idx].copy()
                            dagger_queue_target_node[dagger_pointer] = rec_target_node_list[idx]

                        dagger_pointer += 1
                        dagger_pointer = dagger_pointer % dagger_size

            time_e = time.time()
            print("time cost:", time_e - time_s)
            print("---------------")

        model_name = '{}_ex{}.npy'.format(graph_name, str(random.randint(1, 10000)))
        torch.save(model.state_dict(), "./models/community_generators/models/{}".format(model_name))
        print("cg_train finish")
        return model_name.split('.')[0]


device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


def model_train_request(dataset, graph_name, comms, train_size=500, epoch_ss=5, batch_size_ss=256, lr_ss=0.01,
                        epoch_cg=5, batch_size_cg=8, lr_cg=0.01, ada=1000, pairs=1024):
    this_model_ss = GPT_GIN2(gnn_input_dim=64, gnn_hidden_dim=64, gnn_output_dim=64, device=device)
    model_name = fine_tune_tr(model=this_model_ss, dataset=dataset, graph_name=graph_name, train_size=train_size,
                              epochs=epoch_ss, batch_size=batch_size_ss, lr=lr_ss, pairs=pairs, cat="NC")

    ss_detail_dict = np.load('./models/seed_selectors/model_details_dict.npy', allow_pickle=True).item()
    ss_detail_dict[model_name] = {"create_time": time.asctime(time.localtime(time.time())), "train_graph": graph_name,
                                  "train_comms": comms, "train_size": train_size, "pairs": pairs,
                                  "train_epoch": epoch_ss, "batch_size": batch_size_ss, "learning_rate": lr_ss}
    np.save('./models/seed_selectors/model_details_dict.npy', ss_detail_dict)

    this_model_cg = GPT_GIN2(gnn_input_dim=64, gnn_hidden_dim=64, gnn_output_dim=64, device=device)
    model_name = fine_tune_tr(model=this_model_cg, dataset=dataset, graph_name=graph_name, train_size=train_size,
                              epochs=epoch_cg, batch_size=batch_size_cg, lr=lr_cg, ada=ada, cat="AR")

    cg_detail_dict = np.load('./models/community_generators/model_details_dict.npy', allow_pickle=True).item()
    cg_detail_dict[model_name] = {"create_time": time.asctime(time.localtime(time.time())), "train_graph": graph_name,
                                  "train_comms": comms, "train_size": train_size, "ada": ada,
                                  "train_epoch": epoch_cg, "batch_size": batch_size_cg, "learning_rate": lr_cg}
    np.save('./models/community_generators/model_details_dict.npy', cg_detail_dict)

    print("model train finish")
