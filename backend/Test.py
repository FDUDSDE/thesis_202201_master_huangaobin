import torch
from Sample import *
import numpy as np
from Model import *
import time


def node_classification_test(model, dataset):
    all_graph_fea_list = np.load(dataset + "./all_graph_fea_list.npy", allow_pickle=True)
    all_graph_edge_index_list = np.load(dataset + "./all_graph_edge_index_list.npy", allow_pickle=True)

    y = model.classifier.forward(
        model.classifier_gnn.forward(torch.FloatTensor(all_graph_fea_list),
                                     torch.tensor(all_graph_edge_index_list, dtype=torch.long).t())).detach()

    sorted_list = sorted(enumerate(y), key=lambda x: x[1], reverse=True)
    sorted_node_list = []

    for node, _ in sorted_list:
        sorted_node_list.append(node)

    return sorted_node_list


def community_generate_test(model, sorted_node_list, dataset, budget=100):
    node_adj_dict = np.load(dataset + "/node_adj_dict.npy", allow_pickle=True).item()

    all_graph_fea_list = np.load(dataset + "./all_graph_fea_list.npy", allow_pickle=True)
    all_graph_edge_index_list = np.load(dataset + "./all_graph_edge_index_list.npy", allow_pickle=True)

    generated_communities_list = []
    point = 0
    # print(sorted_node_list)

    count = 0
    # print(sorted_node_list)
    # print(node_adj_dict)
    while count < budget:
        if len(node_adj_dict[sorted_node_list[point]]) >= 1:
            print("generating......")
            seed_node = sorted_node_list[point]
            target_node = seed_node
            temp_list = [seed_node]
            delete_node_list = []
            while target_node != -1:
                x_tensor = torch.FloatTensor(all_graph_fea_list)
                for node in temp_list:
                    x_tensor[node] += model.c_signal

                candidate0 = boundary_sample(node_adj_dict, temp_list)
                for node in candidate0:
                    x_tensor[node] += model.b_signal

                y = model.auto_regressive_gnn(x_tensor, torch.tensor(all_graph_edge_index_list, dtype=torch.long).t())

                candidate = []
                for node in candidate0:
                    if node not in delete_node_list:
                        candidate.append(node)
                if len(candidate) == 0:
                    target_node = -1
                    continue

                candidate_tensor_list = []
                for node in candidate:
                    candidate_tensor_list.append(y[node])

                partition_tensor_list = []
                for node_pas in temp_list:
                    partition_tensor_list.append(y[node_pas])

                pro_list, _ = model.auto_regressive.forward(candidate_tensor_list, partition_tensor_list)

                if len(pro_list) > 11:
                    temp2_list = []
                    for i in range(len(pro_list) - 1):
                        temp2_list.append([i, pro_list[i]])
                    s = sorted(temp2_list, key=lambda x: x[1], reverse=True)

                    for idx, _ in s[10:]:
                        delete_node_list.append(candidate[idx])

                target_idx = int(pro_list.argmax())
                if target_idx == len(pro_list) - 1:
                    target_node = -1
                else:
                    target_node = candidate[target_idx]
                    temp_list.append(target_node)

                if len(temp_list) >= 16:
                    break

            generated_communities_list.append(temp_list)
            print("get: num_of_com:", len(temp_list))
            count += 1
        point += 1

    return generated_communities_list


def randomly_expend(sorted_node_list, dataset, valid_comms, budget):
    node_adj_dict = np.load(dataset + "/node_adj_dict.npy", allow_pickle=True).item()
    count = 0
    idx = 0
    real_comms = np.load('./communities/{}.npy'.format(valid_comms), allow_pickle=True)
    size_list = [len(i) for i in real_comms]
    output_list = []
    while count < budget:
        seed = sorted_node_list[idx]
        size = random.choice(size_list)
        cur_c = [seed]

        temp_size = 1
        while temp_size < size:
            candidates = boundary_sample(node_adj_dict=node_adj_dict, community=cur_c)
            if len(candidates) == 0:
                break
            else:
                next_node = random.choice(candidates)
                cur_c.append(next_node)
                temp_size += 1

        output_list.append(cur_c)
        idx += 1
        count += 1

    return output_list


def f1_calc(list_a, list_b):
    tp = 0
    fp = 0
    for node in list_a:
        if node in list_b:
            tp += 1
        else:
            fp += 1
    if tp == 0:
        return 0
    fn = len(list_b) - tp
    p = float(tp / (tp + fp))
    r = float(tp / (tp + fn))
    return 2 * p * r / (p + r)


def jac_calc(list_a, list_b):
    cross_num = 0
    for node in list_a:
        if node in list_b:
            cross_num += 1
    jac = float(cross_num / (len(list_a) + len(list_b) - cross_num))
    return jac


def model_detection_request(ss_model, cg_model, dataset, graph_name, budget=10):
    cur_model = GPT_GIN2(gnn_input_dim=64, gnn_hidden_dim=64, gnn_output_dim=64, device='cpu')
    cur_model.load_state_dict(
        torch.load("./models/seed_selectors/models/{}.npy".format(ss_model), map_location='cpu'))
    node_rank_list = node_classification_test(model=cur_model, dataset=dataset)
    cur_model = GPT_GIN2(gnn_input_dim=64, gnn_hidden_dim=64, gnn_output_dim=64, device='cpu')
    cur_model.load_state_dict(
        torch.load("./models/community_generators/models/{}.npy".format(cg_model), map_location='cpu'))

    result_comms = community_generate_test(model=cur_model, sorted_node_list=node_rank_list, dataset=dataset,
                                           budget=budget)
    return result_comms


def model_valid_request(ss_model, cg_model, dataset, graph_name, comms, budget=10):
    print("seed select start")
    if ss_model == "random":
        node_rank_list = list(np.load('{}/node_adj_dict.npy'.format(dataset), allow_pickle=True).item())
        random.shuffle(node_rank_list)
    else:

        cur_model = GPT_GIN2(gnn_input_dim=64, gnn_hidden_dim=64, gnn_output_dim=64, device='cpu')
        cur_model.load_state_dict(
            torch.load("./models/seed_selectors/models/{}.npy".format(ss_model), map_location='cpu'))

        node_rank_list = node_classification_test(model=cur_model, dataset=dataset)
    # node_rank_list = np.load("./sorted_node_list.npy", allow_pickle=True)

    print("community generate start")
    if cg_model == "random":
        result_comms = randomly_expend(sorted_node_list=node_rank_list, dataset=dataset, valid_comms=comms,
                                       budget=budget)
    else:
        cur_model = GPT_GIN2(gnn_input_dim=64, gnn_hidden_dim=64, gnn_output_dim=64, device='cpu')
        cur_model.load_state_dict(
            torch.load("./models/community_generators/models/{}.npy".format(cg_model), map_location='cpu'))

        result_comms = community_generate_test(model=cur_model, sorted_node_list=node_rank_list, dataset=dataset,
                                               budget=budget)
    real_comms = np.load('./communities/{}.npy'.format(comms), allow_pickle=True)

    print("access start")
    f1_score = 0
    jac_score = 0
    count = 0
    for result_comm in result_comms:
        temp_max_f1 = 0
        temp_max_jac = 0
        for real_comm in real_comms:
            temp_max_f1 = max(temp_max_f1, f1_calc(real_comm, result_comm))
            temp_max_jac = max(temp_max_jac, jac_calc(real_comm, result_comm))
        f1_score += temp_max_f1
        jac_score += temp_max_jac
        count += 1

    f1_result = f1_score / count
    jac_result = jac_score / count

    output = []
    for com in result_comms[:5]:
        output.append([str(j) for j in com])

    record = list(np.load('./models/model_valid.npy', allow_pickle=True))
    record.append({"selector": ss_model, "expender": cg_model, "output_nums": budget, "valid_comms": comms,
                   'f1': format(f1_result, '.4f'), "jac": format(jac_result, '.4f'),
                   "create_time": time.asctime(time.localtime(time.time())), "valid_graph": graph_name})
    np.save('./models/model_valid.npy', record)
    return f1_result, jac_result, output


def seed_search_request(cg_model, seed, dataset):
    cur_model = GPT_GIN2(gnn_input_dim=64, gnn_hidden_dim=64, gnn_output_dim=64, device='cpu')
    cur_model.load_state_dict(
        torch.load("./models/community_generators/models/{}.npy".format(cg_model), map_location='cpu'))

    node_rank_list = [seed]

    result_comms = community_generate_test(model=cur_model, sorted_node_list=node_rank_list, dataset=dataset,
                                           budget=1)

    return result_comms[0]
