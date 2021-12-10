import random
import numpy as np
import collections
import torch

# from Model import *


def instance_sample(node_adj_dict, node_fea_dict, instance_pair_num=100000, instance_size=50, rs_radio=0.6):
    roots = random.sample(list(node_adj_dict), instance_pair_num)

    instances_pair_node_num_list = []
    instance_pair_list = []
    max_node_count = 0
    for root in roots:
        temp_list = []
        temp_list_num = []
        for i in range(2):
            current_instance = [root]
            current_node = root
            node_count = 1
            for count in range(instance_size):
                temp = random.random()
                if temp >= rs_radio:
                    candidate_list = node_adj_dict[current_node]
                    new_node = random.choice(candidate_list)
                    if new_node not in current_instance:
                        current_instance.append(new_node)
                        node_count += 1
                    current_node = new_node
                else:
                    current_node = root
            temp_list.append(current_instance)
            temp_list_num.append(node_count)
            max_node_count = max(max_node_count, node_count)
        instance_pair_list.append(temp_list)
        instances_pair_node_num_list.append(temp_list_num)

    instance_pair_edge_num_list = []
    instance_pair_adj_matrices = []
    instance_pair_fea_matrices = []
    max_count = 0
    for instance_pair in instance_pair_list:
        temp_list = []
        temp2_list = []
        temp3_list = []
        for instance in instance_pair:
            instance_temp_list = []
            instance_temp2_list = []
            edge_count = 0
            for node in instance:
                for target_node in instance[instance.index(node) + 1:]:
                    if target_node in node_adj_dict[node]:
                        edge_count += 2
                        instance_temp_list.append([instance.index(node), instance.index(target_node)])
                        instance_temp_list.append([instance.index(target_node), instance.index(node)])
                instance_temp2_list.append(node_fea_dict[node])
            for _ in range(max_node_count - len(instance)):
                instance_temp2_list.append([0] * 64)
            temp_list.append(instance_temp_list)
            temp2_list.append(instance_temp2_list)
            temp3_list.append(edge_count)
            max_count = max(max_count, edge_count)
        instance_pair_adj_matrices.append(temp_list)
        instance_pair_fea_matrices.append(temp2_list)
        instance_pair_edge_num_list.append(temp3_list)

    for instance_pair_idx in range(len(instance_pair_list)):
        for i in range(2):
            for i_i in range(max_count - instance_pair_edge_num_list[instance_pair_idx][i]):
                instance_pair_adj_matrices[instance_pair_idx][i].append([-1, -1])

    return instance_pair_list, instance_pair_fea_matrices, instance_pair_adj_matrices, instance_pair_edge_num_list, instances_pair_node_num_list


def permutation_sample(node_adj_dict, community):
    permutation = []
    seed_node = random.choice(community)
    permutation.append(seed_node)
    while len(permutation) != len(community):
        candidate = []
        for node in community:
            if node not in permutation:
                for node_y in permutation:
                    if node_y in node_adj_dict[node]:
                        candidate.append(node)
                        break
        next_node = random.choice(candidate)
        permutation.append(next_node)
    return permutation


def boundary_sample(node_adj_dict, community):
    candidate = []
    for node in community:
        for target_node in node_adj_dict[node]:
            if target_node not in community and target_node not in candidate:
                candidate.append(target_node)
    return candidate


def communities_boundary_sample(node_adj_dict, communities):
    boundaries = []
    for community in communities:
        boundaries.append(boundary_sample(node_adj_dict, community))

    return boundaries


def pair_wise_process(sampled_communities, sampled_communities_boundaries):
    count_node_list_dict = collections.defaultdict(list)
    node_count_dict = collections.defaultdict(int)
    for community in sampled_communities:
        for node in community:
            node_count_dict[node] += 1

    for boundary in sampled_communities_boundaries:
        for out_node in boundary:
            node_count_dict[out_node] += 0

    for node in list(node_count_dict):
        count_node_list_dict[node_count_dict[node]].append(node)

    count_list = list(count_node_list_dict)
    #   由高(0)到低(len()-1)
    sorted_count_list = sorted(count_list, reverse=True)

    out_list = []
    for sorted_count in sorted_count_list:
        out_list.append(count_node_list_dict[sorted_count])

    pair_list = []
    for idx in range(len(out_list) - 1):
        for node_1 in out_list[idx]:
            for down_idx in range(1, len(out_list) - idx):
                for node_2 in out_list[idx + down_idx]:
                    pair_list.append([node_1, node_2])
    #   O(1) > O(2)
    return pair_list


def dagger(model, sample_communities, node_adj_dict, all_graph_fea_list, all_graph_edge_index_list):
    model.eval()

    rec_partition_solution_list = []
    rec_candidate_list = []
    rec_target_node_list = []
    adj_tensor = torch.tensor(all_graph_edge_index_list, dtype=torch.long).t().to(model.device)
    for community in sample_communities:

        seed = random.choice(community)
        solution = [seed]
        target_node = seed

        while target_node != -1:

            x_tensor = torch.FloatTensor(all_graph_fea_list).to(model.device)
            for node in solution:
                x_tensor[node] += model.c_signal

            candidate = boundary_sample(node_adj_dict=node_adj_dict, community=solution)
            if len(candidate) == 0:
                target_node = -1
                solution.append(target_node)
                continue

            for node in candidate:
                x_tensor[node] += model.b_signal

            y = model.auto_regressive_gnn(x_tensor, adj_tensor)

            candidate_tensor_list = []
            for node in candidate:
                candidate_tensor_list.append(y[node])

            partition_tensor_list = []
            for node in solution:
                partition_tensor_list.append(y[node])

            pro_list, _ = model.auto_regressive.forward(candidate_tensor_list, partition_tensor_list)

            target_idx = int(pro_list.argmax())

            if target_idx == len(pro_list) - 1:
                target_node = -1
                solution.append(target_node)
            else:
                target_node = candidate[target_idx]
                solution.append(target_node)
                if len(solution) > 16:
                    target_node = -1
                    solution.append(target_node)

        temp_sign = 0

        for idx in range(1, len(solution)):
            if solution[idx] not in community:
                if solution[idx] == -1:
                    node_lost = []
                    for node in community:
                        if node not in solution[: -1]:
                            node_lost.append(node)
                    temp_partition_solution = solution[: -1]
                    while len(node_lost) >= 1:
                        temp_candidate = boundary_sample(node_adj_dict=node_adj_dict, community=temp_partition_solution)
                        temp_list = []
                        for node in node_lost:
                            if node in temp_candidate:
                                temp_list.append(node)
                        rec_target_node = random.choice(temp_list)

                        rec_partition_solution_list.append(temp_partition_solution.copy())
                        rec_candidate_list.append(temp_candidate.copy())
                        rec_target_node_list.append(rec_target_node)

                        temp_partition_solution.append(rec_target_node)
                        node_lost.remove(rec_target_node)
                        temp_sign = 1

                    if temp_sign == 1:
                        temp_candidate = boundary_sample(node_adj_dict=node_adj_dict, community=temp_partition_solution)

                        rec_partition_solution_list.append(temp_partition_solution.copy())
                        rec_candidate_list.append(temp_candidate.copy())
                        rec_target_node_list.append(-1)

                    # if len(node_lost) >= 1:
                    #     temp_candidate = boundary_sample(node_adj_dict=node_adj_dict, community=solution[: -1])
                    #     temp_list = []
                    #     for node in node_lost:
                    #         if node in temp_candidate:
                    #             temp_list.append(node)
                    #     rec_target_node = random.choice(temp_list)
                    #
                    #     temp_sign = 1
                    #     rec_partition_solution_list.append(solution[: idx])
                    #     rec_candidate_list.append(temp_candidate)
                    #     rec_target_node_list.append(rec_target_node)

                else:
                    if solution[idx + 1] not in community:
                        if solution[idx + 1] == -1:
                            node_lost = []
                            for node in community:
                                if node not in solution[: idx]:
                                    node_lost.append(node)

                            temp_partition_solution = solution[: idx + 1]
                            while len(node_lost) >= 1:
                                temp_candidate = boundary_sample(node_adj_dict=node_adj_dict,
                                                                 community=temp_partition_solution)
                                temp_list = []
                                for node in node_lost:
                                    if node in temp_candidate:
                                        temp_list.append(node)
                                rec_target_node = random.choice(temp_list)

                                rec_partition_solution_list.append(temp_partition_solution.copy())
                                rec_candidate_list.append(temp_candidate.copy())
                                rec_target_node_list.append(rec_target_node)

                                temp_partition_solution.append(rec_target_node)
                                node_lost.remove(rec_target_node)
                                temp_sign = 1

                            if temp_sign == 1:
                                temp_candidate = boundary_sample(node_adj_dict=node_adj_dict,
                                                                 community=temp_partition_solution)

                                rec_partition_solution_list.append(temp_partition_solution.copy())
                                rec_candidate_list.append(temp_candidate.copy())
                                rec_target_node_list.append(-1)
                            # if len(node_lost) >= 1:
                            #     temp_candidate = boundary_sample(node_adj_dict=node_adj_dict,
                            #                                      community=solution[: idx + 1])
                            #     temp_list = []
                            #     for node in node_lost:
                            #         if node in temp_candidate:
                            #             temp_list.append(node)
                            #     rec_target_node = random.choice(temp_list)
                            #
                            #     temp_sign = 1
                            #     rec_partition_solution_list.append(solution[: idx + 1])
                            #     rec_candidate_list.append(temp_candidate)
                            #     rec_target_node_list.append(rec_target_node)
                        else:
                            node_lost = []
                            for node in community:
                                if node not in solution[: idx]:
                                    node_lost.append(node)
                            if len(node_lost) == 0:
                                temp_candidate = boundary_sample(node_adj_dict=node_adj_dict,
                                                                 community=solution[: idx + 1])

                                temp_sign = 1
                                rec_partition_solution_list.append(solution[: idx + 1].copy())
                                rec_candidate_list.append(temp_candidate.copy())
                                rec_target_node_list.append(-1)

                            else:
                                temp_partition_solution = solution[: idx + 1]
                                while len(node_lost) >= 1:
                                    temp_candidate = boundary_sample(node_adj_dict=node_adj_dict,
                                                                     community=temp_partition_solution)
                                    temp_list = []
                                    for node in node_lost:
                                        if node in temp_candidate:
                                            temp_list.append(node)
                                    rec_target_node = random.choice(temp_list)

                                    rec_partition_solution_list.append(temp_partition_solution.copy())
                                    rec_candidate_list.append(temp_candidate.copy())
                                    rec_target_node_list.append(rec_target_node)

                                    temp_partition_solution.append(rec_target_node)
                                    node_lost.remove(rec_target_node)
                                    temp_sign = 1

                                if temp_sign == 1:
                                    temp_candidate = boundary_sample(node_adj_dict=node_adj_dict,
                                                                     community=temp_partition_solution)

                                    rec_partition_solution_list.append(temp_partition_solution.copy())
                                    rec_candidate_list.append(temp_candidate.copy())
                                    rec_target_node_list.append(-1)
            if temp_sign == 1:
                break

    return rec_partition_solution_list, rec_candidate_list, rec_target_node_list


def AR_data_generate(communities_list, node_adj_dict):
    partition_solution_list = []
    candidate_list = []
    target_node_list = []
    for community in communities_list:
        permutation = permutation_sample(node_adj_dict=node_adj_dict, community=community)
        for t in range(1, len(permutation) + 1):
            current_community = permutation[: t]
            current_boundary = boundary_sample(node_adj_dict=node_adj_dict, community=current_community)

            partition_solution_list.append(current_community.copy())
            candidate_list.append(current_boundary.copy())

            if t == len(permutation):
                target_node_list.append(-1)
            else:
                target_node_list.append(permutation[t])

    return partition_solution_list, candidate_list, target_node_list
