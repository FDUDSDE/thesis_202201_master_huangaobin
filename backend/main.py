from flask import Flask, request
import json
import os
from TR import *
from Test import *
import time
import numpy as np
import _thread

from flask_cors import *

app = Flask(__name__)

CORS(app, supports_credentials=True)


#   请求数据集列表
@app.route('/get_all_datasets', methods=['POST', 'GET'])
def get_all_datasets(time_interval=300, is_bin=False):
    print("receive_get_all_datasets_request")
    datasets = os.listdir("./datasets")

    res = {"datasets": datasets, "code": 500}
    return json.dumps(res)


#   请求生成器列表
@app.route('/get_all_cg_models', methods=['POST', 'GET'])
def get_all_cg_models(time_interval=300, is_bin=False):
    print("receive_get_all_clusterers_request")
    models = os.listdir("./models/community_generators")
    models = [i.split(".")[0] for i in models]

    res = {"cg_models": models, "code": 500}
    return json.dumps(res)


#   请求选择器列表
@app.route('/get_all_ss_models', methods=['POST', 'GET'])
def get_all_ss_models(time_interval=300, is_bin=False):
    print("receive_get_all_selectors_request")
    models = os.listdir("./models/seed_selectors")
    models = [i.split(".")[0] for i in models]

    res = {"ss_models": models, "code": 500}
    return json.dumps(res)


#   请求群组数据列表
@app.route('/get_all_communities', methods=['POST', 'GET'])
def get_all_communities(time_interval=300, is_bin=False):
    print("receive_get_all_communities_request")
    comms = os.listdir("./communities")
    comms = [i.split(".")[0] for i in comms]

    res = {"comms": comms, "code": 500}
    return json.dumps(res)


#   登录请求
@app.route('/login', methods=['POST', 'GET'])
def login_request(time_interval=300, is_bin=False):
    data_raw = request.data
    data_raw = eval(data_raw)

    user = data_raw['user']
    password = data_raw['password']

    print("--{}___{}--".format(user, password))

    if user == 'admin' and password == "admin":
        res = {"mes": 'success', "code": 500}
    else:
        res = {"mes": 'fail', "code": 13}
    return json.dumps(res)


#   模型训练请求
@app.route('/model_train', methods=['POST', 'GET'])
def model_train(time_interval=300, is_bin=False):
    print("receive_model_train_request")
    #   request解析
    data_raw = request.args
    graph = data_raw['graph']
    comms = data_raw['comms']
    epochs = int(data_raw["epochs"])

    #   数据集路径
    dataset_path = './datasets/{}'.format(graph)

    if graph not in os.listdir("./datasets"):
        res = {"state": "Data Wrong", "ss_model": "", "cl_model": "", "time_cost": ""}
    else:
        t1 = time.time()
        model_train_request(dataset=dataset_path, graph_name=graph, epoch=epochs, comms=comms)
        t2 = time.time()
        res = {"state": "Train Finish", "ss_model": "{}_selector".format(graph),
               "cl_model": "{}_clusterer".format(graph),
               "time_cost": t2 - t1}
    time.sleep(30)
    return json.dumps(res)


#   模型验证请求
@app.route('/model_valid', methods=['POST', 'GET'])
def model_valid(time_interval=300, is_bin=False):
    print("receive_model_valid_request")
    #   request解析
    data_raw = request.args
    ss_model = data_raw['ss_model']
    cg_model = data_raw['cg_model']
    graph = data_raw["graph"]
    # comms = data_raw['comms']
    output_nums = int(data_raw['output_nums'])

    #   数据集路径
    dataset_path = './datasets/{}'.format(graph)

    f1, jac, results = model_valid_request(ss_model=ss_model, cg_model=cg_model, dataset=dataset_path, graph_name=graph,
                                           budget=output_nums)
    res = {'f1': str(f1), 'jac': str(jac), "output_comms": results}
    print(res)
    return json.dumps(res)


@app.route('/seed_search', methods=['POST', 'GET'])
def seed_search(time_interval=300, is_bin=False):
    print("receive_seed_search_request")
    data_raw = request.args

    seed = int(data_raw['seed'])
    cg_model = data_raw['cg_model']
    graph = data_raw['graph']

    dataset_path = './datasets/{}'.format(graph)

    result = seed_search_request(cg_model=cg_model, seed=seed, dataset=dataset_path)
    res = {'search_result': result}
    return json.dumps(res)


@app.route('/get_data_details', methods=['POST', 'GET'])
def get_data_details(time_interval=300, is_bin=False):
    print("receive_get_data_details_request")
    data_raw = request.args

    graph = data_raw['graph']
    comms = data_raw['comms']
    print(graph, comms)
    dataset_path = './datasets/{}'.format(graph)

    a = np.load(dataset_path + '/all_graph_fea_list.npy', allow_pickle=True)
    node_num = len(a)
    a = np.load(dataset_path + '/all_graph_edge_index_list.npy', allow_pickle=True)
    edge_num = len(a) / 2

    comms_data = np.load("./communities/{}.npy".format(comms), allow_pickle=True)
    size_list = [len(i) for i in comms_data]

    res = {'graph': graph, 'node_num': str(node_num), 'edge_num': str(int(edge_num)), 'comms': comms,
           'comms_num': str(len(size_list)), 'max_size': str(np.max(size_list)), 'min_size': str(np.min(size_list)),
           'mean_size': str(np.mean(size_list))}
    return json.dumps(res)


@app.route('/data_upload', methods=['POST', 'GET'])
def data_upload(time_interval=300, is_bin=False):
    print("receive_get_data_details_request")
    # data_raw = request.args
    time.sleep(5)
    # graph = data_raw['graph']
    # comms = data_raw['comms']
    # print(graph, comms)
    # dataset_path = './datasets/{}'.format(graph)
    #
    # a = np.load(dataset_path + '/all_graph_fea_list.npy', allow_pickle=True)
    # node_num = len(a)
    # a = np.load(dataset_path + '/all_graph_edge_index_list.npy', allow_pickle=True)
    # edge_num = len(a) / 2
    #
    # comms_data = np.load("./communities/{}.npy".format(comms), allow_pickle=True)
    # size_list = [len(i) for i in comms_data]
    #
    # res = {'graph': graph, 'node_num': str(node_num), 'edge_num': str(int(edge_num)), 'comms': comms,
    #        'comms_num': str(len(size_list)), 'max_size': str(np.max(size_list)), 'min_size': str(np.min(size_list)),
    #        'mean_size': str(np.mean(size_list))}
    res = {'graph': 'amazon', "edges": '33767', 'nodes': '13178', 'time': '20210901'}
    return json.dumps(res)


@app.route('/graph_data/request', methods=['GET'])
def graph_data_request():
    print("receive: graph_data_request")

    res = []
    datasets = os.listdir('./datasets')
    for dataset in datasets:
        edges = np.load("./datasets/{}/all_graph_edge_index_list.npy".format(dataset), allow_pickle=True)
        edges_num = len(edges) / 2

        nodes = np.load("./datasets/{}/all_graph_fea_list.npy".format(dataset), allow_pickle=True)
        nodes_num = len(nodes)

        comms = np.load("./datasets/{}/communities_list.npy".format(dataset), allow_pickle=True)
        comms_num = len(comms)

        len_comms = [len(i) for i in comms]
        max_comm = max(len_comms)
        min_comm = min(len_comms)
        ave_comm = format(np.mean(len_comms), '.4f')

        res.append({'graph': dataset, 'edges': edges_num, 'nodes': nodes_num, 'comms': comms_num, 'min_comm': min_comm,
                    'max_comm': max_comm, 'ave_comm': ave_comm, 'accessable': 'Y'})

    return json.dumps(res)


@app.route('/comms_data/request', methods=['GET'])
def comms_data_request():
    print("receive: comms_data_request")

    res = []
    datasets = os.listdir('./communities')
    for dataset in datasets:
        comms = np.load('./communities/{}'.format(dataset), allow_pickle=True)

        len_comms = [len(i) for i in comms]
        max_comm = max(len_comms)
        min_comm = min(len_comms)
        ave_comm = format(np.mean(len_comms), '.4f')

        res.append({'comms': dataset.split('.')[0], 'nums': len(comms), 'min_comm': min_comm, 'max_comm': max_comm,
                    'ave_comm': ave_comm, 'accessable': 'Y'})

    return json.dumps(res)


@app.route('/model/request', methods=['GET'])
def model_request():
    print("receive: model_request")

    selector_list = []
    detail_dict = np.load('./models/seed_selectors/model_details_dict.npy', allow_pickle=True).item()
    selectors = os.listdir('./models/seed_selectors/models')
    for selector in selectors:
        selector_name = selector.split('.')[0]

        selector_list.append({'name': selector_name, 'create_time': detail_dict[selector_name]["create_time"],
                              'train_epoch': detail_dict[selector_name]["train_epoch"],
                              'learning_rate': detail_dict[selector_name]["learning_rate"],
                              'train_graph': detail_dict[selector_name]["train_graph"],
                              'train_comms': detail_dict[selector_name]["train_comms"],
                              'batch_size': detail_dict[selector_name]["batch_size"],
                              'train_size': detail_dict[selector_name]["train_size"],
                              'pairs': detail_dict[selector_name]["pairs"]})

    expender_list = []
    detail_dict = np.load('./models/community_generators/model_details_dict.npy', allow_pickle=True).item()
    expenders = os.listdir('./models/community_generators/models')
    for expender in expenders:
        expender_name = expender.split('.')[0]
        expender_list.append(
            {'name': expender_name, 'create_time': detail_dict[expender_name]["create_time"],
             'train_epoch': detail_dict[expender_name]["train_epoch"],
             'learning_rate': detail_dict[expender_name]["learning_rate"],
             'train_graph': detail_dict[expender_name]["train_graph"],
             'train_comms': detail_dict[expender_name]["train_comms"],
             'batch_size': detail_dict[expender_name]["batch_size"],
             'train_size': detail_dict[expender_name]["train_size"], 'ada_size': detail_dict[expender_name]["ada"]})

    res = {'selectors': selector_list, 'expenders': expender_list}

    return json.dumps(res)


@app.route('/graph/request', methods=['GET'])
def graph_request():
    print("receive: graph_request")

    res = []
    datasets = os.listdir('./datasets')
    for idx, dataset in enumerate(datasets):
        res.append({'graph': dataset, 'id': idx})

    return json.dumps(res)


@app.route('/comms/request', methods=['GET'])
def comms_request():
    print("receive: comms_request")

    res = []
    datasets = os.listdir('./communities')
    for idx, dataset in enumerate(datasets):
        res.append({'comms': dataset.split('.')[0], 'id': idx})

    return json.dumps(res)


@app.route('/model/train', methods=['POST'])
def model_train1():
    print("receive: train_request")

    #   request解析
    try:
        data_raw = request.form
        graph = data_raw['graph']
        comms = data_raw['comms']
        train_size = int(data_raw['transize'])
        ss_epochs = int(data_raw['ssepochs'])
        ss_batch_size = int(data_raw['ssbatchsize'])
        ss_lr = float(data_raw['sslr'])
        pairs = int(data_raw['sspairs'])
        cg_epochs = int(data_raw['cgepochs'])
        cg_batch_size = int(data_raw['cgbatchsize'])
        cg_lr = float(data_raw['cglr'])
        ada_num = int(data_raw['ada'])
    except Exception as e:
        res = {"state": 501, "msg": e}
        return json.dumps(res)

    #   数据集路径
    dataset_path = './datasets/{}'.format(graph)
    print(data_raw)

    model_train_request(dataset=dataset_path, graph_name=graph, comms=comms, train_size=train_size, epoch_ss=ss_epochs,
                        batch_size_ss=ss_batch_size, lr_ss=ss_lr, epoch_cg=cg_epochs, batch_size_cg=cg_batch_size,
                        lr_cg=cg_lr, ada=ada_num, pairs=pairs)

    res = {"state": 20000, 'msg': 'start train'}
    return json.dumps(res)


@app.route('/model/valid', methods=['POST'])
def model_valid1():
    print("receive: valid_request")

    #   request解析
    try:
        data_raw = request.form
        selector = data_raw['selector']
        generator = data_raw['expender']
        graph = data_raw['graph']
        comms = data_raw['comms']
        output_nums = int(data_raw['outputnums'])
    except Exception as e:
        res = {"state": 501, "msg": e}
        return json.dumps(res)

    #   数据集路径
    dataset_path = './datasets/{}'.format(graph)

    f1_result, jac_result, output = model_valid_request(ss_model=selector, cg_model=generator, dataset=dataset_path,
                                                        graph_name=graph, comms=comms, budget=output_nums)

    res = {"state": 20000, 'msg': 'start valid'}
    print("valid finish f1:{} jac:{}".format(f1_result, jac_result))
    return json.dumps(res)


@app.route('/valid/request', methods=['GET'])
def valid_request():
    print("receive: validRecord_request")

    records = list(np.load('./models/model_valid.npy', allow_pickle=True))
    res = [i for i in records]

    return json.dumps(res)


@app.route('/community/detection', methods=['POST'])
def community_detection():
    print("receive: community_detection_request")

    #   request解析
    try:
        data_raw = request.form
        selector = data_raw['selector']
        generator = data_raw['expender']
        graph = data_raw['graph']
        output_nums = int(data_raw['outputnums'])
    except Exception as e:
        res = {"state": 501, "msg": e}
        return json.dumps(res)

    #   数据集路径
    dataset_path = './datasets/{}'.format(graph)

    output = model_detection_request(ss_model=selector, cg_model=generator, dataset=dataset_path, graph_name=graph,
                                     budget=output_nums)

    output_dict = {}
    for idx, com in enumerate(output[:5]):
        output_dict[idx] = com
    file_name = "{}_{}_{}_{}.npy".format(selector, generator, graph, str(random.randint(1, 10000)))
    np.save('./results/{}'.format(file_name), output)
    detection_result_dict = list(np.load('./models/model_detection.npy', allow_pickle=True))
    detection_result_dict.append(
        {"selector": selector, "expender": generator, "graph": graph, "nums": len(output), "results": output_dict,
         'file': file_name})
    np.save('./models/model_detection.npy', detection_result_dict)
    res = {"state": 20000, 'msg': 'start valid'}
    print(output)
    return json.dumps(res)


@app.route('/detection/request', methods=['GET'])
def detection_request():
    print("receive: detection_request")

    records = list(np.load('./models/model_detection.npy', allow_pickle=True))
    res = [i for i in records]

    return json.dumps(res)


@app.route('/community/search', methods=['POST'])
def community_search():
    print("receive: community_search_request")

    #   request解析
    try:
        data_raw = request.form
        generator = data_raw['expender']
        graph = data_raw['graph']
        seed = int(data_raw['seed'])
    except Exception as e:
        res = {"state": 501, "msg": e}
        return json.dumps(res)

    #   数据集路径
    dataset_path = './datasets/{}'.format(graph)

    result = seed_search_request(cg_model=generator, seed=seed, dataset=dataset_path)
    print(result)
    res = {"state": 500, "output": result}
    return json.dumps(res)


if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=7000,
        debug=False,
    )