import os
from pathlib import Path
import shutil
import itertools
import yaml

home = str(Path.home())
# base = os.path.join(home, "Downloads/kernel_base_4_8_8")
base = os.path.join(home, "Downloads/kernel_base_2_4_4")
models = ['detection_ssd', 'detection_yolov3', 'fccm20_inceptionv3', 'fccm20_mobilenetv1', 'fccm20_resnet50', 'fccm20_vgg16', 'mobilenet_ssd', 'yolov3_tiny']

base_pp = 2
base_icp = 4
base_ocp = 4
core_num = int(16 * (4/base_pp) * (8/base_icp) * (8/base_ocp))

"""
We need a data structure to store all the latency information and parallelism configurations
For each model, it can occupy 1-n cores, and for each number of cores it can chooose many parallelism configs.

To store the parallelism configs, we need such data structure:
{model_name : { perf_model_index : [core1_config, ..., coren_config]}}

To store the latency info:
{model_name : { perf_model_index : {layer_name : [latency_core1, ..., latency_coren] }}}

"""
def build_voted_latency_model(name, para_confg):
    os.makedirs(name)
    for model in models:
        model_path = os.path.join(name, model)
        if not os.path.exists(model_path): os.makedirs(model_path)
        for core in range(1, 1+core_num):
            para = para_confg[core-1]
            src = os.path.join(base, model, 'latency_output', 'kernel_size_' + str(core), para)
            if not os.path.exists(src): continue
            if len(os.listdir(src)) == 0: print("WARNING: Empty folder " + src)
            tgt = os.path.join(model_path, 'kernel_size_' + str(core))
            shutil.copytree(src, tgt)
            print(f"Copying from {src} to {tgt}")


def get_latency(parent_dir, as_dict=False):
    latency_dict = dict()
    for file in os.listdir(parent_dir):
        if not file.endswith('1_card_1.txt'): continue
        file = os.path.join(parent_dir, file)
        with open(file, 'r') as f:
            lines = f.readlines()
            lines = lines[0].replace('{', '').replace('}', '')
            layers = lines.split(',')
            for layer in layers:
                layer_name, latency = layer.split(':')
                layer_name = layer_name.replace('\'', '')
                latency = float(latency)
                latency_dict[layer_name] = latency
        break
    if as_dict:
        return latency_dict
    return latency['total']


def vote(dictionary):
    count = dict()
    # k is model name, v is parallelism
    for k, v in dictionary.items():
        if v in count: 
            count[v] += 1
        else:
            count[v] = 0
    best_v = max(count, key=count.get)
    return best_v


def find_parallelism(enable_partial_sum_ring=True):
    para_dict = dict()
    latency_dict = dict()
    for model in models:
        para_dict[model] = dict()
        latency_dict[model] = dict()
        for core in range(1, 1+core_num):
            para_dict[model][core] = list()
            latency_dict[model][core] = dict()
            target_path = os.path.join(base, model, 'latency_output', 'kernel_size_' + str(core))
            for para in os.listdir(target_path):
                para_dir = os.path.join(target_path, para)
                if not os.path.isdir(para_dir): continue
                if para_dir.endswith('_error'): continue
                pp, icp, ocp = [int(s) for s in para.split('_')]
                if enable_partial_sum_ring is False and icp > base_icp: continue 
                latency = get_latency(para_dir, as_dict=True)
                para_dict[model][core].append(para)
                latency_dict[model][core][para] = latency
    return para_dict, latency_dict
        

def build_perf_models(para_dict, latency_dict, name):
    """
    para_dict:     para_dict[model][core] = [para1, para2, ...]
    latency_dict:  latency_dict[model][core][para] = {layer1 : 1ms, layer2 : 2ms, ...}
    """
    config_dict = dict()
    perfmodel_dict = dict()
    core_num_dict = dict()

    for model in models:
        core_num_list = [i for i in range(1, 1+core_num)]
        config_dict[model] = dict()
        perfmodel_dict[model] = dict()

        index_list = []
        for core in range(1, 1+core_num):
            num_para = len(para_dict[model][core]) 
            if num_para == 0:
                core_num_list.remove(core)
                continue 
            index_list.append([i for i in range(num_para)])

        core_num_dict[model] = core_num_list

        perf_model_idx = 0
        for idx in itertools.product(*index_list):
            # idx is a tuple. It's length is core_num.
            config_dict[model][perf_model_idx] = list()
            perfmodel_dict[model][perf_model_idx] = dict()
            for core in core_num_list:
                para = para_dict[model][core][ idx[core_num_list.index(core)] ]
                config_dict[model][perf_model_idx].append(para)
                layer_dict = latency_dict[model][core][para]
                for layer_name, latency in layer_dict.items():
                    if layer_name not in perfmodel_dict[model][perf_model_idx]:
                        perfmodel_dict[model][perf_model_idx][layer_name] = list()
                    perfmodel_dict[model][perf_model_idx][layer_name].append(latency)
            perf_model_idx += 1
            if perf_model_idx == 1000:
                break
    
    # print perf_model to file
    name = "./" + name
    if not os.path.exists(name): os.makedirs(name)
    for model in models:
        core_num_list = core_num_dict[model]

        model_dir = os.path.join(name, model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for perf_model_idx, latency_dict in perfmodel_dict[model].items():
            filename = os.path.join(model_dir, str(perf_model_idx) + ".txt")
            lines = "kernel_num"
            for num in core_num_list: lines += (',' + str(num))
            lines += "\n"
            for layer_name, latency_list in latency_dict.items():
                lines += layer_name
                for l in latency_list: lines += (',' + str(l))
                lines += "\n"
            with open(filename, "w") as f:
                f.writelines(lines)
            print("written file: " + filename)
        with open(os.path.join(model_dir, "config_table.yaml"), 'w') as f:
            yaml.dump(config_dict[model], f, default_flow_style=False)


if __name__ == "__main__":
    # para_dict, latency_dict = find_parallelism(enable_partial_sum_ring=True)
    # build_perf_models(para_dict, latency_dict, name="224_with_partialsum")
    
    para_dict, latency_dict = find_parallelism(enable_partial_sum_ring=False)
    build_perf_models(para_dict, latency_dict, name="224_without_partialsum")
    


    # print("When partial sum ring is disabled: ")
    # config = find_parallelism(enable_partial_sum_ring=False)
    # build_voted_latency_model("no_partialsum_244", config)



