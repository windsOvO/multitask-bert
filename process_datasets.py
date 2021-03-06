import json
from collections import defaultdict
from math import log


# 分割生成训练集和验证集
def split_dataset(dev_data_cnt):
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']:
        cnt = 0
        with open('./datasets/' + e + '/total.csv') as f:
            with open('./datasets/' + e + '/train.csv', 'w') as f_train:
                with open('./datasets/' + e + '/dev.csv', 'w') as f_dev:
                    for line in f:
                        cnt += 1
                        if cnt <= dev_data_cnt:
                            f_dev.write(line)
                        else:
                            f_train.write(line)


# 输出json文件中的数据个数或内容
def print_dataset(path, name, print_content=False):
    data_cnt = 0
    with open(path) as f:
        for line in f:
            tmp = json.loads(line)
            for _, v in tmp.items():
                data_cnt += 1
                if print_content:
                    print(v)
    print(name, 'contains:', data_cnt, 'numbers of data')


def generate_data():
    # 统计所有标签的类型
    label_set = dict()
    label_cnt_set = dict()
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']:
        label_set[e] = set()
        label_cnt_set[e] = defaultdict(int)
        with open('./datasets/' + e + '/total.csv') as f:
            for line in f:
                label = line.strip().split('\t')[-1]
                label_set[e].add(label)
                label_cnt_set[e][label] += 1
    for k in label_set:
        label_set[k] = sorted(list(label_set[k])) # 字母or数字排序
    print('all labels:')
    for k, v in label_set.items():
        print(k, v)
    print()
    # json化存储标签集
    with open('./datasets/label.json', 'w') as fw:
        fw.write(json.dumps(label_set))

    # 所有的标签权重值
    label_weight_set = dict()
    for k in label_set:
        label_weight_set[k] = [label_cnt_set[k][e] for e in label_set[k]]
        total_weight = sum(label_weight_set[k])
        label_weight_set[k] = [log(total_weight / e) for e in label_weight_set[k]]
    print('weight of all labels:')
    for k, v in label_weight_set.items():
        print(k, v)
    print()
    # json化存储标签权重
    with open('./datasets/label_weights.json', 'w') as fw:
        fw.write(json.dumps(label_weight_set))
    
    # csv数据集转json
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']:
        for name in ['dev', 'train']:
            with open('./datasets/' + e + '/' + name + '.csv') as fr:
                with open('./datasets/' + e + '/' + name + '.json', 'w') as fw:
                    json_dict = dict()
                    for line in fr:
                        tmp_list = line.strip().split('\t')
                        json_dict[tmp_list[0]] = dict()
                        json_dict[tmp_list[0]]['s1'] = tmp_list[1]
                        if e == 'OCNLI': # 包含两个sentence
                            json_dict[tmp_list[0]]['s2'] = tmp_list[2]
                            json_dict[tmp_list[0]]['label'] = tmp_list[3]
                        else:
                            json_dict[tmp_list[0]]['label'] = tmp_list[2]
                    fw.write(json.dumps(json_dict))
    
    print('datasets info:')
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']:
        for name in ['dev', 'train']:
            cur_path = './datasets/' + e + '/' + name + '.json'
            data_name = e + '_' + name
            print_dataset(cur_path, data_name)
            
    print_dataset('./datasets/label.json', 'label_set')
    

if __name__ == '__main__':
    print('--------------------- start processing ---------------------')
    split_dataset(dev_data_cnt=2500)
    generate_data()
    print('--------------------- finish processing ---------------------')