import torch
from transformers import BertModel, BertTokenizer
import json
import time
from utils import get_f1, print_result
from net import Net
from data_generator import Data_generator
from calculate_loss import Criterion


def train(epochs=20, print_freq=100, batchSize=64, lr=0.0001, device='cuda', accumulate=True, a_step=16, use_dtp=False,
        tokenizer_model='bert-base-chinese', pretrained_model='./pretrain_model', load_saved=False, finetuning_model='./models/saved_best.pt',
        weighted_loss=False):
    
    print('-------------------- prepare training --------------------')

    # 使用transformers加载预训练分词器
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
    # 加载已微调模型，或使用transformers加载预训练模型
    if load_saved:
        model = torch.load(finetuning_model)
        print('loading finetunning model...')
    else:
        model = Net(BertModel.from_pretrained(pretrained_model))
        print('loading pretrained model...')
    # 模型转换
    model.to(device, non_blocking=True) # 非阻塞提高速度

    # 加载数据集
    print('loading data...')

    label_dict = dict()
    with open('./datasets/label.json') as f:
        for line in f:
            label_dict = json.loads(line)
            break
    label_weights_dict = dict()
    with open('./datasets/label_weights.json') as f:
        for line in f:
            label_weights_dict = json.loads(line)
            break
    ocnli_train = dict()
    with open('./datasets/OCNLI/train.json') as f:
        for line in f:
            ocnli_train = json.loads(line)
            break
    ocnli_dev = dict()
    with open('./datasets/OCNLI/dev.json') as f:
        for line in f:
            ocnli_dev = json.loads(line)
            break
    ocemotion_train = dict()
    with open('./datasets/OCEMOTION/train.json') as f:
        for line in f:
            ocemotion_train = json.loads(line)
            break
    ocemotion_dev = dict()
    with open('./datasets/OCEMOTION/dev.json') as f:
        for line in f:
            ocemotion_dev = json.loads(line)
            break
    tnews_train = dict()
    with open('./datasets/TNEWS/train.json') as f:
        for line in f:
            tnews_train = json.loads(line)
            break
    tnews_dev = dict()
    with open('./datasets/TNEWS/dev.json') as f:
        for line in f:
            tnews_dev = json.loads(line)
            break

    # 数据处理
    train_data_generator = Data_generator(ocnli_train, ocemotion_train, tnews_train, label_dict, device, tokenizer)
    dev_data_generator = Data_generator(ocnli_dev, ocemotion_dev, tnews_dev, label_dict, device, tokenizer)

    print('configure criterion and optimizer...')
    # 标签权重加载
    tnews_weights = torch.tensor(label_weights_dict['TNEWS']).to(device, non_blocking=True)
    ocnli_weights = torch.tensor(label_weights_dict['OCNLI']).to(device, non_blocking=True)
    ocemotion_weights = torch.tensor(label_weights_dict['OCEMOTION']).to(device, non_blocking=True)

    # 评判器
    criterion = Criterion(label_dict, weighted=weighted_loss, tnews_weights=tnews_weights, ocnli_weights=ocnli_weights, ocemotion_weights=ocemotion_weights)
    # 优化器
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)

    # 标记最优epoch
    best_dev_f1 = 0.0
    best_epoch = -1

    print('-------------------- start training --------------------')

    for epoch in range(1, epochs + 1):
        # 训练模式
        model.train()
        train_loss = 0.0 # 训练误差
        train_total = 0 # 总量
        train_correct = 0 # 正确量
        train_ocnli_correct = 0
        train_ocemotion_correct = 0
        train_tnews_correct = 0
        train_ocnli_pred_list = [] # 预测列表
        train_ocnli_gold_list = []
        train_ocemotion_pred_list = []
        train_ocemotion_gold_list = []
        train_tnews_pred_list = []
        train_tnews_gold_list = []
        cnt_train = 0 # 训练数量

        while True:
            raw_data = train_data_generator.get_next_batch(batchSize)
            if raw_data == None:
                break

            data = dict()
            data['input_ids'] = raw_data['input_ids']
            data['token_type_ids'] = raw_data['token_type_ids']
            data['attention_mask'] = raw_data['attention_mask']
            data['ocnli_ids'] = raw_data['ocnli_ids']
            data['ocemotion_ids'] = raw_data['ocemotion_ids']
            data['tnews_ids'] = raw_data['tnews_ids']
            tnews_gold = raw_data['tnews_gold']
            ocnli_gold = raw_data['ocnli_gold']
            ocemotion_gold = raw_data['ocemotion_gold']

            # 不累计则清空梯度
            if not accumulate:
                optimizer.zero_grad()

            # 前向传播
            ocnli_pred, ocemotion_pred, tnews_pred = model(**data)

            if use_dtp:
                tnews_kpi = 0.1 if len(train_tnews_pred_list) == 0 else train_tnews_correct / len(train_tnews_pred_list)
                ocnli_kpi = 0.1 if len(train_ocnli_pred_list) == 0 else train_ocnli_correct / len(train_ocnli_pred_list)
                ocemotion_kpi = 0.1 if len(train_ocemotion_pred_list) == 0 else train_ocemotion_correct / len(train_ocemotion_pred_list)
                current_loss = criterion.compute_dtp(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold,
                                                   ocemotion_gold, tnews_kpi, ocnli_kpi, ocemotion_kpi)
            else:
                current_loss = criterion.compute(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
            
            train_loss += current_loss.item()

            # 反向传播
            current_loss.backward()

            # 梯度更新
            if accumulate and (cnt_train + 1) % a_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            if not accumulate:
                optimizer.step()
            if use_dtp:
                good_tnews_nb, good_ocnli_nb, good_ocemotion_nb, total_tnews_nb, total_ocnli_nb, total_ocemotion_nb = criterion.correct_cnt_each(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                tmp_good = sum([good_tnews_nb, good_ocnli_nb, good_ocemotion_nb])
                tmp_total = sum([total_tnews_nb, total_ocnli_nb, total_ocemotion_nb])
                train_ocemotion_correct += good_ocemotion_nb
                train_ocnli_correct += good_ocnli_nb
                train_tnews_correct += good_tnews_nb
            else:
                tmp_good, tmp_total = criterion.correct_cnt(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
            train_correct += tmp_good
            train_total += tmp_total
            p, g = criterion.collect_pred_and_gold(ocnli_pred, ocnli_gold)
            train_ocnli_pred_list += p
            train_ocnli_gold_list += g
            p, g = criterion.collect_pred_and_gold(ocemotion_pred, ocemotion_gold)
            train_ocemotion_pred_list += p
            train_ocemotion_gold_list += g
            p, g = criterion.collect_pred_and_gold(tnews_pred, tnews_gold)
            train_tnews_pred_list += p
            train_tnews_gold_list += g
            cnt_train += 1

            torch.cuda.empty_cache()
            if (cnt_train + 1) % print_freq == 0:
                print('Batch [{}]: accuracy: {}, loss: {}'.format(cnt_train+1, train_correct / train_total, train_loss / cnt_train))
        
        if accumulate:
            optimizer.step()
        optimizer.zero_grad()

        train_ocnli_f1 = get_f1(train_ocnli_gold_list, train_ocnli_pred_list)
        train_ocemotion_f1 = get_f1(train_ocemotion_gold_list, train_ocemotion_pred_list)
        train_tnews_f1 = get_f1(train_tnews_gold_list, train_tnews_pred_list)
        train_avg_f1 = (train_ocnli_f1 + train_ocemotion_f1 + train_tnews_f1) / 3

        print('---------------- training info --------------------')
        print('Epoch [{}/{}]: train average f1: {}'.format(epoch, epochs, train_avg_f1))
        print('ocnli average f1:', train_ocnli_f1)
        print('ocemotion average f1:', train_ocemotion_f1)
        print('tnews average f1:', train_tnews_f1)

        # print(epoch, 'the epoch train average f1 is:', train_avg_f1)
        # print(epoch, 'the epoch train ocnli is below:')
        # print_result(train_ocnli_gold_list, train_ocnli_pred_list)
        # print(epoch, 'the epoch train ocemotion is below:')
        # print_result(train_ocemotion_gold_list, train_ocemotion_pred_list)
        # print(epoch, 'the epoch train tnews is below:')
        # print_result(train_tnews_gold_list, train_tnews_pred_list)
        
        train_data_generator.reset()
        
        # 验证
        model.eval()
        dev_loss = 0.0
        dev_total = 0
        dev_correct = 0
        dev_ocnli_correct = 0
        dev_ocemotion_correct = 0
        dev_tnews_correct = 0
        dev_ocnli_pred_list = []
        dev_ocnli_gold_list = []
        dev_ocemotion_pred_list = []
        dev_ocemotion_gold_list = []
        dev_tnews_pred_list = []
        dev_tnews_gold_list = []
        cnt_dev = 0
        with torch.no_grad():
            while True:
                raw_data = dev_data_generator.get_next_batch(batchSize)
                if raw_data == None:
                    break
                data = dict()
                data['input_ids'] = raw_data['input_ids']
                data['token_type_ids'] = raw_data['token_type_ids']
                data['attention_mask'] = raw_data['attention_mask']
                data['ocnli_ids'] = raw_data['ocnli_ids']
                data['ocemotion_ids'] = raw_data['ocemotion_ids']
                data['tnews_ids'] = raw_data['tnews_ids']
                tnews_gold = raw_data['tnews_gold']
                ocnli_gold = raw_data['ocnli_gold']
                ocemotion_gold = raw_data['ocemotion_gold']
                ocnli_pred, ocemotion_pred, tnews_pred = model(**data)
                if use_dtp:
                    tnews_kpi = 0.1 if len(dev_tnews_pred_list) == 0 else dev_tnews_correct / len(
                        dev_tnews_pred_list)
                    ocnli_kpi = 0.1 if len(dev_ocnli_pred_list) == 0 else dev_ocnli_correct / len(
                        dev_ocnli_pred_list)
                    ocemotion_kpi = 0.1 if len(dev_ocemotion_pred_list) == 0 else dev_ocemotion_correct / len(
                        dev_ocemotion_pred_list)
                    current_loss = criterion.compute_dtp(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold,
                                                           ocnli_gold,
                                                           ocemotion_gold, tnews_kpi, ocnli_kpi, ocemotion_kpi)
                else:
                    current_loss = criterion.compute(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                dev_loss += current_loss.item()
                if use_dtp:
                    good_tnews_nb, good_ocnli_nb, good_ocemotion_nb, total_tnews_nb, total_ocnli_nb, total_ocemotion_nb = criterion.correct_cnt_each(
                        tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                    tmp_good += sum([good_tnews_nb, good_ocnli_nb, good_ocemotion_nb])
                    tmp_total += sum([total_tnews_nb, total_ocnli_nb, total_ocemotion_nb])
                    dev_ocemotion_correct += good_ocemotion_nb
                    dev_ocnli_correct += good_ocnli_nb
                    dev_tnews_correct += good_tnews_nb
                else:
                    tmp_good, tmp_total = criterion.correct_cnt(tnews_pred, ocnli_pred, ocemotion_pred, tnews_gold, ocnli_gold, ocemotion_gold)
                dev_correct += tmp_good
                dev_total += tmp_total
                p, g = criterion.collect_pred_and_gold(ocnli_pred, ocnli_gold)
                dev_ocnli_pred_list += p
                dev_ocnli_gold_list += g
                p, g = criterion.collect_pred_and_gold(ocemotion_pred, ocemotion_gold)
                dev_ocemotion_pred_list += p
                dev_ocemotion_gold_list += g
                p, g = criterion.collect_pred_and_gold(tnews_pred, tnews_gold)
                dev_tnews_pred_list += p
                dev_tnews_gold_list += g
                cnt_dev += 1

                torch.cuda.empty_cache()
                #if (cnt_dev + 1) % 1000 == 0:
                #    print('[', cnt_dev + 1, '- th batch : dev acc is:', dev_correct / dev_total, '; dev loss is:', dev_loss / cnt_dev, ']')
            dev_ocnli_f1 = get_f1(dev_ocnli_gold_list, dev_ocnli_pred_list)
            dev_ocemotion_f1 = get_f1(dev_ocemotion_gold_list, dev_ocemotion_pred_list)
            dev_tnews_f1 = get_f1(dev_tnews_gold_list, dev_tnews_pred_list)
            dev_avg_f1 = (dev_ocnli_f1 + dev_ocemotion_f1 + dev_tnews_f1) / 3


            print('Epoch [{}/{}]: dev average f1: {}'.format(epoch, epochs, dev_avg_f1))
            print('ocnli average f1:', dev_ocnli_f1)
            print('ocemotion average f1:', dev_ocemotion_f1)
            print('tnews average f1:', dev_tnews_f1)
            print('---------------------------------------------------')

            # print(epoch, 'th epoch dev average f1 is:', dev_avg_f1)
            # print(epoch, 'th epoch dev ocnli is below:')
            # print_result(dev_ocnli_gold_list, dev_ocnli_pred_list)
            # print(epoch, 'th epoch dev ocemotion is below:')
            # print_result(dev_ocemotion_gold_list, dev_ocemotion_pred_list)
            # print(epoch, 'th epoch dev tnews is below:')
            # print_result(dev_tnews_gold_list, dev_tnews_pred_list)

            dev_data_generator.reset()
            
            # record score to text
            with open('score_record.txt', 'a+') as f:
                line0 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n'
                line1 = 'Epoch ' + str(epoch) + '\n'
                line2 = 'train set f1: ' + str(train_avg_f1) + '\n'
                line3 = 'ocnli: ' + str(train_ocnli_f1) + ' ocemotion: ' + str(train_ocemotion_f1) + ' tnews: ' + str(train_tnews_f1) + '\n'
                line4 = 'dev set f1:' + str(dev_avg_f1) + '\n'
                line5 = 'ocnli: ' + str(dev_ocnli_f1) + ' ocemotion: ' + str(dev_ocemotion_f1) +  ' tnews: ' + str(dev_tnews_f1) + '\n\n'
                f.write(line0 + line1 + line2 + line3 + line4 + line5)

            if dev_avg_f1 > best_dev_f1:
                best_dev_f1 = dev_avg_f1
                best_epoch = epoch
                torch.save(model, finetuning_model)
            print('-------------------- new score --------------------')
            print('The best epoch is: {}, with  the best f1 is: {}'.format(best_epoch, best_dev_f1))
            print('---------------------------------------------------')


    print('-------------------- finish training --------------------')
    print('The best epoch is: {}, with the best f1 is: {}'.format(best_epoch, best_dev_f1))
    