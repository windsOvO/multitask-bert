from net import Net
import json
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
# from utils import get_task_chinese


def test_csv_to_json():
    for e in ['TNEWS', 'OCNLI', 'OCEMOTION']:
        with open('./datasets/' + e + '/test.csv') as fr:
            with open('./datasets/' + e + '/test.json', 'w') as fw:
                json_dict = dict()
                for line in fr:
                    tmp_list = line.strip().split('\t')
                    json_dict[tmp_list[0]] = dict()
                    json_dict[tmp_list[0]]['s1'] = tmp_list[1]
                    if e == 'OCNLI':
                        json_dict[tmp_list[0]]['s2'] = tmp_list[2]
                fw.write(json.dumps(json_dict))
                
def inference_warpper(tokenizer_model, device):
    ocnli_test = dict()
    with open('./datasets/OCNLI/test.json') as f:
        for line in f:
            ocnli_test = json.loads(line)
            break
        
    ocemotion_test = dict()
    with open('./datasets/OCEMOTION/test.json') as f:
        for line in f:
            ocemotion_test = json.loads(line)
            break
        
    tnews_test = dict()
    with open('./datasets/TNEWS/test.json') as f:
        for line in f:
            tnews_test = json.loads(line)
            break
        
    label_dict = dict()
    with open('./datasets/label.json') as f:
        for line in f:
            label_dict = json.loads(line)
            break

    tokenizer = BertTokenizer.from_pretrained(tokenizer_model)   
    # gpu模型转cpu
    if device == 'cpu':
        # model = torch.load('./models/saved_best.pt', map_location='cpu')
        model = torch.load('./models/saved_best.pt')
    else:
         model = torch.load('./models/saved_best.pt')

    inference('./submission/5928/ocnli_predict.json', ocnli_test, model, tokenizer, label_dict['OCNLI'], 'ocnli', device, 64, True)
    inference('./submission/5928/ocemotion_predict.json', ocemotion_test, model, tokenizer, label_dict['OCEMOTION'], 'ocemotion', device, 64, True)
    inference('./submission/5928/tnews_predict.json', tnews_test, model, tokenizer, label_dict['TNEWS'], 'tnews', device, 64, True)
        
def inference(path, data_dict, model, tokenizer, idx2label, task_type, device='cuda', batchSize=64, max_len=512, print_result=True):
    if task_type != 'ocnli' and task_type != 'ocemotion' and task_type != 'tnews':
        print('task_type is incorrect!')
        return
    #task_chinese = get_task_chinese(task_type)
    model.to(device, non_blocking=True)
    model.eval()
    ids_list = [k for k, _ in data_dict.items()]
    next_start_ids = 0
    with torch.no_grad():
        with open(path, 'w') as f:
            while next_start_ids < len(ids_list):
                cur_ids_list = ids_list[next_start_ids: next_start_ids + batchSize]
                next_start_ids += batchSize
                sentence1 = [data_dict[idx]['s1'] for idx in cur_ids_list]
                if task_type == 'ocnli':
                    flower = tokenizer([data_dict[idx]['s1'] for idx in cur_ids_list], [data_dict[idx]['s2'] for idx in cur_ids_list], add_special_tokens=True, padding=True, return_tensors='pt')
                else:
                    flower = tokenizer([data_dict[idx]['s1'] for idx in cur_ids_list], add_special_tokens=True, padding=True, return_tensors='pt')
                input_ids = flower['input_ids'].to(device, non_blocking=True)
                token_type_ids = flower['token_type_ids'].to(device, non_blocking=True)
                attention_mask = flower['attention_mask'].to(device, non_blocking=True)
                ocnli_ids = torch.tensor([]).to(device, non_blocking=True)
                ocemotion_ids = torch.tensor([]).to(device, non_blocking=True)
                tnews_ids = torch.tensor([]).to(device, non_blocking=True)
                if task_type == 'ocnli':
                    ocnli_ids = torch.tensor([i for i in range(len(cur_ids_list))]).to(device, non_blocking=True)
                elif task_type == 'ocemotion':
                    ocemotion_ids = torch.tensor([i for i in range(len(cur_ids_list))]).to(device, non_blocking=True)
                else:
                    tnews_ids = torch.tensor([i for i in range(len(cur_ids_list))]).to(device, non_blocking=True)
                ocnli_out, ocemotion_out, tnews_out = model(input_ids, ocnli_ids, ocemotion_ids, tnews_ids, token_type_ids, attention_mask)
                if task_type == 'ocnli':
                    pred = torch.argmax(ocnli_out, axis=1)
                elif task_type == 'ocemotion':
                    pred = torch.argmax(ocemotion_out, axis=1)
                else:
                    pred = torch.argmax(tnews_out, axis=1)
                pred_final = [idx2label[e] for e in np.array(pred.cpu()).tolist()]

                torch.cuda.empty_cache()
                for i, idx in enumerate(cur_ids_list):
                    if print_result:
                        print_str = '[ ' + task_type + ' : ' + 'sentence one: ' + data_dict[idx]['s1']
                        if task_type == 'ocnli':
                            print_str += '; sentence two: ' + data_dict[idx]['s2']
                        print_str += '; result: ' + pred_final[i] + ' ]'
                        print(print_str)
                    single_result_dict = dict()
                    single_result_dict['id'] = idx
                    single_result_dict['label'] = pred_final[i]
                    f.write(json.dumps(single_result_dict, ensure_ascii=False))
                    if not (next_start_ids >= len(ids_list) and i == len(cur_ids_list) - 1):
                        f.write('\n')
                        
if __name__ == '__main__':
    # test_csv_to_json()
    print('------------------- start predicting -------------------')
    inference_warpper(tokenizer_model='./pretrain_model', device='cpu')
    
    
    
    