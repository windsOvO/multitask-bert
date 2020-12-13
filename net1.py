import torch
from torch import nn
from transformers import BertModel


class Net(nn.Module):
    def __init__(self, bert_model):
        super(Net, self).__init__()
        self.bert = bert_model # bert预训练模型层

        self.atten_layer = nn.Linear(768, 16) # soft attention
        self.softmax_d1 = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2) # p=0.2指该层参数在每次迭代训练时会随机有0.2的可能性被丢弃，即不参与训练

        self.OCNLI_layer = nn.Linear(768, 16 * 3)
        self.OCEMOTION_layer = nn.Linear(768, 16 * 7)
        self.TNEWS_layer = nn.Linear(768, 16 * 15)


    def forward(self, input_ids, ocnli_ids, ocemotion_ids, tnews_ids, token_type_ids=None, attention_mask=None):
        # reference to https://huggingface.co/transformers/model_doc/bert.html#bertmodel
        # position_ids is created automatically
        # return last_hidden_state: torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        # torch.squeeze(): eliminate the dimension 1(2nd)
        # [0] -> select last_hidden_state return's item
        cls_emb = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0][:, 0, :].squeeze(1)
        
        # BERT模型在文本前插入一个[CLS]符号，并将该符号对应的输出向量作为整篇文本的语义表示
        # [:,0,:]中0即选择CLS符号
        # cls_emb.size() = (batchSize, 768) 768 is bert default output dimension
        # ocnli_ids = [0,1,2,3], ocemotion_ids = [4,5,6], tnews_ids = [0,1,...n] 标记这个sample属于哪个任务

        # 存在对应任务的序列
        if ocnli_ids.size()[0] > 0:
            # attention weight
            attention_score = self.atten_layer(cls_emb[ocnli_ids, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))

            # 全连接层768维->16*3维
            # contiguous把tensor变成在内存中连续分布的形式
            # view = reshape只能用于连续
            ocnli_value = self.OCNLI_layer(cls_emb[ocnli_ids, :]).contiguous().view(-1, 16, 3)

            # use attention and reduce dimension to (3,)
            ocnli_out = torch.matmul(attention_score, ocnli_value).squeeze(1)
        else:
            ocnli_out = None

        if ocemotion_ids.size()[0] > 0:
            attention_score = self.atten_layer(cls_emb[ocemotion_ids, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))
            ocemotion_value = self.OCEMOTION_layer(cls_emb[ocemotion_ids, :]).contiguous().view(-1, 16, 7)
            ocemotion_out = torch.matmul(attention_score, ocemotion_value).squeeze(1)
        else:
            ocemotion_out = None

        if tnews_ids.size()[0] > 0:
            attention_score = self.atten_layer(cls_emb[tnews_ids, :])
            attention_score = self.dropout(self.softmax_d1(attention_score).unsqueeze(1))

            tnews_value = self.TNEWS_layer(cls_emb[tnews_ids, :]).contiguous().view(-1, 16, 15)
            tnews_out = torch.matmul(attention_score, tnews_value).squeeze(1)
            
        else:
            tnews_out = None

        # out.shape = (3,) / (7,) / (15,)
        return ocnli_out, ocemotion_out, tnews_out