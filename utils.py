from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report, f1_score
from transformers import BertModel, BertTokenizer

# 官方评分loss函数
def get_f1(l_t, l_p):
    marco_f1_score = f1_score(l_t, l_p, average='macro')
    return marco_f1_score

# 输出结果各项指标
def print_result(l_t, l_p):
    marco_f1_score = f1_score(l_t, l_p, average='macro')
    print(marco_f1_score)
    print(f"{'confusion_matrix':*^80}")
    print(confusion_matrix(l_t, l_p, ))
    print(f"{'classification_report':*^80}")
    print(classification_report(l_t, l_p, ))


# 类型中文名
def get_task_chinese(task_type):
    if task_type == 'ocnli':
        return '(中文原版自然语言推理)'
    elif task_type == 'ocemotion':
        return '(中文情感分类)'
    else:
        return '(今日头条新闻标题分类)'