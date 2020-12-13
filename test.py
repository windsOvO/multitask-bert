import time

epoch = train_ocnli_f1 = train_ocemotion_f1 = train_tnews_f1 = 1
epoch = dev_ocnli_f1 =dev_ocemotion_f1 = dev_tnews_f1 = 0

with open('score_record.txt', 'a+') as f:
    line0 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n'
    line1 = 'Epoch ' + str(epoch) + '\n'
    line2 = 'train set f1:' + '\n'
    line3 = 'ocnli: ' + str(train_ocnli_f1) + ' ocemotion: ' + str(train_ocemotion_f1) + ' tnews: ' + str(train_tnews_f1) + '\n'
    line4 = 'dev set f1:' + '\n'
    line5 = 'ocnli: ' + str(dev_ocnli_f1) + ' ocemotion: ' + str(dev_ocemotion_f1) +  ' tnews: ' + str(dev_tnews_f1) + '\n\n'
    f.write(line0 + line1 + line2 + line3 + line4 + line5)