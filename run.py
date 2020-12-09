from train import train

pretrained_model = './pretrain_model'
tokenizer_model = './pretrain_model'
train(epochs=20, batchSize=16, device='cuda', lr=0.0001, use_dtp=True,
                    pretrained_model=pretrained_model, tokenizer_model=tokenizer_model, weighted_loss=True)

train(epochs=20, batchSize=16, device='cuda', lr=0.0001, use_dtp=True,
                    load_saved=True, finetuning_model='./saved_best.pt', weighted_loss=True)
