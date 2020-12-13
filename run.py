from train import train

pretrained_model = './pretrain_model'
tokenizer_model = './pretrain_model'
train(epochs=20, print_freq=100, batchSize=4, device='cuda', lr=0.0001, a_step=16, use_dtp=True,
                    pretrained_model=pretrained_model, tokenizer_model=tokenizer_model, weighted_loss=True)

# train(epochs=5, batchSize=16, device='cuda', lr=0.0001, a_step=32, use_dtp=True,
#                     load_saved=True, finetuning_model='./models/saved_best.pt', weighted_loss=True)