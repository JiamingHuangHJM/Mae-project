# Mae-project


 * To pretrain the model, run
 ```python
 python pretrain_cifar.py

 * To finetune the model using pre-trained checkpoints:
 ```python
 python finetune_cifar.py 
 
 Meanwhile, specify your checkpoint's path  `PRETRAINED_PATH`. If you want to train a ViT classifier from scrath, 
 just leave the `PRETRAINED_PATH` as None. 

 