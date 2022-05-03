# ECE595MLproject
## attack.py:
generate attack dataset
```
generate_clean_dataset()
```
generate clean mnist and cifar data
```
generate_attacked_dataset(eps)
```
generate adversarial examples with eps as perturbation
```
ensemble_prediction(model1,model2,model3,test_data,mode)
```
with mode ensemble, it returns the accuracy of prediction combination; 
with mode separate, it returns the accuracy of models without defend; 
with mode ens_vote, it returns the accuracy of majority voting
## train.py: 
train three models (resnet50, mobilenetv2, efficientnetb0) with cifar10, generate training loss graph, generate runtime accuracy graph and save model
```
run_code_for_training(net,loaders,net_save_path)  
```
train the net with input dataloader and model, output training loss and runtime validation accuracy
```
run_code_for_validation_direct(net, validation_data_loader)
```
generate accuracy with input validation dataloader
