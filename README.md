# ECE595MLproject
attack.py:
generate_clean_dataset() --generate clean mnist and cifar data
train.py: train three models (resnet50, mobilenetv2, efficientnetb0) with cifar10, generate training loss graph, generate runtime accuracy graph and save model
run_code_for_training(net,loaders,net_save_path)  --train the net with input dataloader and model, output training loss and runtime validation accuracy
run_code_for_validation_direct(net, validation_data_loader)  -- generate accuracy with input validation dataloader
