mkdir -p log/ablation/resnet34/
mkdir -p log/ablation/vgg16/
mkdir -p log/ablation/ResNet18/




python brew_poison.py  --net ResNet34 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2100000000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet34/2100000000.log
#python brew_poison.py  --net VGG16 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2100000000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_vgg16  --table_path tables/ablation  >& log/ablation/vgg16/2100000000.log
python brew_poison.py  --net ResNet18 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2100000000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_ResNet18  --table_path tables/ablation  >& log/ablation/ResNet18/2100000000.log

python brew_poison.py  --net ResNet34 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2110000000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet34/2110000000.log
#python brew_poison.py  --net VGG16 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2110000000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_vgg16  --table_path tables/ablation  >& log/ablation/vgg16/2110000000.log
python brew_poison.py  --net ResNet18 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2110000000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_ResNet18  --table_path tables/ablation  >& log/ablation/ResNet18/2110000000.log

python brew_poison.py  --net ResNet34 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111000000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet34/2111000000.log
#python brew_poison.py  --net VGG16 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111000000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_vgg16  --table_path tables/ablation  >& log/ablation/vgg16/2111000000.log
python brew_poison.py  --net ResNet18 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111000000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_ResNet18  --table_path tables/ablation  >& log/ablation/ResNet18/2111000000.log

python brew_poison.py  --net ResNet34 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111100000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet34/2111100000.log
#python brew_poison.py  --net VGG16 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111100000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_vgg16  --table_path tables/ablation  >& log/ablation/vgg16/2111100000.log
python brew_poison.py  --net ResNet18 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111100000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_ResNet18  --table_path tables/ablation  >& log/ablation/ResNet18/2111100000.log

python brew_poison.py  --net ResNet34 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111110000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet34/2111110000.log
#python brew_poison.py  --net VGG16 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111110000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_vgg16  --table_path tables/ablation  >& log/ablation/vgg16/2111110000.log
python brew_poison.py  --net ResNet18 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111110000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_ResNet18  --table_path tables/ablation  >& log/ablation/ResNet18/2111110000.log

python brew_poison.py  --net ResNet34 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111111000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet34/2111111000.log
#python brew_poison.py  --net VGG16 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111111000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_vgg16  --table_path tables/ablation  >& log/ablation/vgg16/2111111000.log
python brew_poison.py  --net ResNet18 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111111000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_ResNet18  --table_path tables/ablation  >& log/ablation/ResNet18/2111111000.log

python brew_poison.py  --net ResNet34 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111111100  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet34/2111111100.log
#python brew_poison.py  --net VGG16 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111111100  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_vgg16  --table_path tables/ablation  >& log/ablation/vgg16/2111111100.log
python brew_poison.py  --net ResNet18 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111111100  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_ResNet18  --table_path tables/ablation  >& log/ablation/ResNet18/2111111100.log

python brew_poison.py  --net ResNet34 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111111110  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet34/2111111110.log
#python brew_poison.py  --net VGG16 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111111110  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_vgg16  --table_path tables/ablation  >& log/ablation/vgg16/2111111110.log
python brew_poison.py  --net ResNet18 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111111110  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_ResNet18  --table_path tables/ablation  >& log/ablation/ResNet18/2111111110.log

python brew_poison.py  --net ResNet34 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111111111  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet34/2111111111.log
#python brew_poison.py  --net VGG16 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111111111  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_vgg16  --table_path tables/ablation  >& log/ablation/vgg16/2111111111.log
python brew_poison.py  --net ResNet18 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20 --poisonkey 2111111111  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_ResNet18  --table_path tables/ablation  >& log/ablation/ResNet18/2111111111.log

python brew_poison.py  --net ResNet34 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20  --poisonkey 2000000000  --data_path datasets/tiny-imagenet-200 --budget 0.005  --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet34/2000000000.log
#python brew_poison.py  --net VGG16 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20  --poisonkey 2000000000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_vgg16  --table_path tables/ablation  >& log/ablation/vgg16/2000000000.log
python brew_poison.py  --net ResNet18 --dataset TinyImageNet  --wolfe 0.9 1e-4 --linesearch_epoch 20  --poisonkey 2000000000  --data_path datasets/tiny-imagenet-200 --budget 0.005 --eps 16 --restarts 5 --pbatch 256 --ensemble 1 --vruns 5 --optimization conservative_batch256  --name ablation_ResNet18  --table_path tables/ablation  >& log/ablation/ResNet18/2000000000.log