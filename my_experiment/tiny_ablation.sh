mkdir -p log/ablation/resnet
mkdir -p log/ablation/vit
mkdir -p log/ablation/vgg16

python brew_poison.py  --net ResNet34 --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2000000000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet/2000000000.log
python brew_poison.py  --net vit      --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2000000000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vit       --table_path tables/ablation  >& log/ablation/vit/2000000000.log
python brew_poison.py  --net VGG16    --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2000000000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vgg16     --table_path tables/ablation  >& log/ablation/vgg16/2000000000.log

python brew_poison.py  --net ResNet34 --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2100000000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet/2100000000.log
python brew_poison.py  --net vit      --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2100000000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vit       --table_path tables/ablation  >& log/ablation/vit/2100000000.log
python brew_poison.py  --net VGG16    --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2100000000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vgg16     --table_path tables/ablation  >& log/ablation/vgg16/2100000000.log


python brew_poison.py  --net ResNet34 --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2110000000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet/2110000000.log
python brew_poison.py  --net vit      --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2110000000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vit       --table_path tables/ablation  >& log/ablation/vit/2110000000.log
python brew_poison.py  --net VGG16    --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2110000000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vgg16     --table_path tables/ablation  >& log/ablation/vgg16/2110000000.log


python brew_poison.py  --net ResNet34 --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111000000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet/2111000000.log
python brew_poison.py  --net vit      --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111000000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vit       --table_path tables/ablation  >& log/ablation/vit/2111000000.log
python brew_poison.py  --net VGG16    --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111000000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vgg16     --table_path tables/ablation  >& log/ablation/vgg16/2111000000.log


python brew_poison.py  --net ResNet34 --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111100000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet/2111100000.log
python brew_poison.py  --net vit      --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111100000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vit       --table_path tables/ablation  >& log/ablation/vit/2111100000.log
python brew_poison.py  --net vit      --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111110000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vit       --table_path tables/ablation  >& log/ablation/vit/2111110000.log


python brew_poison.py  --net ResNet34 --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111110000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet/2111110000.log
python brew_poison.py  --net vit      --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111110000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vit       --table_path tables/ablation  >& log/ablation/vit/2111111000.log
python brew_poison.py  --net VGG16    --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111110000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vgg16     --table_path tables/ablation  >& log/ablation/vgg16/2111110000.log


python brew_poison.py  --net ResNet34 --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111111000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet/2111111000.log
python brew_poison.py  --net vit      --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111111000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vit       --table_path tables/ablation  >& log/ablation/vit/2111111000.log
python brew_poison.py  --net VGG16    --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111111000  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vgg16     --table_path tables/ablation  >& log/ablation/vgg16/2111111000.log


python brew_poison.py  --net ResNet34 --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111111100  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet/2111111100.log
python brew_poison.py  --net vit      --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111111100  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vit       --table_path tables/ablation  >& log/ablation/vit/2111111100.log
python brew_poison.py  --net VGG16    --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111111100  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vgg16     --table_path tables/ablation  >& log/ablation/vgg16/2111111100.log


python brew_poison.py  --net ResNet34 --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111111110  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet/2111111110.log
python brew_poison.py  --net vit      --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111111110  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vit       --table_path tables/ablation  >& log/ablation/vit/2111111110.log
python brew_poison.py  --net VGG16    --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111111110  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vgg16     --table_path tables/ablation  >& log/ablation/vgg16/2111111110.log


python brew_poison.py  --net ResNet34 --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111111111  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_resnet34  --table_path tables/ablation  >& log/ablation/resnet/2111111111.log
python brew_poison.py  --net vit      --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111111111  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vit       --table_path tables/ablation  >& log/ablation/vit/2111111111.log
python brew_poison.py  --net VGG16    --dataset TinyImageNet --data_path sets/tiny-imagenet-200  --vruns 8 --poisonkey 2111111111  --budget 0.01 --eps 8 --restarts 8 --ensemble 1 --name ablation_vgg16     --table_path tables/ablation  >& log/ablation/vgg16/2111111111.log



