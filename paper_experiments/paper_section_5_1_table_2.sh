mkdir -p logs/paper_section_5_1_table_2/comparison_figure3a_ours
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2000000000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2000000000.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2100000000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2100000000.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2110000000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2110000000.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111000000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2111000000.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111100000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2111100000.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111110000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours    >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2111110000.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111111000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours   >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2111111000.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111111100 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours  >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2111111100.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111111110 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2111111110.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111111111 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2111111111.log
mkdir -p logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2000000000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2000000000.log
python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2100000000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2100000000.log 
python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2110000000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128  >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2110000000.log
python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111000000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2111000000.log 
python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111100000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2111100000.log
python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111110000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2111110000.log
python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111111000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128  >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2111111000.log
python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111111100 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2111111100.log
python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111111110 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2111111110.log 
python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111111111 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2111111111.log
#mkdir -p logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2000000000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2000000000.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2100000000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2100000000.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2110000000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2110000000.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111000000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2111000000.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111100000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2111100000.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111110000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2111110000.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111111000 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2111111000.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111111100 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2111111100.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111111110 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2111111110.log
#python brew_poison.py  --net ConvNet64 --vruns 8 --poisonkey 2111111111 --budget 0.01 --eps 32 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2111111111.log
#mkdir -p logs/paper_section_5_1_table_2/comparison_figure3a_ours
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2000000000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2000000000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2100000000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2100000000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2110000000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2110000000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111000000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2111000000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111100000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2111100000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111110000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours   >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2111110000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2111111000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111100 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2111111100.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111110 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2111111110.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111111 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_ours --table_path tables/comparison_figure3a_ours     >& logs/paper_section_5_1_table_2/comparison_figure3a_ours/2111111111.log
#mkdir -p logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2000000000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2000000000.logn
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2100000000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2100000000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2110000000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2110000000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111000000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2111000000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111100000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2111100000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111110000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2111110000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2111111000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111100 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2111111100.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111110 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2111111110.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111111 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_poisonfrogsOurs --recipe poison-frogs --pbatch 128 >& logs/paper_section_5_1_table_2/comparison_figure3a_poisonfrogsOurs/2111111111.log
#mkdir -p logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2000000000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2000000000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2100000000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2100000000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2110000000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2110000000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111000000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2111000000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111100000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2111100000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111110000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2111110000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111000 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2111111000.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111100 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2111111100.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111110 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2111111110.log
#python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111111 --budget 0.01 --eps 16 --restarts 8 --ensemble 1 --wandb --table_path tables/comparison_figure3a_ours --name comparison_figure3a_metapoisonOurs --recipe metapoison --clean_grad >& logs/paper_section_5_1_table_2/comparison_figure3a_metapoisonOurs/2111111111.log
