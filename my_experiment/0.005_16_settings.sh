mkdir -p log/budget_0.005/neutral
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2000000000  --budget 0.005 --restarts 8 --ensemble 1 --name budget0.005 --table_path tables/budget_0.005/ >& log/budget_0.005/neutral/2000000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2100000000  --budget 0.005 --restarts 8 --ensemble 1 --name budget0.005 --table_path tables/budget_0.005/ >& log/budget_0.005/neutral/2100000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2110000000  --budget 0.005 --restarts 8 --ensemble 1 --name budget0.005 --table_path tables/budget_0.005/ >& log/budget_0.005/neutral/2110000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111000000  --budget 0.005 --restarts 8 --ensemble 1 --name budget0.005 --table_path tables/budget_0.005/ >& log/budget_0.005/neutral/2111000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111100000  --budget 0.005 --restarts 8 --ensemble 1 --name budget0.005 --table_path tables/budget_0.005/ >& log/budget_0.005/neutral/2111100000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111110000  --budget 0.005 --restarts 8 --ensemble 1 --name budget0.005 --table_path tables/budget_0.005/ >& log/budget_0.005/neutral/2111110000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111000  --budget 0.005 --restarts 8 --ensemble 1 --name budget0.005 --table_path tables/budget_0.005/ >& log/budget_0.005/neutral/2111111000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111100  --budget 0.005 --restarts 8 --ensemble 1 --name budget0.005 --table_path tables/budget_0.005/ >& log/budget_0.005/neutral/2111111100.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111110  --budget 0.005 --restarts 8 --ensemble 1 --name budget0.005 --table_path tables/budget_0.005/ >& log/budget_0.005/neutral/2111111110.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111111  --budget 0.005 --restarts 8 --ensemble 1 --name budget0.005 --table_path tables/budget_0.005/ >& log/budget_0.005/neutral/2111111111.log

mkdir -p log/budget_0.005/0.9_1e-4
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2000000000  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_budget0.005 --table_path tables/budget_0.005/0.9_1e-4/ --wolfe 0.9 1e-4  >& log/budget_0.005/0.9_1e-4/2000000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2100000000  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_budget0.005 --table_path tables/budget_0.005/0.9_1e-4/ --wolfe 0.9 1e-4  >& log/budget_0.005/0.9_1e-4/2100000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2110000000  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_budget0.005 --table_path tables/budget_0.005/0.9_1e-4/ --wolfe 0.9 1e-4  >& log/budget_0.005/0.9_1e-4/2110000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111000000  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_budget0.005 --table_path tables/budget_0.005/0.9_1e-4/ --wolfe 0.9 1e-4  >& log/budget_0.005/0.9_1e-4/2111000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111100000  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_budget0.005 --table_path tables/budget_0.005/0.9_1e-4/ --wolfe 0.9 1e-4  >& log/budget_0.005/0.9_1e-4/2111100000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111110000  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_budget0.005 --table_path tables/budget_0.005/0.9_1e-4/ --wolfe 0.9 1e-4  >& log/budget_0.005/0.9_1e-4/2111110000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111000  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_budget0.005 --table_path tables/budget_0.005/0.9_1e-4/ --wolfe 0.9 1e-4  >& log/budget_0.005/0.9_1e-4/2111111000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111100  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_budget0.005 --table_path tables/budget_0.005/0.9_1e-4/ --wolfe 0.9 1e-4  >& log/budget_0.005/0.9_1e-4/2111111100.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111110  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_budget0.005 --table_path tables/budget_0.005/0.9_1e-4/ --wolfe 0.9 1e-4  >& log/budget_0.005/0.9_1e-4/2111111110.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111111  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_budget0.005 --table_path tables/budget_0.005/0.9_1e-4/ --wolfe 0.9 1e-4  >& log/budget_0.005/0.9_1e-4/2111111111.log

mkdir -p log/budget_0.005/worstsharp
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2000000000  --budget 0.005 --restarts 8 --ensemble 1 --name worstsharp_budget0.005 --target_criterion worstsharp --table_path tables/budget_0.005/worstsharp/  >& log/budget_0.005/worstsharp/2000000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2100000000  --budget 0.005 --restarts 8 --ensemble 1 --name worstsharp_budget0.005 --target_criterion worstsharp --table_path tables/budget_0.005/worstsharp/  >& log/budget_0.005/worstsharp/2100000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2110000000  --budget 0.005 --restarts 8 --ensemble 1 --name worstsharp_budget0.005 --target_criterion worstsharp --table_path tables/budget_0.005/worstsharp/  >& log/budget_0.005/worstsharp/2110000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111000000  --budget 0.005 --restarts 8 --ensemble 1 --name worstsharp_budget0.005 --target_criterion worstsharp --table_path tables/budget_0.005/worstsharp/  >& log/budget_0.005/worstsharp/2111000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111100000  --budget 0.005 --restarts 8 --ensemble 1 --name worstsharp_budget0.005 --target_criterion worstsharp --table_path tables/budget_0.005/worstsharp/  >& log/budget_0.005/worstsharp/2111100000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111110000  --budget 0.005 --restarts 8 --ensemble 1 --name worstsharp_budget0.005 --target_criterion worstsharp --table_path tables/budget_0.005/worstsharp/  >& log/budget_0.005/worstsharp/2111110000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111000  --budget 0.005 --restarts 8 --ensemble 1 --name worstsharp_budget0.005 --target_criterion worstsharp --table_path tables/budget_0.005/worstsharp/  >& log/budget_0.005/worstsharp/2111111000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111100  --budget 0.005 --restarts 8 --ensemble 1 --name worstsharp_budget0.005 --target_criterion worstsharp --table_path tables/budget_0.005/worstsharp/  >& log/budget_0.005/worstsharp/2111111100.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111110  --budget 0.005 --restarts 8 --ensemble 1 --name worstsharp_budget0.005 --target_criterion worstsharp --table_path tables/budget_0.005/worstsharp/  >& log/budget_0.005/worstsharp/2111111110.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111111  --budget 0.005 --restarts 8 --ensemble 1 --name worstsharp_budget0.005 --target_criterion worstsharp --table_path tables/budget_0.005/worstsharp/  >& log/budget_0.005/worstsharp/2111111111.log

mkdir -p log/budget_0.005/0.9_1e-4_worstsharp
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2000000000  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_worstsharp_budget0.005 --table_path tables/budget_0.005/0.9_1e-4_worstsharp/ --wolfe 0.9 1e-4  --target_criterion worstsharp >& log/budget_0.005/0.9_1e-4_worstsharp/2000000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2100000000  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_worstsharp_budget0.005 --table_path tables/budget_0.005/0.9_1e-4_worstsharp/ --wolfe 0.9 1e-4  --target_criterion worstsharp >& log/budget_0.005/0.9_1e-4_worstsharp/2100000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2110000000  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_worstsharp_budget0.005 --table_path tables/budget_0.005/0.9_1e-4_worstsharp/ --wolfe 0.9 1e-4  --target_criterion worstsharp >& log/budget_0.005/0.9_1e-4_worstsharp/2110000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111000000  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_worstsharp_budget0.005 --table_path tables/budget_0.005/0.9_1e-4_worstsharp/ --wolfe 0.9 1e-4  --target_criterion worstsharp >& log/budget_0.005/0.9_1e-4_worstsharp/2111000000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111100000  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_worstsharp_budget0.005 --table_path tables/budget_0.005/0.9_1e-4_worstsharp/ --wolfe 0.9 1e-4  --target_criterion worstsharp >& log/budget_0.005/0.9_1e-4_worstsharp/2111100000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111110000  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_worstsharp_budget0.005 --table_path tables/budget_0.005/0.9_1e-4_worstsharp/ --wolfe 0.9 1e-4  --target_criterion worstsharp >& log/budget_0.005/0.9_1e-4_worstsharp/2111110000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111000  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_worstsharp_budget0.005 --table_path tables/budget_0.005/0.9_1e-4_worstsharp/ --wolfe 0.9 1e-4  --target_criterion worstsharp >& log/budget_0.005/0.9_1e-4_worstsharp/2111111000.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111100  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_worstsharp_budget0.005 --table_path tables/budget_0.005/0.9_1e-4_worstsharp/ --wolfe 0.9 1e-4  --target_criterion worstsharp >& log/budget_0.005/0.9_1e-4_worstsharp/2111111100.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111110  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_worstsharp_budget0.005 --table_path tables/budget_0.005/0.9_1e-4_worstsharp/ --wolfe 0.9 1e-4  --target_criterion worstsharp >& log/budget_0.005/0.9_1e-4_worstsharp/2111111110.log
python brew_poison.py  --net ResNet18 --vruns 8 --poisonkey 2111111111  --budget 0.005 --restarts 8 --ensemble 1 --name 0.9_1e-4_worstsharp_budget0.005 --table_path tables/budget_0.005/0.9_1e-4_worstsharp/ --wolfe 0.9 1e-4  --target_criterion worstsharp >& log/budget_0.005/0.9_1e-4_worstsharp/2111111111.log