# Charades
python weak_random.py ../data/charades_gt/train.json ../data/charades_P10_RE0/train.json  10
cp ../data/charades_gt/test.json ../data/charades_P10_RE0/test.json

python weak_random.py ../data/charades_gt/train.json ../data/charades_P20_RE0/train.json  20
cp ../data/charades_gt/test.json ../data/charades_P20_RE0/test.json

python weak_random.py ../data/charades_gt/train.json ../data/charades_P30_RE0/train.json  30
cp ../data/charades_gt/test.json ../data/charades_P30_RE0/test.json

python weak_random.py ../data/charades_gt/train.json ../data/charades_P50_RE0/train.json  50
cp ../data/charades_gt/test.json ../data/charades_P50_RE0/test.json

python weak_random.py ../data/charades_gt/train.json ../data/charades_P70_RE0/train.json  70
cp ../data/charades_gt/test.json ../data/charades_P70_RE0/test.json

# Anet
python weak_random.py ../data/anet_i3d_gt/train.json ../data/anet_P10_RE0/train.json  10
cp ../data/anet_i3d_gt/test.json ../data/anet_P10_RE0/test.json

python weak_random.py ../data/anet_i3d_gt/train.json ../data/anet_P30_RE0/train.json  30
cp ../data/anet_i3d_gt/test.json ../data/anet_P30_RE0/test.json

python weak_random.py ../data/anet_i3d_gt/train.json ../data/anet_P50_RE0/train.json  50
cp ../data/anet_i3d_gt/test.json ../data/anet_P50_RE0/test.json

python weak_random.py ../data/anet_i3d_gt/train.json ../data/anet_P70_RE0/train.json  70
cp ../data/anet_i3d_gt/test.json ../data/anet_P70_RE0/test.json

# Tacos
python weak_random.py ../data/tacos_gt/train.json ../data/tacos_P10_RE0/train.json  10
cp ../data/tacos_gt/test.json ../data/tacos_P10_RE0/test.json

python weak_random.py ../data/tacos_gt/train.json ../data/tacos_P30_RE0/train.json  30
cp ../data/tacos_gt/test.json ../data/tacos_P30_RE0/test.json

python weak_random.py ../data/tacos_gt/train.json ../data/tacos_P50_RE0/train.json  50
cp ../data/tacos_gt/test.json ../data/tacos_P50_RE0/test.json

python weak_random.py ../data/tacos_gt/train.json ../data/tacos_P70_RE0/train.json  70
cp ../data/tacos_gt/test.json ../data/tacos_P70_RE0/test.json



#
python weak_random_beta.py ../data/charades_gt/train.json ../data/charades_B30_RE0/train.json  30
cp ../data/charades_gt/test.json ../data/charades_B30_RE0/test.json
python weak_random_beta.py ../data/charades_gt/train.json ../data/charades_B20_RE0/train.json  20
cp ../data/charades_gt/test.json ../data/charades_B20_RE0/test.json
python weak_random_beta.py ../data/charades_gt/train.json ../data/charades_B10_RE0/train.json  10
cp ../data/charades_gt/test.json ../data/charades_B10_RE0/test.json


python weak_random_beta.py ../data/anet_gt/train.json ../data/anet_B20_RE0/train.json  20
cp ../data/anet_gt/test.json ../data/anet_B20_RE0/test.json


python weak_random_beta.py ../data/anet_gt/train.json ../data/anet_B30_RE0/train.json  30
cp ../data/anet_gt/test.json ../data/anet_B30_RE0/test.json



python weak_random_beta.py ../data/tacos_gt/train.json ../data/tacos_B30_RE0/train.json  30
cp ../data/tacos_gt/test.json ../data/tacos_B30_RE0/test.json
