from utils_weak import load_json, load_pickle, save_json, miou_two_dataset

seed_path = './data/charades_P30_RE0/train.json'
old_path = './data/charades_P30_IOU5_RE6/train.json'
new_path = './data/charades_P30_IOU5_RE6_SEED/train.json'
old_data = load_json(old_path)
seed_data = load_json(seed_path)

new_data = []
for sample, seed_sample in zip(old_data, seed_data):
    vid, duration, se_old, sent = sample[:4]
    se_seed = sample[2]
    record = [vid, duration, se_old, sent, se_seed]
    new_data.append(record)

save_json(new_data, new_path)
print(len(old_data),"---->", len(new_data))