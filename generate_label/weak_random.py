import json
import random
import sys
import os
random.seed(0)

def random_interval(se, proportion):
    assert 0.0 <= proportion <= 1.0, "proportion needs less 1.0"
    s, e = se
    duration_ = proportion*(e - s)
    e_random = e - duration_
    s_ = s + (e_random-s) * random.random()
    e_ = s_ + duration_

    assert s_ >= s and e_ <= e, "Error, random_interval {} {} {} {}".format(s, s_, e, e_)
    return [s_, e_]


def weak_data(old_path, new_path, p):
    os.makedirs(os.path.split(new_path)[0], exist_ok=True)
    with open(old_path,'r')as f:
        data = json.load(f)

    new_data = []
    for sample in data:
        vid, duration, se, sent = sample[:4]
        se_new = random_interval(se, p)
        
        record = [vid, duration, se_new, sent]
        new_data.append(record)

    with open(new_path, mode='w') as f:
        json.dump(new_data, f)



if __name__ == '__main__':
    old_path, new_path, p = sys.argv[1:4]
    p = float(p) / 100
    print(p, "|", old_path, "->", new_path)
    weak_data(old_path, new_path, p)
    print("Done!")