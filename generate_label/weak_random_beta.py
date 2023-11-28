import json
import random
import sys
import os
import numpy as np
random.seed(0)
import matplotlib.pyplot as plt

def random_interval(se, proportion):
    assert 0.0 <= proportion <= 1.0, "proportion needs less 1.0"
    s, e = se
    duration_ = proportion*(e - s)
    e_random = e - duration_
    s_ = s + (e_random-s) * random.random()
    e_ = s_ + duration_

    assert s_ >= s and e_ <= e, "Error, random_interval {} {} {} {}".format(s, s_, e, e_)
    return [s_, e_]


def weak_data(old_path, new_path, a, b):
    os.makedirs(os.path.split(new_path)[0], exist_ok=True)
    with open(old_path,'r')as f:
        data = json.load(f)

    new_data = []
    random_list = []
    for sample in data:
        vid, duration, se, sent = sample[:4]
        p = np.random.beta(a, b)
        random_list.append(p)
        se_new = random_interval(se, p)
        
        record = [vid, duration, se_new, sent]
        new_data.append(record)

    with open(new_path, mode='w') as f:
        json.dump(new_data, f)
    return random_list

BETA_DIC = {
    "30":[9, 21],
    "20":[2, 8],
    "10":[1, 9]
}

if __name__ == '__main__':
    old_path, new_path, P = sys.argv[1:4]
    a, b = BETA_DIC[P]
    np.random.seed(4)

    print(P, "|", old_path, "->", new_path)
    random_list = weak_data(old_path, new_path, a, b)
    print(len(random_list))
    mean_theory = a / (a + b)
    var_theory = (a*b) / ((a+b)**2 * (a+b+1))
    print("theory mean: {:.4f}, var: {:.4f}".format(mean_theory,var_theory))
    print("actual mean: {:.4f}, var: {:.4f}".format(np.mean(random_list), np.var(random_list)))
    _, _, _ = plt.hist(random_list, bins=40)
    plt.savefig("../images/beta_{}".format(P))
    print("Done!")