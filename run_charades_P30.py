import os

# renew label -> train model -> test model -> ...
gpu_idx = 0
task,  P, THRESOLD, = "charades", 30, 5
for I in range(1, 10):
    SUFFIX = "P{}_T{}_RE{}".format(P, THRESOLD, I)

    # ----------------- RENEW LABEL -----------------
    renew_cmd = "python update_label_charades_P.py {} {} {} {}".format(task, P, THRESOLD, I)
    print(renew_cmd)
    os.system(renew_cmd)
    print("----------------- RENEW LABEL -----------------\n\n")


    train_cmd = "python  main.py --task {} --max_pos_len 64 --char_dim 50 --suffix {} --gpu_idx {} --epochs 50".format(task, SUFFIX, gpu_idx)
    os.system("rm ./data_pkl/{}_i3d_64_{}.pkl".format(task, SUFFIX))
    print(train_cmd)
    os.system(train_cmd)
    print("----------------- TRAIN MODEL ----------------- \n\n")

    test_cmd = train_cmd + " --mode test_save"
    print(test_cmd)
    os.system(test_cmd)
    print("----------------- TEST MODEL -----------------\n\n")
