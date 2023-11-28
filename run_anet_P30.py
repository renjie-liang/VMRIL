import os

# renew label -> train model -> test model -> ...
gpu_idx = 1
task, P, THRESOLD = "anet", 30, 5
for I in range(1, 2):
    SUFFIX = "P{}_T{}_RE{}".format(P, THRESOLD, I)

    # ----------------- RENEW LABEL -----------------
    renew_cmd = "python update_label_anet_P.py {} {} {} {}".format(task, P, THRESOLD, I)
    print(renew_cmd)
    # os.system(renew_cmd)
    print("----------------- RENEW LABEL -----------------\n\n")

    train_cmd = "python  main.py --task {} --max_pos_len 100 --char_dim 100 --suffix {} --gpu_idx {} --epochs 30".format(task, SUFFIX, gpu_idx)
    os.system("rm ./data_pkl/{}_i3d_100_{}.pkl".format(task, SUFFIX))
    print(train_cmd)
    os.system(train_cmd)
    print("----------------- TRAIN MODEL ----------------- \n\n")

    test_cmd = train_cmd + " --mode test_save"
    print(test_cmd)
    os.system(test_cmd)
    print("----------------- TEST MODEL -----------------\n\n")
