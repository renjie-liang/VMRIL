import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.data_utils import index_to_time
import pickle

def set_tf_config(seed, gpu_idx):
    # os environment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    # random seed
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    tf.compat.v1.random.set_random_seed(seed)


def write_tf_summary(writer, value_pairs, global_step):
    for tag, value in value_pairs:
        summ = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summ, global_step=global_step)
    writer.flush()


def calculate_iou_accuracy(ious, threshold):
    total_size = float(len(ious))
    count = 0
    for iou in ious:
        if iou >= threshold:
            count += 1
    return float(count) / total_size * 100.0


def calculate_iou(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0])
    return max(0.0, iou)

def plot_se_label(s_labels, e_labels, match_labels):
    from matplotlib import pyplot as plt
    import numpy as np
    for i in range(s_labels.shape[0]):
        plt.plot(s_labels[i])
        plt.plot(e_labels[i])
        plt.scatter(np.arange(match_labels.shape[1]), match_labels[i])
        save_path = "./imgs/charades/{}.jpg".format(i)
        print(save_path)
        plt.savefig(save_path)
        plt.cla()


def get_feed_dict(batch_data, model, lr=None, drop_rate=None, mode='train'):
    if mode == 'train':  # training
        (_, vfeats, vfeat_lens, word_ids, char_ids, s_labels, e_labels, match_labels, inner_labels) = batch_data
        # plot_se_label(s_labels, e_labels, match_labels)
        feed_dict = {model.video_inputs: vfeats, model.video_seq_len: vfeat_lens, model.word_ids: word_ids,
                     model.char_ids: char_ids, model.y1: s_labels, model.y2: e_labels, model.lr: lr,
                     model.match_labels: match_labels, model.inner_labels: inner_labels, model.drop_rate: drop_rate}
        return feed_dict
    else:  # eval
        raw_data, vfeats, vfeat_lens, word_ids, char_ids = batch_data
        feed_dict = {model.video_inputs: vfeats, model.video_seq_len: vfeat_lens, model.word_ids: word_ids,
                     model.char_ids: char_ids}
        return raw_data, feed_dict


def eval_test(sess, model, data_loader, epoch=None, global_step=None, mode="test"):
    ious = list()
    for data in tqdm(data_loader.test_iter(mode), total=data_loader.num_batches(mode), desc="evaluate {}".format(mode)):
        raw_data, feed_dict = get_feed_dict(data, model, mode=mode)
        start_indexes, end_indexes = sess.run([model.start_index, model.end_index], feed_dict=feed_dict)
        for record, start_index, end_index in zip(raw_data, start_indexes, end_indexes):
            # print(record["vid"], record["words"])
            start_time, end_time = index_to_time(start_index, end_index, record["v_len"], record["duration"])
            gs, ge = index_to_time(record['s_ind'], record['e_ind'], record['v_len'], record["duration"])
            iou = calculate_iou(i0=[start_time, end_time], i1=[gs, ge])
            # print("iou:{} | gt: {}  {} | predict: {}  {}".format(iou, gs, ge ,start_time, end_time))

            ious.append(iou)
    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    mi = np.mean(ious) * 100.0
    value_pairs = [("{}/Rank@1, IoU=0.3".format(mode), r1i3), ("{}/Rank@1, IoU=0.5".format(mode), r1i5),
                   ("{}/Rank@1, IoU=0.7".format(mode), r1i7), ("{}/mean IoU".format(mode), mi)]
    # write the scores
    score_str = "Epoch {}, Step {}:\n".format(epoch, global_step)
    score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
    score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
    score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
    score_str += "mean IoU: {:.2f}\n".format(mi)
    return r1i3, r1i5, r1i7, mi, value_pairs, score_str



def eval_test_save(sess, model, data_loader, task, suffix, epoch=None, global_step=None, mode="test"):
    import json
    ious = list()
    save_list = []
    for data in tqdm(data_loader.test_iter(mode), total=data_loader.num_batches(mode), desc="evaluate {}".format(mode)):
        raw_data, feed_dict = get_feed_dict(data, model, mode=mode)


        match_scores = sess.run(model.match_scores, feed_dict=feed_dict)
        # print(match_scores.shape)
        start_indexes, end_indexes = sess.run([model.start_index, model.end_index], feed_dict=feed_dict)
        start_logits, end_logits = sess.run([model.start_logits, model.end_logits], feed_dict=feed_dict)

        # raw_data, feed_dict_dropout05 = get_feed_dict(data, model, drop_rate=0.5, mode=mode)
        # start_logits1, end_logits1 = sess.run([model.start_logits, model.end_logits], feed_dict=feed_dict_dropout05)
        # start_logits2, end_logits2 = sess.run([model.start_logits, model.end_logits], feed_dict=feed_dict_dropout05)

        for record, start_index, end_index in zip(raw_data, start_indexes, end_indexes):
            start_time, end_time = index_to_time(start_index, end_index, record["v_len"], record["duration"])
            gs, ge = index_to_time(record['s_ind'], record['e_ind'], record['v_len'], record["duration"])
            iou = calculate_iou(i0=[start_time, end_time], i1=[gs, ge])
            ious.append(iou)

        for i in range(len(start_indexes)):
            tmp = {'vid': raw_data[i]["vid"], 
                #    "duration": raw_data[i]["duration"],
                #     'psuedo_idx': [raw_data[i]["s_ind"], raw_data[i]["e_ind"]],
                #    'sentence': " ".join(raw_data[i]["words"]),
                   'vlen': int(raw_data[i]["v_len"]),
                #    'prop_idx': [int(start_indexes[i]), int(end_indexes[i])],
                   'prop_logits': [start_logits[i], end_logits[i]],
            }

            save_list.append(tmp)
    
    os.makedirs("./results/{}".format(task), exist_ok=True)
    outpath = "./results/{}/{}.pkl".format(task, suffix)
    print(outpath)
    with open(outpath, 'wb') as f:
        pickle.dump(save_list, f)

    r1i3 = calculate_iou_accuracy(ious, threshold=0.3)
    r1i5 = calculate_iou_accuracy(ious, threshold=0.5)
    r1i7 = calculate_iou_accuracy(ious, threshold=0.7)
    mi = np.mean(ious) * 100.0
    value_pairs = [("{}/Rank@1, IoU=0.3".format(mode), r1i3), ("{}/Rank@1, IoU=0.5".format(mode), r1i5),
                   ("{}/Rank@1, IoU=0.7".format(mode), r1i7), ("{}/mean IoU".format(mode), mi)]
    # write the scores
    score_str = "Epoch {}, Step {}:\n".format(epoch, global_step)
    score_str += "Rank@1, IoU=0.3: {:.2f}\t".format(r1i3)
    score_str += "Rank@1, IoU=0.5: {:.2f}\t".format(r1i5)
    score_str += "Rank@1, IoU=0.7: {:.2f}\t".format(r1i7)
    score_str += "mean IoU: {:.2f}\n".format(mi)
    return r1i3, r1i5, r1i7, mi, value_pairs, score_str
