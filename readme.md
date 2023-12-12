## The code for "Partial Annotation-based Video Moment Retrieval via Iterative Learning" MM 2023

Note:
1. This code is based on the SeqPAN Tensorflow version, so I strongly recommend adapting the idea to some Pytorch methods.
2. Here is a lower-performance version based on VSLNet, which codes with Pytorch https://github.com/renjie-liang/VMRIL_VSLNet.


## Download Feature
1. We use features on https://app.box.com/s/d7q5atlidb31cuj1u8znd7prgrck1r1s from https://github.com/26hzhang/SeqPAN.
2. You can also download the video features from https://huggingface.co/datasets/k-nick/NLVL. But the feature should be converted from h5 to .npy files. Or you can follow this repository to modify the load code. https://github.com/renjie-liang/TSGVZoo



## Quick Start
```
# train
# modify the feature_path and emb_path in main.py
python run_charades_P30.py

# generate pseudo label
sh generate_label/weak_random.sh

# Summary the performance
python get_miou_P.py
```



### Code Comments
update_label_charades_P.py : update the 30% fix partial annotation


### Citation
If you feel this project is helpful to your research, please cite our work.
```
@inproceedings{ji2023partial,
  title={Partial annotation-based video moment retrieval via iterative learning},
  author={Ji, Wei and Liang, Renjie and Liao, Lizi and Fei, Hao and Feng, Fuli},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={4330--4339},
  year={2023}
}

```
