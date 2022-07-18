from yaml.loader import Loader
from model import LeoModel
import torch
import torch.nn as nn
import argparse
import yaml
from torchsummary import summary

from leo import LEO


def parse():
    parser = argparse.ArgumentParser(description="leo")
    parser.add_argument("-exp_name", default="simple_train_1k_1", type=str)
    parser.add_argument("-save_path", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    config = yaml.load(open("configv2.0.yaml", "r"), Loader=yaml.SafeLoader)
    exp_conf = config[args.exp_name]
    leo_model = LEO(
        N=1,
        K=exp_conf["K"],
        batch_size=exp_conf["batch_size"],
        embedding_size=exp_conf["embedding_size"],
        hidden_size=exp_conf["hidden_size"],
        drop_out=exp_conf["drop_out"],
        inner_lr=exp_conf["inner_lr"],
        fintune_lr=exp_conf["fintune_lr"],
        cuda=exp_conf["cuda"],
        out_lr=exp_conf["out_lr"],
        kl_weight=exp_conf["kl_weight"],
        encoder_penalty_weight=exp_conf["encoder_penalty_weight"],
        orthogonality_penalty_weight=exp_conf["orthogonality_penalty_weight"],
        l2_penalty_weight=exp_conf["l2_penalty_weight"],
        inner_update_epoch=exp_conf["inner_update_epoch"],
        out_update_epoch=exp_conf["out_update_epoch"],
        total_step=exp_conf["total_step"],
        clip_vale=exp_conf["clip_vale"],
    )
    summary(leo_model.model)
    leo_model.train(task_path=exp_conf["split_data"])
    save_path = args.save_path + args.exp_name + '.pt'
    leo_model.save(save_path)


# nohup python -u main.py -exp_name simple_train_1k_1 -save_path model/ > model/simple_train_1k_1.log 2>&1 &
# nohup python -u main.py -exp_name osm_train -save_path model/ > model/osm_train.log 2>&1 &
# nohup python -u main.py -exp_name osm_train12 -save_path log/ > log/osm_train12.log 2>&1 &
# nohup python -u main.py -exp_name osm_train18 -save_path log/osm_en_us/osm_en_us > log/osm_en_us/osm_train18.log 2>&1 &
# nohup python -u main.py -exp_name osm_cn_1 -save_path log/osm_cn/ > log/osm_cn/osm_cn_1.log 2>&1 &
# nohup python -u main.py -exp_name osm_ne_us_1 -save_path log/osm_en_us/ > log/osm_en_us/osm_en_us_1.log 2>&1 &
# nohup python -u main.py -exp_name tiger_1 -save_path log/tiger/ > log/tiger/tiger_1.log 2>&1 &
# nohup python -u main.py -exp_name skewed1 -save_path log/skewed/ > log/skewed/skewed_1.log 2>&1 &
# nohup python -u main.py -exp_name uniform1 -save_path log/uniform/ > log/uniform/uniform_1.log 2>&1 &