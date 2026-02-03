from training.hyper_search import greedy_search
from training.train_loop import train_model
from build_data import build_all_loaders


class Config:

    # Пути к эмбеддингам и меткам
    mosei_train_emb = "your_path.npz"
    mosei_train_csv = "your_path.npz"
    mosei_test_emb = "your_path.npz"
    mosei_test_csv = "your_path.csv"

    resd_train_emb = "your_path.npz"
    resd_train_csv = "your_path.csv"
    resd_test_emb = "your_path.npz"
    resd_test_csv = "your_path.csv"

    fiv2_train_emb = "your_path.npz"
    fiv2_csv = "your_path.csv"
    fiv2_test_emb = "your_path.npz"

    bah_emb = "your_path.npz"
    bah_csv = "your_path.csv"
    bah_train_split = "your_path.txt"
    bah_test_split = "your_path.txt"

    # Базовые настройки обучения
    batch_size = 16
    eval_bs = 8
    num_epochs = 30
    lr = 1e-4
    device = "cuda"
    checkpoint_dir = "./checkpoints"
    selection_metric = "mean_all"

    # MTL метод
    # "none" | "ntkmtl" | "taskgroup" | "fairgrad"

    mt_weight_method = "none"

    # NTKMTL
    mt_max_norm = 1.0
    ntk_exp = 0.5

    # TaskGroup
    group_decay = 1e-3
    group_regroup_interval = 50
    group_affinity_threshold = 0.0

    # FairGrad
    fairgrad_alpha = 1.0

    # Модель
    # "transformer" | "mamba" | "zeros"
    encoder_type = "mamba"

    # Zeros
    zeros_use_norm = True

    d_model = 128
    n_heads = 4
    num_layers = 4
    dim_feedforward = 1024
    dropout = 0.1
    max_len = 15000

    # Лоссы
    label_smoothing = 0.0
    regression_loss_type = "mse"

    # Mamba
    mamba_d_state = 16
    mamba_d_conv = 3
    mamba_expand = 2

    # Раздельные lr
    # encoder_lr = lr * encoder_lr_mult
    # heads_lr   = lr * heads_lr_mult
    encoder_lr_mult = 1.0
    heads_lr_mult = 1.0


def main():
    cfg = Config()
    cfg.checkpoint_dir = "./checkpoints"

    # greedy_search
    param_grid = {
        # Перебор нужных параметров и гиперпараметров
        "batch_size": [8, 16],
        "d_model": [128, 256],
        "num_layers": [2, 3, 4],
        "dropout": [0.1, 0.2, 0.3],
        # ...
    }

    def train_fn_for_search(local_cfg, _train_loader_ignored, _test_loaders_ignored):
        train_loader_local, test_loaders_local = build_all_loaders(local_cfg)
        return train_model(local_cfg, train_loader_local, test_loaders_local)

    best_params, best_metrics = greedy_search(
        base_cfg=cfg,
        train_loader=None,
        test_loaders=None,
        train_fn=train_fn_for_search,
        param_grid=param_grid,
        log_file="greedy_search.log",
        selection_metric="mean_all",
    )

    print("Best params:", best_params)
    print("Best metrics:", best_metrics)


if __name__ == "__main__":
    main()
