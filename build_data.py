from data.dataset_builders import (
    CMUMOSEIEmotionDataset,
    RESDEmotionDataset,
    FIV2PersonalityDataset,
    BAHAbsencePresenceDataset,
)
from data.dataloaders import make_dataloader, make_concat_dataset


def build_all_loaders(cfg):

    emotion_mapping = {
        'anger': 0,
        'disgust': 1,
        'fear': 2,
        'happiness': 3,
        'neutral': 4,
        'sadness': 5,
        'enthusiasm': 6,
    }

    # Train
    ds_mosei = CMUMOSEIEmotionDataset(
        embeddings_npy_path=cfg.mosei_train_emb,
        labels_csv_path=cfg.mosei_train_csv,
    )

    ds_resd = RESDEmotionDataset(
        embeddings_npy_path=cfg.resd_train_emb,
        labels_csv_path=cfg.resd_train_csv,
        label_mapping=emotion_mapping,
    )

    ds_fiv2 = FIV2PersonalityDataset(
        embeddings_npy_path=cfg.fiv2_train_emb,
        labels_csv_path=cfg.fiv2_csv,
        subset="train",
    )

    ds_bah = BAHAbsencePresenceDataset(
        embeddings_npy_path=cfg.bah_emb,
        labels_csv_path=cfg.bah_csv,
        split_txt_path=cfg.bah_train_split,
    )

    train_datasets = [ds_mosei, ds_resd, ds_fiv2, ds_bah]
    train_concat = make_concat_dataset(train_datasets)

    train_loader = make_dataloader(
        train_concat,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    # Test
    test_loaders = {
        "mosei": make_dataloader(
            CMUMOSEIEmotionDataset(cfg.mosei_test_emb, cfg.mosei_test_csv),
            batch_size=cfg.eval_bs,
            shuffle=False,
        ),
        "resd": make_dataloader(
            RESDEmotionDataset(cfg.resd_test_emb, cfg.resd_test_csv, label_mapping=emotion_mapping),
            batch_size=cfg.eval_bs,
            shuffle=False,
        ),
        "fiv2": make_dataloader(
            FIV2PersonalityDataset(cfg.fiv2_test_emb, cfg.fiv2_csv, subset="test"),
            batch_size=cfg.eval_bs,
            shuffle=False,
        ),
        "bah": make_dataloader(
            BAHAbsencePresenceDataset(cfg.bah_emb, cfg.bah_csv, cfg.bah_test_split),
            batch_size=cfg.eval_bs,
            shuffle=False,
        ),
    }

    return train_loader, test_loaders
