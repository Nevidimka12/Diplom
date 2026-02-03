import torch
from torch.nn.utils.rnn import pad_sequence


def multitask_collate_fn(batch):
    # Эмбеддинги
    xs = [item["x"] for item in batch]  # список (T_i, H)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

    # (B, T_max, H)
    x_padded = pad_sequence(xs, batch_first=True)

    B = len(batch)

    # CMU-MOSEI:
    has_emo_mosei = torch.tensor(
        [item.get("emotion_mosei") is not None for item in batch],
        dtype=torch.bool,
    )

    if has_emo_mosei.any():
        emo_mosei_list = []
        for item in batch:
            if item.get("emotion_mosei") is not None:
                emo_mosei_list.append(item["emotion_mosei"])
            else:
                emo_mosei_list.append(torch.zeros(7, dtype=torch.float32))
        emotion_mosei = torch.stack(emo_mosei_list, dim=0)  # (B, 7)
    else:
        emotion_mosei = torch.zeros(B, 7, dtype=torch.float32)


    # RESD:
    has_emo_resd = torch.tensor(
        [item.get("emotion_resd") is not None for item in batch],
        dtype=torch.bool,
    )

    if has_emo_resd.any():
        emo_resd_list = []
        for item in batch:
            if item.get("emotion_resd") is not None:
                emo_resd_list.append(int(item["emotion_resd"]))
            else:
                emo_resd_list.append(0)
        emotion_resd = torch.tensor(emo_resd_list, dtype=torch.long)  # (B,)
    else:
        emotion_resd = torch.zeros(B, dtype=torch.long)

    # FirstImpressionsV2:
    has_pers = torch.tensor(
        [item.get("personality") is not None for item in batch],
        dtype=torch.bool,
    )

    if has_pers.any():
        pers_list = []
        for item in batch:
            if item.get("personality") is not None:
                pers_list.append(item["personality"])
            else:
                pers_list.append(torch.zeros(5, dtype=torch.float32))
        personality = torch.stack(pers_list, dim=0)  # (B, 5)
    else:
        personality = torch.zeros(B, 5, dtype=torch.float32)

    # BAH:
    has_ah = torch.tensor(
        [item.get("ah") is not None for item in batch],
        dtype=torch.bool,
    )

    if has_ah.any():
        ah_list = []
        for item in batch:
            if item.get("ah") is not None:
                ah_list.append(int(item["ah"]))
            else:
                ah_list.append(0)
        ah = torch.tensor(ah_list, dtype=torch.long)  # (B,)
    else:
        ah = torch.zeros(B, dtype=torch.long)

    dataset_names = [item["dataset_name"] for item in batch]

    return {
        "x": x_padded,          # (B, T_max, H)
        "lengths": lengths,     # (B,)

        "emotion_mosei": emotion_mosei,   # (B, 7)
        "emotion_resd": emotion_resd,     # (B,)
        "personality": personality,       # (B, 5)
        "ah": ah,                         # (B,)

        "has_emo_mosei": has_emo_mosei,   # (B,)
        "has_emo_resd": has_emo_resd,     # (B,)
        "has_pers": has_pers,             # (B,)
        "has_ah": has_ah,                 # (B,)

        "dataset_names": dataset_names,
    }
