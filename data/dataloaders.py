from typing import List, Optional

from torch.utils.data import ConcatDataset, DataLoader, Sampler

from data.collate import multitask_collate_fn


def make_concat_dataset(datasets: List) -> ConcatDataset:
    if len(datasets) == 0:
        raise ValueError("Список датасетов пуст.")
    return ConcatDataset(datasets)


def make_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    sampler: Optional[Sampler] = None,
    batch_sampler: Optional[Sampler] = None,
):
    if batch_sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=multitask_collate_fn,
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=multitask_collate_fn,
    )
    return loader
