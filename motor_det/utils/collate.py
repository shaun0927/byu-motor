from torch.utils.data.dataloader import default_collate


def collate_with_centers(batch):
    """Custom collate_fn that keeps variable length ``centers_Å`` as a list."""
    centers = [b["centers_Å"] for b in batch]
    batch_no_centers = [{k: v for k, v in b.items() if k != "centers_Å"} for b in batch]
    out = default_collate(batch_no_centers)
    out["centers_Å"] = centers
    return out

