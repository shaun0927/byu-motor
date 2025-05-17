from torch.utils.data.dataloader import default_collate


def collate_with_centers(batch):
    """Custom ``collate_fn`` that keeps variable length ``centers_Å`` as a list.

    This function also validates that all tensor fields have consistent shapes
    across the batch.  If a mismatch is detected a ``ValueError`` is raised with
    a message describing which key caused the issue.  This helps catch dataset
    errors where crops of different sizes are accidentally yielded.
    """

    centers = [b["centers_Å"] for b in batch]
    batch_no_centers = [{k: v for k, v in b.items() if k != "centers_Å"} for b in batch]

    # Validate that tensor shapes are consistent across the batch before calling
    # ``default_collate`` which would otherwise fail with a less informative
    # message.
    if len(batch_no_centers) > 1:
        keys = batch_no_centers[0].keys()
        for k in keys:
            shapes = [b[k].shape for b in batch_no_centers]
            if not all(s == shapes[0] for s in shapes):
                raise ValueError(
                    f"Inconsistent shapes for key '{k}' when collating batch: {shapes}"
                )

    out = default_collate(batch_no_centers)
    out["centers_Å"] = centers
    return out

