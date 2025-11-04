from itertools import islice

# TODO: remove if not used
def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def chunked(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

def resolve_checkpoint(resume_flag, output_dir):
    """Resolve checkpoint path from flag and output directory."""
    if isinstance(resume_flag, str):
        return resume_flag
    if resume_flag:
        try:
            return trainer_utils.get_last_checkpoint(output_dir)
        except Exception as e:
            logger.warning(f"Could not detect checkpoint: {e}")
            return None
    return None