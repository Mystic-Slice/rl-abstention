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
