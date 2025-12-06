import random
from dataclasses import dataclass
from typing import List, Literal

BoundType = Literal["fixed", "variable"]

@dataclass
class BKPItem:
    w: int
    p: int
    b: int

@dataclass
class BKPDataset:
    items: List[BKPItem]
    capacity: int


def generate_bkp_dataset(n: int, b_max: int, bound_type: BoundType = "fixed", seed=None) -> BKPDataset:
    if seed is not None:
        random.seed(seed)

    weights = [random.randint(1, 100) for _ in range(n)]
    c = sum(weights) // 2

    items: List[BKPItem] = []

    for w in weights:
        max_bound = min(b_max, c // w)
        if max_bound < 1:
            max_bound = 1

        if bound_type == "fixed":
            b = max_bound
        else:
            b = random.randint(1, max_bound)

        p = w + 5
        items.append(BKPItem(w=w, p=p, b=b))

    items.sort(key=lambda it: it.p / it.w, reverse=True)

    return BKPDataset(items=items, capacity=c)


def save_bkp_to_txt(dataset: BKPDataset, filename: str):
    with open(filename, "w") as f:
        n = len(dataset.items)
        f.write(f"{n} {dataset.capacity}\n")
        for item in dataset.items:
            f.write(f"{item.w} {item.p} {item.b}\n")

    print(f"✓ Saved {filename}")

# Generate multiple datasets with fixed bounds
def generate_multiple_fixed(n: int, b_max: int, repeats: int = 5):
    for i in range(1, repeats + 1):
        seed = 100 + i  # 保證每次不同
        ds = generate_bkp_dataset(n=n, b_max=b_max, bound_type="fixed", seed=seed)

        filename = f"bkp_n{n}_bm{b_max}_fixed_{i}.txt"
        save_bkp_to_txt(ds, filename)

# Generate multiple datasets with variable bounds
def generate_multiple_variable(n: int, b_max: int, repeats: int = 5):
    for i in range(1, repeats + 1):
        # Ensure different seeds for different datasets
        seed = 100 + i  
        ds = generate_bkp_dataset(n=n, b_max=b_max, bound_type="variable", seed=seed)

        filename = f"bkp_n{n}_bm{b_max}_variable_{i}.txt"
        save_bkp_to_txt(ds, filename)

if __name__ == "__main__":
    generate_multiple_fixed(n=10000, b_max=200, repeats=5)
    generate_multiple_variable(n=10000, b_max=200, repeats=5)