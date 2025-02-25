import time
import torch
from torch.utils.data import DataLoader
from datasets import MammoLocalizationDataset, ResizeMammoClassification

def benchmark_num_workers(data_dir, resize_size, num_workers_list):
    # Koristimo transformaciju za resize slika
    transform_fn = ResizeMammoClassification(resize_size)
    # Inicijaliziramo dataset (ovdje se automatski skaliraju slike i bounding box koordinate)
    dataset = MammoLocalizationDataset(data_dir=data_dir, transform=transform_fn, resize_output_size=resize_size)
    
    results = {}
    for nw in num_workers_list:
        print(f"\nTestiranje num_workers = {nw}")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=nw)
        start_time = time.time()
        # Prolazimo kroz cijeli dataloader samo da izmjerimo vrijeme učitavanja
        for i, batch in enumerate(dataloader):
            # Možeš dodati i neku simulaciju obrade ako želiš,
            # ali ovdje samo iteriramo.
            pass
        elapsed = time.time() - start_time
        avg_time = elapsed / len(dataloader)
        results[nw] = (elapsed, avg_time)
        print(f"Ukupno vrijeme: {elapsed:.2f} sekundi, prosječno vrijeme po batchu: {avg_time:.4f} sekundi")
    
    return results

if __name__ == "__main__":
    # Postavi putanju do podataka
    data_dir = "/home/data/train/"  # prilagodi prema svom okruženju
    resize_size = (512, 512)
    num_workers_list = [0, 2, 4, 8, 16]  # Možeš proširiti listu prema potrebama
    benchmark_results = benchmark_num_workers(data_dir, resize_size, num_workers_list)
