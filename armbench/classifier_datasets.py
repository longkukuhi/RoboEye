import torch
import os
import numpy as np
from beit3_tools.beit3_datasets import create_dataloader

class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

        features_path = os.path.join(data_path,  f'features_3t1.npy')
        labels_path = os.path.join(data_path,  f'labels_3t1.npy')
        self.features = torch.tensor(np.load(features_path), dtype=torch.float32)  #[N, 768]
        self.labels = torch.tensor(np.load(labels_path), dtype=torch.long)  #[N]

        
    def _get_target_example(self, index: int, data: dict):
        data["features"] = self.features[index]  #[768]
        data["labels"] = self.labels[index]  #[]


    def __getitem__(self, index: int):
        data = dict()
        self._get_target_example(index, data)
        return data
    
    def __len__(self) -> int:
        return len(self.labels)
    
def create_classifier_dataset(args): 
    is_train = True

    dataset = ClassifierDataset(
        data_path=args.data_path,
    )

    dataloader = create_dataloader(
        dataset, is_train=is_train, batch_size=args.batch_size,
        num_workers=args.num_workers, dist_eval=args.distributed,
        pin_mem=args.pin_mem,
    )

    return dataloader