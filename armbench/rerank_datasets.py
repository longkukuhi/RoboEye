import torch
import os
import json
from vggt.utils.load_fn import load_and_preprocess_images
from beit3_tools.beit3_datasets import create_dataloader
from sift import get_query_points

class RerankDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

        with open(os.path.join(data_path,  f'sample_paths_full.json'), 'r') as f:
            self.items = json.load(f)
        
    def _get_image_example(self, index: int, data: dict):
        data_per_pick = self.items[index]
        data["query_points"] = get_query_points(data_per_pick["query_path"])  #[N, 2]
        data["query_image"] = load_and_preprocess_images([data_per_pick["query_path"]], mode="pad")  #[1, C, H, W]
        data["positive_image"] = load_and_preprocess_images([data_per_pick["positive_path"]], mode="pad")  #[1, C, H, W]
        data["negative_images"] = load_and_preprocess_images(data_per_pick["negative_paths"], mode="pad")  #[3, C, H, W]


    def __getitem__(self, index: int):
        data = dict()
        self._get_image_example(index, data)
        return data
    
    def __len__(self) -> int:
        return len(self.items)

class RerankDataset3t1(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

        with open(os.path.join(data_path,  f'sample_paths_full_3t1.json'), 'r') as f:
            self.items = json.load(f)
        
    def _get_image_example(self, index: int, data: dict):
        data_per_pick = self.items[index]
        data["query_points0"] = get_query_points(data_per_pick["query_path0"])  #[N, 2]
        data["query_points1"] = get_query_points(data_per_pick["query_path1"])  #[N, 2]
        data["query_points2"] = get_query_points(data_per_pick["query_path2"])  #[N, 2]
        data["query_image0"] = load_and_preprocess_images([data_per_pick["query_path0"]], mode="pad")  #[1, C, H, W]
        data["query_image1"] = load_and_preprocess_images([data_per_pick["query_path1"]], mode="pad")  #[1, C, H, W]
        data["query_image2"] = load_and_preprocess_images([data_per_pick["query_path2"]], mode="pad")  #[1, C, H, W]
        data["positive_image"] = load_and_preprocess_images([data_per_pick["positive_path"]], mode="pad")  #[1, C, H, W]
        data["negative_images"] = load_and_preprocess_images(data_per_pick["negative_paths"], mode="pad")  #[3, C, H, W]


    def __getitem__(self, index: int):
        data = dict()
        self._get_image_example(index, data)
        return data
    
    def __len__(self) -> int:
        return len(self.items)
    
class RerankDataset3t1test(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

        with open(os.path.join(data_path,  f'sample_paths_full_3t1_together.json'), 'r') as f:
            self.items = json.load(f)
        
    def _get_image_example(self, index: int, data: dict):
        data_per_pick = self.items[index]
        #data["query_points0"] = get_query_points(data_per_pick["query_path0"])  #[N, 2]
        data["query_points"] = get_query_points(data_per_pick["query_path1"])  #[N, 2]
        #data["query_points2"] = get_query_points(data_per_pick["query_path2"])  #[N, 2]
        data["query_image0"] = load_and_preprocess_images([data_per_pick["query_path0"]], mode="pad")  #[1, C, H, W]
        data["query_image1"] = load_and_preprocess_images([data_per_pick["query_path1"]], mode="pad")  #[1, C, H, W]
        data["query_image2"] = load_and_preprocess_images([data_per_pick["query_path2"]], mode="pad")  #[1, C, H, W]
        data["positive_image"] = load_and_preprocess_images([data_per_pick["positive_path"]], mode="pad")  #[1, C, H, W]
        data["negative_images"] = load_and_preprocess_images(data_per_pick["negative_paths"], mode="pad")  #[3, C, H, W]


    def __getitem__(self, index: int):
        data = dict()
        self._get_image_example(index, data)
        return data
    
    def __len__(self) -> int:
        return len(self.items)
    
def create_rerank_dataset(args): 
    is_train = True

    dataset = RerankDataset(
        data_path=args.data_path,
    )

    dataloader = create_dataloader(
        dataset, is_train=is_train, batch_size=args.batch_size,
        num_workers=args.num_workers, dist_eval=args.distributed,
        pin_mem=args.pin_mem,
    )

    return dataloader

def create_rerank_dataset3t1(args): 
    is_train = True

    dataset = RerankDataset3t1(
        data_path=args.data_path,
    )

    dataloader = create_dataloader(
        dataset, is_train=is_train, batch_size=args.batch_size,
        num_workers=args.num_workers, dist_eval=args.distributed,
        pin_mem=args.pin_mem,
    )

    return dataloader