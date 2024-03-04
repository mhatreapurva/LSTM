from datasets import load_dataset

class DownloadDataset:
    def __init__(self, dataset_name) -> None:
        self.dataset_name = dataset_name
    
    def download_dataset(self):
        dataset = load_dataset(self.dataset_name)
        return dataset
