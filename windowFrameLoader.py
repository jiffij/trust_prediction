from torch.utils.data import Dataset, DataLoader


class WindowFrameDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_window_frame_dataset(X, y, batch_size):
    return DataLoader(WindowFrameDataset(X, y), batch_size=batch_size)

