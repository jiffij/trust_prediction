from torch.utils.data import Dataset, DataLoader


class WindowFrameDataset(Dataset):
    def __init__(self, X, y, length=None):
        self.X = X
        self.y = y
        self.length = length

    def __len__(self):
        if self.length is None:
            return len(self.X)
        else:
            return self.length

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_window_frame_dataset(X, y, batch_size, sampler=None, length=None):
    if sampler is None:
        return DataLoader(WindowFrameDataset(X, y), batch_size=batch_size)
    else:
        return DataLoader(WindowFrameDataset(X, y, length), batch_size=batch_size, sampler=sampler)

