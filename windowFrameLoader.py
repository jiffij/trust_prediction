from torch.utils.data import Dataset, DataLoader


class WindowFrameDataset(Dataset):
    """
    This is a custom dataset that for separating the trust dataset
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        # self.length = length

    def __len__(self):
        # if self.length is None:
        return len(self.X)
        # else:
        #     return self.length

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def load_window_frame_dataset(X, y, batch_size):
    """
    This function will create a custom WindowFrameDataset and load it as a pytorch dataloader
    :param X: trust price of t-1 (today) to t-n (yesterday), in tensor type
    :param y: trust price of t (tomorrow), in tensor type
    :param batch_size: batch size
    :return: DataLoader with custom dataset
    """
    # if sampler is None:
    return DataLoader(WindowFrameDataset(X, y), batch_size=batch_size)
    # else:
    #     return DataLoader(WindowFrameDataset(X, y, length), batch_size=batch_size, sampler=sampler)

