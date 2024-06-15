from data_provider.data_loader import Dataset_Delta, Dataset_Regular
from utils.constants import FUTURE_STEPS, HISTORY_STEPS, BATCH
from torch.utils.data import DataLoader

def data_provider(x, y, delta=True):
  if delta==True:
    Data = Dataset_Delta
  else:
    Data = Dataset_Regular

    shuffle_flag = False

  data_set = Data(
        x,
        y,
        size=[HISTORY_STEPS, FUTURE_STEPS]
    )

  data_loader = DataLoader(
        data_set,
        batch_size=BATCH,
        shuffle=False,
        num_workers=1,
        drop_last=True)
  
  return data_set, data_loader