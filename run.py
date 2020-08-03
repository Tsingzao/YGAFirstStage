import torch
from MetNet.model import MetNet
from torch.autograd import Variable
from MetNet import *
from torch.utils.data import DataLoader


if __name__ == '__main__':
    model = MetNet().to(cfg.device)
    checkpoint = torch.load(cfg.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    loader = DataLoader(TestLoader(), batch_size=1, shuffle=False)


    print('Start generating...')
    model.eval()
    with torch.no_grad():
        for batch, (data, path) in enumerate(loader):
            data = Variable(data).float().to(cfg.device).permute((1, 0, 2, 3, 4))
            output = model(data).permute((1, 0, 2, 3, 4))
            saveOutput(output.data.cpu().numpy()[0], path)
            progress_bar(batch, len(loader))

