from utils import *

## Params
batch_size = 128
num_time_steps = 1000
num_epochs = 3
seed = -1
ema_decay = 0.9999 
lr=1e-2
checkpoint_path=None

##
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = torch.device("cpu")
print(f"Using device: {device}")
# set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

if __name__ == "__main__":
    torch.set_printoptions(threshold=10, edgeitems=2, linewidth=80)

    train_dataset = datasets.MNIST(root='./data', train=True, download=False,transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = UNET(Channels=[4, 8, 16, 32, 32, 24]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    criterion = nn.MSELoss(reduction='mean')

    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x,_) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            x = x.to(device).contiguous()
            x = F.pad(x, (2,2,2,2)).contiguous()
            t = torch.randint(0,num_time_steps,(batch_size,)).contiguous()
            e = torch.randn_like(x, requires_grad=False).contiguous()
            a = scheduler.alpha[t].reshape(batch_size,1,1,1).expand_as(x).contiguous()
            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e).contiguous()

            x = x.contiguous().to(memory_format=torch.contiguous_format)
            e = e.contiguous().to(memory_format=torch.contiguous_format)
            a = a.contiguous().to(memory_format=torch.contiguous_format)
            x = x.to(memory_format=torch.contiguous_format)

            optimizer.zero_grad()

            # with torch.autograd.set_detect_anomaly():
            torch.autograd.set_detect_anomaly(True)
            output = model(x, t)
            output = output.contiguous().to(memory_format=torch.contiguous_format)
            loss = criterion(output, e).contiguous().to(memory_format=torch.contiguous_format)

            total_loss += loss.item()
            # try:
                # loss.backward()
            # except RuntimeError as e:
            #     print("Backward pass error:", e)
            #     raise

            loss.backward()

            optimizer.step()
            ema.update(model)
        print(f'Epoch {i+1} | Loss {total_loss / (60000/batch_size):.5f}')

    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }
    save_path = '/Users/jamalesmaily/Desktop/Cambridge/Projects/DL_models1/DLModels_Pytorch/DDPM/checkpoints/'

    # os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(checkpoint, save_path + 'model4.pth')