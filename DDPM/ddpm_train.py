from utils import *

## Params
batch_size = 64
num_time_steps = 1000
num_epochs = 15
seed = -1
ema_decay = 0.9999 
lr=2e-5
checkpoint_path=None

##
set_seed(random.randint(0, 2**32-1)) if seed == -1 else set_seed(seed)

train_dataset = datasets.MNIST(root='./data', train=True, download=False,transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
model = UNET().cuda()
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
        x = x.cuda()
        x = F.pad(x, (2,2,2,2))
        t = torch.randint(0,num_time_steps,(batch_size,))
        e = torch.randn_like(x, requires_grad=False)
        a = scheduler.alpha[t].view(batch_size,1,1,1).cuda()
        x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)
        output = model(x, t)
        optimizer.zero_grad()
        loss = criterion(output, e)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        ema.update(model)
    print(f'Epoch {i+1} | Loss {total_loss / (60000/batch_size):.5f}')

checkpoint = {
    'weights': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'ema': ema.state_dict()
}
torch.save(checkpoint, 'checkpoints/ddpm_checkpoint')