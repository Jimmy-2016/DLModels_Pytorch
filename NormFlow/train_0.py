
from util_0 import *


# Training setup
num_samples = 1000
in_features = 2
num_layers = 5
model = NormalizingFlow(in_features, num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

x_data = torch.randn(num_samples, in_features)

target_data = torch.tensor(target_distribution(num_samples), dtype=torch.float32)

# Training loop
for epoch in range(2000):
    model.train()
    optimizer.zero_grad()

    x_transformed, log_det_jacobian = model(target_data)

    # Calculate the loss
    # logprob = prior_logprob + log_det
    # loss = torch.sum(logprob)

    loss = criterion(x_transformed, target_data)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')


with torch.no_grad():
    model.eval()
    x_transformed, _ = model(x_data)

    plt.figure(figsize=(8, 8))
    plt.scatter(target_data[:, 0], target_data[:, 1], label='Target Distribution', alpha=0.5)
    plt.scatter(x_transformed[:, 0], x_transformed[:, 1], label='Transformed Distribution', alpha=0.5)
    plt.title('Normalizing Flow Transformation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()


plt.show()
