import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim import Adam
import network
import wandb


def load_and_transform_data(IMG_SIZE, BATCH_SIZE):
    data_transforms = torchvision.transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * x - 1)  # normalize to [-1, 1]
    ])

    train = torchvision.datasets.Flowers102(root='../data/flower', download=True, split='train',
                                            transform=data_transforms)
    test = torchvision.datasets.Flowers102(root='../data/flower', download=True, split='test',
                                           transform=data_transforms)

    # Since we are training a generative model, we don't need
    # splitted train and test datasets. We can just concatenate them
    data = torch.utils.data.ConcatDataset([train, test])
    data_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8)

    return data_loader


def reverse_transforms(img):
    '''
    Reverse the transforms we applied while loading the data
    in order to visualize the images
    '''
    reverse_tr = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1) / 2),  # denormalize to [0, 1]
        transforms.ToPILImage()
    ])

    return reverse_tr(img)


def linear_beta_schedule(beta_0, beta_T, T):
    '''Linear schedule for beta from beta_0 to beta_T in T steps'''
    return torch.linspace(beta_0, beta_T, T)


# Take the original image and time step, and return the image after the diffusion process
def forward_diffusion(x_0, t, alpha_cumprods):
    alpha_cumprod = alpha_cumprods[t]
    if len(alpha_cumprod.shape) == 0:
        alpha_cumprod = alpha_cumprod.unsqueeze(0)
    alpha_cumprod = alpha_cumprod[:, None, None, None]

    noise = torch.randn_like(x_0)
    x_t = x_0 * torch.sqrt(alpha_cumprod) + torch.sqrt(1 - alpha_cumprod) * noise

    return x_t.squeeze(), noise


def get_loss(model, x_0, t):
    x_t, noise = forward_diffusion(x_0, t, alpha_cumprods)
    pred_noise = model(x_t, t)

    return torch.mean((pred_noise - noise) ** 2)


@torch.no_grad()
def sample_timestep(x_t, t, alpha_cumprods):
    if t == 0:
        alpha_t = alpha_cumprods[0]
        z = 0
    else:
        alpha_t = alpha_cumprods[t] / alpha_cumprods[t - 1]
        z = torch.randn_like(x_t, device=device)  # Sample noise from a normal distribution

    alphas_cumprod_prev = torch.nn.functional.pad(alpha_cumprods[:-1], (1, 0), value=1.0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alpha_cumprods)
    posterior_variance_t = posterior_variance[t].to(device)
    pred_noise = model(x_t, t)
    prev_x = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprods[t]) * pred_noise) / torch.sqrt(
        alpha_t) + z * torch.sqrt(posterior_variance_t)

    return prev_x


@torch.no_grad()
def sample_plot_image(return_final=False):
    imgs = []  # store images during denoising
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)

    num_images = 8  # number of images to plot
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, alpha_cumprods)
        if return_final and i == 0:  # we already got x_0
            return reverse_transforms(img.detach().cpu().squeeze())
        if i % stepsize == 0:
            imgs.append(reverse_transforms(img.detach().cpu().squeeze()))

    return imgs


if __name__ == '__main__':
    IMG_SIZE = 64
    BATCH_SIZE = 128
    device = torch.device("cuda")

    data_loader = load_and_transform_data(IMG_SIZE, BATCH_SIZE)
    len(data_loader)

    T = 350  # number of times the diffusion process is applied
    betas = linear_beta_schedule(1e-4, 2e-2, T)
    alphas = 1 - betas
    alpha_cumprods = torch.cumprod(alphas, dim=0).to(device)  # pre-compute cumulative product of alphas
    betas = betas.to(device)

    image = next(iter(data_loader))[0][0].to(device)

    # Visualize the forward diffusion process
    num_images = 8
    step_size = int(T / num_images)

    plt.figure(figsize=(20, 15))
    plt.axis('off')
    for idx in range(0, T, step_size):
        t = torch.Tensor([idx]).type(torch.int64).to(device)
        plt.subplot(1, num_images + 1, int(idx / step_size) + 1)

        noisy_image, noise = forward_diffusion(image, idx, alpha_cumprods)
        plt.imshow(reverse_transforms(noisy_image))
        plt.title(f't = {idx}')
        plt.axis('off')
    # plt.show()

    model = network.SimpleUnet()
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    model.to(device)
    learning_rate = 0.001
    epochs = 200
    batch_size = 128
    optimizer = Adam(model.parameters(), lr=learning_rate)

    wandb.init(
        project="Diffusion model",
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(data_loader):
            optimizer.zero_grad()

            batch = batch[0].to(device)
            t = torch.randint(0, T, (batch_size,), device=device).long().to(device)
            loss = get_loss(model, batch, t)
            epoch_loss += loss.detach().cpu().item()

            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} | Batch {step} | Loss: {epoch_loss / len(data_loader)} ", end="\r")

        # Eval
        model.eval()
        imgs = sample_plot_image()
        wandb.log({
            "epoch_loss": epoch_loss / len(data_loader),
            "images": [wandb.Image(img) for img in imgs]
        })

        pause = True
        num_images = len(imgs)

        fig = plt.figure(figsize=(20, 10))  # Adjust as needed
        # Loop over images and add them to subplots
        for i in range(len(imgs)):
            ax = fig.add_subplot(1, len(imgs), i + 1)
            img = imgs[i]
            ax.imshow(img)
            ax.axis('off')  # To not display axis
        plt.savefig(f'./results/flowers/flowers_{epoch}.png')
        plt.show()
