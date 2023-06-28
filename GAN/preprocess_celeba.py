import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the training data
celeba_data = datasets.CelebA(root='../data', download=True, transform=transform)

# root path depends on your computer
root = '../data/celebA/celebA/'
save_root = '../data/resized_celebA/'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root + 'celebA'):
    os.mkdir(save_root + 'celebA')
img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    from skimage.transform import resize
    from skimage import img_as_ubyte

    img = plt.imread(root + img_list[i])
    img_resized = resize(img, (resize_size, resize_size))  # Returns a float array
    img_resized = img_as_ubyte(img_resized)  # Convert it back to an unsigned byte type
    plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img_resized)

    if (i % 1000) == 0:
        print('%d images complete' % i)