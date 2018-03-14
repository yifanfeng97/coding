from torchvision import transforms

def get_transform(img_sz):
    compose = transforms.Compose([
        transforms.Resize(img_sz),
        transforms.ToTensor()
    ])
    return compose

def get_untransform(img_sz):
    compose = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_sz)
    ])
    return compose
