import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# 1. Function to load image from URL and preprocess
def load_image_from_url(url, max_size=400, shape=None):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"[ERROR] Could not load image from {url} \n{e}")
        return None

    # Resize
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    image = in_transform(image).unsqueeze(0)
    return image

# 2. Use valid working image URLs
content_url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Golden_Beach_Sri_Lanka.jpg"
style_url = "https://upload.wikimedia.org/wikipedia/commons/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"

# 3. Load images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content = load_image_from_url(content_url).to(device)
style = load_image_from_url(style_url, shape=content.shape[-2:]).to(device)

# 4. Load VGG19 model
vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad_(False)

# 5. Extract features
def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content
            '28': 'conv5_1'
        }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# 6. Gram matrix for style
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# 7. Get features
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# 8. Initialize target image
target = content.clone().requires_grad_(True).to(device)

# 9. Define weights
style_weight = 1e6
content_weight = 1

# 10. Optimizer
optimizer = optim.Adam([target], lr=0.003)

# 11. Style transfer loop
steps = 300
for i in range(steps):
    target_features = get_features(target, vgg)

    # Content loss
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

    # Style loss
    style_loss = 0
    for layer in style_grams:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[layer]
        layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
        style_loss += layer_style_loss / (target_feature.shape[1] ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"Step {i}/{steps} | Total loss: {total_loss.item():.4f}")

# 12. Convert tensor to image
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    image = image.clip(0, 1)
    return image

# 13. Display result
plt.figure(figsize=(10, 5))
plt.imshow(im_convert(target))
plt.title("Stylized Output")
plt.axis("off")
plt.show()
