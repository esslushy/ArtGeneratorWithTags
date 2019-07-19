from PIL import Image
import glob
Image.MAX_IMAGE_PIXELS = 1e11

for path in glob.glob('../dataset/images/*'):
    print(path)
    image = Image.open(path)
    image = image.resize((256, 256))
    image = image.convert('RGB')
    image.save(path)