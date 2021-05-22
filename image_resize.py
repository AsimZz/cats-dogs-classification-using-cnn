from PIL import Image


def resizedImage(src_image, size=(128, 128), bg_color="white"):
    # resize the image so the longest dimension matches our target size
    src_image.thumbnail(size, Image.ANTIALIAS)

    # Create a new square background image
    new_image = Image.new("RGB", size, bg_color)

    # Paste the resized image into the center of the square background
    new_image.paste(src_image, (int(
        (size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))

    return new_image
