from skimage.util import img_as_ubyte
import cv2


# todo refactor later
def luminosity_tissue_mask(img, luminosity_threshold=0.8):
    """Get a binary mask where true denotes pixels with a luminosity less than the specified threshold.
    Typically, we use to identify tissue in the image and exclude the bright white background.
    
    Args:
        img:
        luminosity_threshold:
    Returns:

    """
    img = img_as_ubyte(img)
    I_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L = I_LAB[:, :, 0] / 255.0  # Convert to range [0,1].
    mask = L < luminosity_threshold
    return mask
