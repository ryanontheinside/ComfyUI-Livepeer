import torch

class BatchInfo:
    """Node that extracts size and count information from an image batch."""
    CATEGORY = "Livepeer/Utils"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "height", "width", "count")
    FUNCTION = "get_batch_info"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    def get_batch_info(self, images):
        # Get dimensions from the image tensor
        # ComfyUI images are in BHWC format (Batch, Height, Width, Channels)
        batch_size = images.shape[0]
        height = images.shape[1]
        width = images.shape[2]
        
        # Return original images and the extracted dimensions
        return (images, height, width, batch_size)

# Mappings for __init__.py
NODE_CLASS_MAPPINGS = {
    "BatchInfo": BatchInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchInfo": "Batch Info",
} 