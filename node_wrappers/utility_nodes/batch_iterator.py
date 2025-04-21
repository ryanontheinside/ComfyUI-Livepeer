import torch
import time

# Global state to track indices for different node instances
_batch_iterators = {}
_batch_iterators_lock = {}

class BatchIterator:
    """Node that iterates through an image batch, advancing by 1 each execution."""
    CATEGORY = "Livepeer/Utils"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_output",)
    FUNCTION = "iterate_batch"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "bounce_mode": ("BOOLEAN", {"default": False}),
                "reset_counter": ("BOOLEAN", {"default": False})
            }
        }

    @classmethod
    def IS_CHANGED(cls, images, bounce_mode=False, reset_counter=False):
        """Forces the node to re-execute each time."""
        return time.time()

    def iterate_batch(self, images, bounce_mode=False, reset_counter=False):
        # Generate a unique ID for this instance based on object ID
        instance_id = id(self)
        
        # Get batch size (first dimension of tensor)
        batch_size = images.shape[0]
        
        # Initialize index tracker if needed
        if instance_id not in _batch_iterators:
            _batch_iterators[instance_id] = {"index": 0, "direction": 1, "last_batch_size": batch_size}
            _batch_iterators_lock[instance_id] = False
        
        # Reset counter if requested or if batch size changed
        if reset_counter or batch_size != _batch_iterators[instance_id]["last_batch_size"]:
            _batch_iterators[instance_id]["index"] = 0
            _batch_iterators[instance_id]["direction"] = 1
            _batch_iterators[instance_id]["last_batch_size"] = batch_size
        
        # Get current state
        state = _batch_iterators[instance_id]
        index = state["index"]
        direction = state["direction"]
        
        # Ensure index is in bounds
        index = max(0, min(index, batch_size - 1))
        
        # Select the current image from the batch
        current_image = images[index:index+1]
        
        # Update index for next execution based on mode
        if bounce_mode:
            # Change direction if we hit the boundary
            if index == 0 and direction < 0:
                direction = 1
            elif index >= batch_size - 1 and direction > 0:
                direction = -1
            
            # Move in the current direction
            index += direction
        else:
            # Cycle mode - increment and wrap around
            index = (index + 1) % batch_size
        
        # Store updated state
        _batch_iterators[instance_id]["index"] = index
        _batch_iterators[instance_id]["direction"] = direction
        _batch_iterators[instance_id]["last_batch_size"] = batch_size
        
        return (current_image,)

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
    
NODE_CLASS_MAPPINGS = {
    "BatchIterator": BatchIterator,
    "BatchInfo": BatchInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchIterator": "Batch Image Iterator",
    "BatchInfo": "Batch Info",
}
