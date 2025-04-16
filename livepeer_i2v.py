from .livepeer_core import LivepeerBase
from livepeer_ai.models import components
import json

class LivepeerI2V(LivepeerBase):
    JOB_TYPE = "i2v" # Define job type for async tracking

    @classmethod
    def INPUT_TYPES(s):
        # Get common inputs first
        common_inputs = s.get_common_inputs()
        # Define node-specific inputs
        node_inputs = {
            "required": {
                "image": ("IMAGE", ),
            },
            "optional": {
                 "model_id": ("STRING", {"multiline": False, "default": "stabilityai/stable-video-diffusion-img2vid-xt-1-1"}), # e.g., stabilityai/stable-video-diffusion-img2vid-xt
                 "height": ("INT", {"default": 576, "min": 64, "max": 2048, "step": 64}),
                 "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                 "fps": ("INT", {"default": 6, "min": 1, "max": 60, "step": 1}),
                 "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 255}),
                 "noise_aug_strength": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
                 "safety_check": ("BOOLEAN", {"default": True}),
                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # Use 0 for None/random
                 "num_inference_steps": ("INT", {"default": 25, "min": 1, "max": 100, "step": 1}),
                 "download_video": ("BOOLEAN", {"default": True}), # New option to download video
             }
        }
        # Add common inputs into the 'optional' category
        if "optional" not in node_inputs:
            node_inputs["optional"] = {}
        node_inputs["optional"].update(common_inputs)
        return node_inputs

    RETURN_TYPES = ("STRING", "STRING", "STRING") # Video URL, file path, job_id
    RETURN_NAMES = ("video_url", "video_path", "job_id")
    FUNCTION = "image_to_video"
    CATEGORY = "Livepeer"

    def image_to_video(self, api_key, max_retries, retry_delay, run_async, image, model_id="", height=576, width=1024, fps=6, motion_bucket_id=127, noise_aug_strength=0.02, safety_check=True, seed=0, num_inference_steps=25, download_video=True):

        # Prepare the input image using the base class method
        livepeer_image = self.prepare_image(image)
        
        # Convert the Image object to the dictionary format expected by BodyGenImageToVideo
        image_dict = {
            "file_name": livepeer_image.file_name,
            "content": livepeer_image.content
        }

        # Prepare arguments for the Livepeer API call using all parameters
        # Note: SDK uses BodyGenImageToVideo as the request object
        i2v_args = components.BodyGenImageToVideo(
            image=image_dict,
            model_id=model_id if model_id else None,
            height=height,
            width=width,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            safety_check=safety_check,
            seed=seed if seed != 0 else None,
            num_inference_steps=num_inference_steps
        )

        # Define the operation function for retry/async logic
        def operation_func(livepeer):
            # Explicitly use livepeer.generate based on SDK structure
            return livepeer.generate.image_to_video(request=i2v_args)

        if run_async:
            # Trigger async job and return job ID
            job_id = self.trigger_async_job(api_key, max_retries, retry_delay, operation_func, self.JOB_TYPE)
            # Return None for video_url, None for video_path, job_id for the string output
            return (None, None, job_id)
        else:
            # Execute synchronously with retry logic
            response = self.execute_with_retry(api_key, max_retries, retry_delay, operation_func)

            # Process the response to get video URLs
            video_urls = self.process_video_response(response)
            video_url = video_urls[0] if video_urls else ""
            
            # Download video if requested
            video_path = None
            if download_video and video_url:
                try:
                    video_path = self.download_video(video_url)
                except Exception as e:
                    print(f"Error downloading video: {e}")
                    video_path = f"Error: {str(e)}"
            
            # Return video URL, video_path, None for job_id
            return (video_url, video_path, None)

NODE_CLASS_MAPPINGS = {
    "LivepeerI2V": LivepeerI2V,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LivepeerI2V": "Livepeer I2V",
} 