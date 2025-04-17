from livepeer_ai.models.operations import (
    GenImageToTextResponse,
    GenLLMResponse,
    GenTextToImageResponse, 
    GenImageToImageResponse,
    GenUpscaleResponse,
    GenSegmentAnything2Response,
    GenImageToVideoResponse,
    GenTextToSpeechResponse,
    GenLiveVideoToVideoResponse
)
from ..config_manager import config_manager

class LivepeerResponseHandler:
    """
    Utility class for handling different Livepeer API response types.
    Responsible for response type checking and data extraction.
    """
    
    @staticmethod
    def extract_image_data(job_id, job_type, raw_result):
        """
        Checks if raw_result contains valid image data and extracts relevant information.
        Returns a tuple: (has_image_data, response_object)
        """
        if raw_result is None:
            config_manager.log("error", f"Job {job_id} ({job_type}) returned None result")
            return False, None
            
        # Handle text-to-image responses
        if job_type == "t2i" and isinstance(raw_result, GenTextToImageResponse):
            if hasattr(raw_result, 'image_response') and raw_result.image_response:
                return True, raw_result
                
        # Handle image-to-image responses
        elif job_type == "i2i" and isinstance(raw_result, GenImageToImageResponse):
            if hasattr(raw_result, 'image_response') and raw_result.image_response:
                return True, raw_result
                
        # Handle upscale responses
        elif job_type == "upscale" and isinstance(raw_result, GenUpscaleResponse):
            if hasattr(raw_result, 'image_response') and raw_result.image_response:
                return True, raw_result
                
        # Handle segment responses
        elif job_type == "segment" and isinstance(raw_result, GenSegmentAnything2Response):
            if hasattr(raw_result, 'image_response') and raw_result.image_response:
                return True, raw_result
        
        # Fallback for any response with image_response
        elif hasattr(raw_result, 'image_response') and raw_result.image_response:
            config_manager.log("warning", f"Job {job_id} returned response type {type(raw_result).__name__} for job type {job_type}")
            return True, raw_result
            
        # No valid image data found
        config_manager.log("error", f"Job {job_id} ({job_type}) has no valid image_response in its result. Response type: {type(raw_result).__name__}")
        return False, None

    @staticmethod
    def extract_video_data(job_id, job_type, raw_result):
        """
        Checks if raw_result contains valid video data and extracts relevant information.
        Returns a tuple: (has_video_data, response_object, urls)
        """
        if raw_result is None:
            config_manager.log("error", f"Job {job_id} ({job_type}) returned None result")
            return False, None, []
            
        # Extract video URLs from response if available
        video_urls = []
        if hasattr(raw_result, 'video_response') and hasattr(raw_result.video_response, 'images'):
            video_urls = [image.url for image in raw_result.video_response.images]
        
        # Image-to-video responses
        if job_type == "i2v" and isinstance(raw_result, GenImageToVideoResponse):
            if hasattr(raw_result, 'video_response') and raw_result.video_response:
                return True, raw_result, video_urls
        
        # Live-to-video responses
        elif job_type == "live2video" and isinstance(raw_result, GenLiveVideoToVideoResponse):
            if hasattr(raw_result, 'video_response') and raw_result.video_response:
                return True, raw_result, video_urls
        
        # Fallback for any response with video_response
        elif hasattr(raw_result, 'video_response') and raw_result.video_response:
            config_manager.log("warning", f"Job {job_id} returned response type {type(raw_result).__name__} for job type {job_type}")
            return True, raw_result, video_urls
            
        # No valid video data found
        config_manager.log("error", f"Job {job_id} ({job_type}) has no valid video_response in its result. Response type: {type(raw_result).__name__}")
        return False, None, []

    @staticmethod
    def extract_text_data(job_id, job_type, raw_result):
        """
        Checks if raw_result contains valid text data and extracts it.
        Returns a tuple: (has_text_data, extracted_text)
        """
        text_out = ""
        
        if raw_result is None:
            config_manager.log("error", f"Job {job_id} ({job_type}) returned None result")
            return False, None
          
        # Handle GenImageToTextResponse (operation response wrapper)
        if isinstance(raw_result, GenImageToTextResponse):
            if (hasattr(raw_result, 'image_to_text_response') and 
                raw_result.image_to_text_response is not None and
                hasattr(raw_result.image_to_text_response, 'text')):
                text_out = str(raw_result.image_to_text_response.text)
                
        # Handle LLMResponse
        elif isinstance(raw_result, GenLLMResponse):
            if hasattr(raw_result, 'choices') and raw_result.choices:
                choice = raw_result.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    text_out = str(choice.message.content)
        
        # Single fallback for string responses
        elif isinstance(raw_result, str):
            text_out = raw_result
            config_manager.log("warning", f"Job {job_id} returned raw string instead of expected response object")
            
        if text_out:
            return True, text_out
        else:
            config_manager.log("error", f"Job {job_id} ({job_type}) did not provide valid text content. Response type: {type(raw_result).__name__}")
            return False, None

    @staticmethod
    def extract_audio_data(job_id, job_type, raw_result):
        """
        Checks if raw_result contains valid audio data and extracts the URL.
        Returns a tuple: (has_audio_data, response_object, audio_url)
        """
        if raw_result is None:
            config_manager.log("error", f"Job {job_id} ({job_type}) returned None result")
            return False, None, None
            
        # Handle T2S responses
        if job_type == "t2s" and isinstance(raw_result, GenTextToSpeechResponse):
            if hasattr(raw_result, 'audio_response') and raw_result.audio_response:
                # Extract audio URL from response (audio is nested inside audio_response)
                if hasattr(raw_result.audio_response, 'audio') and hasattr(raw_result.audio_response.audio, 'url'):
                    audio_url = raw_result.audio_response.audio.url
                    if audio_url:
                        return True, raw_result, audio_url
        
        # Fallback for any response with audio_response
        elif hasattr(raw_result, 'audio_response'):
            config_manager.log("warning", f"Job {job_id} returned response type {type(raw_result).__name__} for job type {job_type}")
            if hasattr(raw_result.audio_response, 'audio') and hasattr(raw_result.audio_response.audio, 'url'):
                audio_url = raw_result.audio_response.audio.url
                if audio_url:
                    return True, raw_result, audio_url
                
        # No valid audio data found
        config_manager.log("error", f"Job {job_id} ({job_type}) has no valid audio_response in its result. Response type: {type(raw_result).__name__}")
        return False, None, None 