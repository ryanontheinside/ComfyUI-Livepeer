import numpy as np
import torch
from PIL import Image as PILImage
from io import BytesIO
import requests
import os
import shutil
import time
import tempfile
import soundfile as sf
import cv2
import subprocess
import json
from livepeer_ai.models.components import Image

from ..config_manager import config_manager

class LivepeerMediaProcessor:
    """Utility class for processing media responses from Livepeer API"""
    
    @staticmethod
    def process_image_response(response):
        """Process image response into BHWC tensor format"""
        try:
            images = []
            for img_data in response.image_response.images:
                image_url = img_data.url
                
                # Get image data
                img_response = requests.get(image_url).content
                img = PILImage.open(BytesIO(img_response)).convert("RGB")
                
                # Convert to numpy array with proper normalization
                img_np = np.array(img).astype(np.float32) / 255.0
                images.append(img_np)
            
            # Stack into batch tensor [B, H, W, C]
            img_batch = np.stack(images, axis=0)
            return torch.from_numpy(img_batch)
        except Exception as e:
            return config_manager.handle_error(e, "Error processing image response")
    
    @staticmethod
    def download_media(url, output_type="videos"):
        """
        Download media from URL to ComfyUI output directory
        
        Args:
            url: Media URL to download
            output_type: Type of media (videos, audio, etc.)
            
        Returns:
            str: Full path to downloaded file
        """
        try:
            # Use configured output directory
            output_dir = config_manager.get_output_path(output_type)
            
            # Generate unique filename using timestamp
            timestamp = int(time.time())
            
            # Determine file extension based on output_type
            extension = "mp4" if output_type == "videos" else "mp3"
            filename = f"livepeer_{output_type[:-1]}_{timestamp}.{extension}"
            file_path = os.path.join(output_dir, filename)
            
            # Download the file
            config_manager.log("info", f"Downloading {output_type[:-1]} from {url} to {file_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
                
            return file_path
        except Exception as e:
            return config_manager.handle_error(e, f"Error downloading {output_type}")
    
    @staticmethod
    def prepare_image(image_batch):
        """
        Convert ComfyUI image tensor batch to file for API upload
        """
        try:
            # For now, we can only process one image at a time with the Livepeer API
            if image_batch.shape[0] > 1:
                config_manager.log("warning", f"Livepeer API only supports one image at a time. Using first image from batch of {image_batch.shape[0]}")
            
            # Convert tensor to PIL Image for uploading
            img = image_batch[0]  # First image in batch
            pil_img = torch.clamp(img * 255, 0, 255).cpu().numpy().astype(np.uint8)
            pil_img = PILImage.fromarray(pil_img)
            
            # Save to BytesIO for uploading
            img_byte_arr = BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Convert BytesIO to bytes for the Livepeer API
            img_bytes = img_byte_arr.getvalue()
            
            # Create file-like object for the API using the correct Image class
            return Image(
                file_name="input_image.png",
                content=img_bytes
            )
        except Exception as e:
            return config_manager.handle_error(e, "Error preparing image")
    
    @staticmethod
    def extract_video_urls(response):
        """Extract video URLs from video response"""
        try:
            if hasattr(response, 'video_response') and hasattr(response.video_response, 'images'):
                return [image.url for image in response.video_response.images]
            else:
                return []
        except Exception as e:
            return config_manager.handle_error(e, "Error extracting video URLs")
            
    @staticmethod
    def handle_audio_format(audio_path, raw_result):
        """Update audio file format based on response metadata"""
        try:
            if hasattr(raw_result, 'format') and raw_result.format:
                base_ext = os.path.splitext(audio_path)[0]
                new_path = f"{base_ext}.{raw_result.format}"
                os.rename(audio_path, new_path)
                return new_path
            return audio_path
        except Exception as e:
            config_manager.handle_error(e, "Error handling audio format", raise_error=False)
            return audio_path
            
    @staticmethod
    def prepare_audio_from_comfy_format(audio_data):
        """
        Convert ComfyUI audio format to a temporary file for API processing
        
        Args:
            audio_data: ComfyUI audio format {'waveform': audio, 'sample_rate': ar}
            
        Returns:
            tuple: (temp_file_path, file_handle) or (None, None) if conversion fails
        """
        try:
            # Validate audio format
            if not isinstance(audio_data, dict) or 'waveform' not in audio_data or 'sample_rate' not in audio_data:
                config_manager.log("error", "Invalid audio format. Expected {'waveform': audio_data, 'sample_rate': sample_rate}")
                return None, None

            # Create two temporary files - one for raw WAV, one for processed MP3
            temp_dir = tempfile.mkdtemp()
            temp_wav_path = os.path.join(temp_dir, "temp_audio.wav")
            temp_mp3_path = os.path.join(temp_dir, "temp_audio.mp3")
            
            try:
                # Extract audio data
                waveform = audio_data['waveform']
                sample_rate = audio_data['sample_rate']
                
                # Process the waveform tensor to match what soundfile expects
                if torch.is_tensor(waveform):
                    # Move tensor to CPU if it's on another device
                    waveform = waveform.cpu()
                    
                    # Handle the 3D tensor format from ComfyUI [batch, channels, samples]
                    if len(waveform.shape) == 3:
                        # Take the first batch
                        waveform = waveform[0]  # Now shape is [channels, samples]
                        
                        # Convert to numpy and transpose to [samples, channels] for soundfile
                        waveform = waveform.numpy().T
                    elif len(waveform.shape) == 2:
                        # Assume shape is [channels, samples]
                        waveform = waveform.numpy().T  # Transpose to [samples, channels]
                    else:
                        # Handle unexpected shape
                        config_manager.log("warning", f"Unexpected waveform shape: {waveform.shape}. Expected 3D or 2D tensor.")
                        waveform = waveform.numpy()
                
                # Ensure we have at least 1 second of audio data (pad with silence if needed)
                min_samples = sample_rate
                if waveform.shape[0] < min_samples:
                    if len(waveform.shape) == 1:  # Mono
                        padding = np.zeros(min_samples - waveform.shape[0], dtype=np.float32)
                        waveform = np.concatenate([waveform, padding])
                    else:  # Stereo or multi-channel
                        padding = np.zeros((min_samples - waveform.shape[0], waveform.shape[1]), dtype=np.float32)
                        waveform = np.concatenate([waveform, padding])
                
                # Save as intermediate WAV file
                sf.write(temp_wav_path, waveform, sample_rate)
                
                # Convert to MP3 using FFmpeg (more universally compatible)
                cmd = [
                    "ffmpeg", "-y", 
                    "-i", temp_wav_path,
                    "-c:a", "libmp3lame",
                    "-b:a", "192k",  # High quality
                    "-ar", str(sample_rate),
                    temp_mp3_path
                ]
                
                # Run FFmpeg, capturing output to check for errors
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
                
                # Check if output file exists and has content
                if not os.path.exists(temp_mp3_path) or os.path.getsize(temp_mp3_path) == 0:
                    raise RuntimeError("FFmpeg produced an empty or missing output file")
                
                # Instead of reading the file as bytes, return the path and let the SDK handle it
                # This avoids potential issues with how the multipart form data is constructed
                
                # Return path to the processed MP3 file
                # IMPORTANT: Make sure we're returning a file with .mp3 extension to help the SDK determine content type
                return temp_mp3_path, open(temp_mp3_path, 'rb')
                
            except Exception as e:
                error_msg = str(e)
                config_manager.log("error", f"Error preparing audio: {error_msg}")
                # Clean up if something goes wrong
                raise e
                
            finally:
                # Clean up temporary files
                try:
                    if os.path.exists(temp_wav_path):
                        os.unlink(temp_wav_path)
                    if os.path.exists(temp_mp3_path) and not temp_mp3_path == temp_wav_path:
                        # Don't delete the MP3 file yet if it's being returned
                        if not temp_mp3_path == temp_wav_path:
                            pass
                    if os.path.exists(temp_dir):
                        # Only try to remove the dir if it's empty (it might not be if we're returning the MP3)
                        try:
                            os.rmdir(temp_dir)
                        except:
                            pass
                except Exception as cleanup_error:
                    config_manager.log("warning", f"Error cleaning up temp files: {str(cleanup_error)}")
                
        except Exception as e:
            config_manager.handle_error(e, "Error preparing audio from ComfyUI format", raise_error=False)
            return None, None
    
    @staticmethod
    def extract_audio_from_video(video_path):
        """
        Extract audio from video file using FFmpeg
        
        Args:
            video_path: Path to video file
            
        Returns:
            dict: ComfyUI audio format {'waveform': audio, 'sample_rate': ar} or empty audio if no audio track
        """
        try:
            # Create temporary file for audio
            temp_dir = tempfile.mkdtemp()
            temp_audio_path = os.path.join(temp_dir, "audio.wav")
            
            try:
                # Check if video has audio stream
                probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", 
                             "-of", "json", video_path]
                result = subprocess.run(probe_cmd, capture_output=True, text=True)
                data = json.loads(result.stdout)
                
                has_audio = False
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'audio':
                        has_audio = True
                        break
                
                if not has_audio:
                    config_manager.log("warning", f"No audio track found in video: {video_path}")
                    # Return empty audio in ComfyUI format instead of None
                    return {
                        'waveform': torch.zeros((1, 2, 1), dtype=torch.float32),  # Empty stereo waveform, shape [1, 2, 1]
                        'sample_rate': 44100  # Standard sample rate
                    }
                
                # Extract audio using FFmpeg
                cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", 
                       "-ar", "44100", "-ac", "2", temp_audio_path, "-y"]
                subprocess.run(cmd, capture_output=True)
                
                # Load audio file
                if os.path.exists(temp_audio_path):
                    audio_data, sample_rate = sf.read(temp_audio_path)
                    
                    # Convert to float32 if not already
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    
                    # Handle dimensions for torchaudio compatibility (channels first: [channels, samples])
                    if len(audio_data.shape) == 1:  # Mono audio
                        # Reshape to [1, samples] - one channel
                        audio_data = audio_data.reshape(1, -1)
                        audio_tensor = torch.from_numpy(audio_data)
                        audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension [1, 1, samples]
                    else:  # Stereo or multichannel audio
                        # Transpose from [samples, channels] to [channels, samples]
                        audio_data = audio_data.T
                        audio_tensor = torch.from_numpy(audio_data)
                        audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension [1, channels, samples]
                    
                    # Return in ComfyUI format
                    return {
                        'waveform': audio_tensor,
                        'sample_rate': sample_rate
                    }
                else:
                    config_manager.log("error", f"Failed to extract audio from {video_path}")
                    # Return empty audio in ComfyUI format instead of None
                    return {
                        'waveform': torch.zeros((1, 2, 1), dtype=torch.float32),  # Empty stereo waveform, shape [1, 2, 1]
                        'sample_rate': 44100  # Standard sample rate
                    }
            
            finally:
                # Clean up temporary files
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
        
        except Exception as e:
            config_manager.handle_error(e, f"Error extracting audio from video {video_path}", raise_error=False)
            # Return empty audio in ComfyUI format instead of None on errors
            return {
                'waveform': torch.zeros((1, 2, 1), dtype=torch.float32),  # Empty stereo waveform, shape [1, 2, 1]
                'sample_rate': 44100  # Standard sample rate
            }
            
    @staticmethod
    def load_audio_to_tensor(audio_path):
        """
        Load an audio file and convert it to the ComfyUI audio format
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: ComfyUI audio format {'waveform': tensor, 'sample_rate': sample_rate} or None if loading failed
        """
        try:
            if not os.path.exists(audio_path):
                config_manager.log("error", f"Audio file not found: {audio_path}")
                return None
                
            # Load audio file
            waveform, sample_rate = sf.read(audio_path)
            
            # Convert to float32 if not already
            if waveform.dtype != np.float32:
                waveform = waveform.astype(np.float32)
                
            # Handle dimensions for ComfyUI audio format [batch, channels, samples]
            if len(waveform.shape) == 1:  # Mono audio
                # Reshape to [1, 1, samples] - one channel with batch dimension
                waveform = waveform.reshape(1, -1)  # First to [1, samples]
                waveform_tensor = torch.from_numpy(waveform)
                waveform_tensor = waveform_tensor.unsqueeze(0)  # To [1, 1, samples]
            else:  # Stereo or multichannel audio
                # From [samples, channels] to [channels, samples] to [batch, channels, samples]
                waveform = waveform.transpose()  # To [channels, samples]
                waveform_tensor = torch.from_numpy(waveform)
                waveform_tensor = waveform_tensor.unsqueeze(0)  # Add batch dimension [1, channels, samples]
            
            # Create ComfyUI audio dictionary format
            return {
                'waveform': waveform_tensor,
                'sample_rate': sample_rate
            }
                
        except Exception as e:
            error_msg = config_manager.handle_error(e, f"Error loading audio file {audio_path}", raise_error=False)
            config_manager.log("error", f"Failed to load audio: {error_msg}")
            return None
    
    @staticmethod
    def load_video_to_tensor(video_path, max_frames=None, frame_step=1, extract_audio=True):
        """
        Load a video file and convert it to a tensor in BHWC format
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to load (None for all frames)
            frame_step: Only load every Nth frame (e.g., 1 = every frame, 2 = every other frame)
            extract_audio: Whether to extract audio track (if present)
            
        Returns:
            dict: {
                'frames': torch.Tensor in BHWC format [batch, height, width, channels],
                'fps': original fps of the video,
                'frame_count': total number of frames loaded,
                'duration': duration of video in seconds,
                'audio': audio data in ComfyUI format {'waveform': audio, 'sample_rate': ar} (None if no audio or extraction failed)
            }
        """
        try:
            if not os.path.exists(video_path):
                config_manager.log("error", f"Video file not found: {video_path}")
                return None
                
            # Open video file
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Limit frame count if needed
            if max_frames is not None:
                frame_count = min(frame_count, max_frames * frame_step)
                
            # Calculate frames to extract with frame_step
            frames_to_extract = (frame_count + frame_step - 1) // frame_step
            
            # Initialize frame tensors list
            frames = []
            
            # Read frames
            frame_idx = 0
            frames_read = 0
            
            while cap.isOpened() and (max_frames is None or frames_read < max_frames):
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                # Only process frames based on frame_step
                if frame_idx % frame_step == 0:
                    # Convert BGR (OpenCV) to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Normalize to 0-1 float values
                    frame_normalized = frame_rgb.astype(np.float32) / 255.0
                    
                    # Append to frames list
                    frames.append(frame_normalized)
                    frames_read += 1
                
                frame_idx += 1
                
                # Check if we've reached the max_frames
                if max_frames is not None and frames_read >= max_frames:
                    break
            
            # Release the video capture object
            cap.release()
            
            # Extract audio if requested
            audio_data = None
            if extract_audio:
                audio_data = LivepeerMediaProcessor.extract_audio_from_video(video_path)
            
            # Convert list to tensor
            if frames:
                # Stack frames into BHWC format [batch, height, width, channels]
                frames_tensor = np.stack(frames, axis=0)
                tensor = torch.from_numpy(frames_tensor)
                
                # Return video info
                return {
                    'frames': tensor,                   # Tensor in BHWC format
                    'fps': fps,                         # Original FPS
                    'frame_count': len(frames),         # Actual frames extracted
                    'duration': frame_count / fps,      # Duration in seconds
                    'audio': audio_data                 # Audio data in ComfyUI format (or None)
                }
            else:
                config_manager.log("error", f"No frames could be extracted from {video_path}")
                return None
                
        except Exception as e:
            config_manager.handle_error(e, f"Error loading video file {video_path}", raise_error=False)
            return None 