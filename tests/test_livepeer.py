#!/usr/bin/env python
import argparse
import sys
import os
import json
from pathlib import Path
from livepeer_ai import Livepeer
from livepeer_ai.models import components
import urllib.request
from PIL import Image

# Add parent directory to path to import from config_manager
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from config_manager import DEFAULT_CONFIG
config_path = os.path.join(parent_dir, "config.json")

# Default test image URL/path for image-based endpoints
DEFAULT_TEST_IMAGE_URL = "https://picsum.photos/seed/livepeer/512/512"
DEFAULT_TEST_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_image.jpg")

def get_api_key():
    """Get the API key from config.json"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            api_key = config.get("api_key", "")
            if not api_key:
                print("API key not found in config.json.")
                return None
            return api_key
    except Exception as e:
        print(f"Error loading config.json: {e}")
        return None

def get_default_model(job_type):
    """Get the default model for a specific job type from DEFAULT_CONFIG"""
    default_models = DEFAULT_CONFIG.get("default_models", {})
    return default_models.get(job_type, "")

def get_test_image():
    """Download or get test image for image-based tests"""
    # Create test image if it doesn't exist
    if not os.path.exists(DEFAULT_TEST_IMAGE_PATH):
        print(f"Downloading test image from {DEFAULT_TEST_IMAGE_URL}")
        try:
            Path(os.path.dirname(DEFAULT_TEST_IMAGE_PATH)).mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(DEFAULT_TEST_IMAGE_URL, DEFAULT_TEST_IMAGE_PATH)
        except Exception as e:
            print(f"Error downloading test image: {e}")
            return None
    
    # Return image as components.Image object
    try:
        image = components.Image(
            file_name="test_image.jpg",
            content=open(DEFAULT_TEST_IMAGE_PATH, "rb")
        )
        return image
    except Exception as e:
        print(f"Error loading test image: {e}")
        return None

def test_text_to_image(livepeer):
    """Test the text-to-image endpoint"""
    print("Testing text-to-image generation...")
    
    # Get default model for text-to-image
    default_model = get_default_model("T2I")
    print(f"Using model: {default_model or 'Default model'}")
    
    try:
        res = livepeer.generate.text_to_image(request={
            "prompt": "A photo of a cat in space",
            "model_id": default_model,
            "height": 576,
            "width": 1024,
            "guidance_scale": 7.5,
            "negative_prompt": "",
            "safety_check": True,
            "num_inference_steps": 50,
            "num_images_per_prompt": 1,
        })
        
        if res.image_response and res.image_response.images:
            print("✅ Text-to-image test successful!")
            for i, img in enumerate(res.image_response.images):
                print(f"  Image {i+1}: {img.url}")
            return True
        else:
            print("❌ No images returned in response")
            return False
            
    except Exception as e:
        print(f"❌ Error in text-to-image test: {e}")
        return False

def test_image_to_image(livepeer):
    """Test the image-to-image endpoint"""
    print("Testing image-to-image transformation...")
    
    # Get default model for image-to-image
    default_model = get_default_model("I2I")
    print(f"Using model: {default_model or 'Default model'}")
    
    image = get_test_image()
    if not image:
        return False
    
    try:
        res = livepeer.generate.image_to_image(request={
            "prompt": "A watercolor painting in vibrant colors",
            "image": image,
            "model_id": default_model,
            "loras": "",
            "strength": 0.8,
            "guidance_scale": 7.5,
            "image_guidance_scale": 1.5,
            "negative_prompt": "",
            "safety_check": True,
            "num_inference_steps": 100,
            "num_images_per_prompt": 1,
        })
        
        if res.image_response and res.image_response.images:
            print("✅ Image-to-image test successful!")
            for i, img in enumerate(res.image_response.images):
                print(f"  Image {i+1}: {img.url}")
            return True
        else:
            print("❌ No images returned in response")
            return False
            
    except Exception as e:
        print(f"❌ Error in image-to-image test: {e}")
        return False

def test_image_to_video(livepeer):
    """Test the image-to-video endpoint"""
    print("Testing image-to-video generation...")
    
    # Get default model for image-to-video
    default_model = get_default_model("I2V")
    print(f"Using model: {default_model or 'Default model'}")
    
    image = get_test_image()
    if not image:
        return False
    
    try:
        res = livepeer.generate.image_to_video(request={
            "image": image,
            "model_id": default_model,
            "height": 576,
            "width": 1024,
            "fps": 6,
            "motion_bucket_id": 127,
            "noise_aug_strength": 0.02,
            "safety_check": True,
            "num_inference_steps": 25,
        })
        
        if res.video_response and res.video_response.url:
            print("✅ Image-to-video test successful!")
            print(f"  Video URL: {res.video_response.url}")
            return True
        else:
            print("❌ No video returned in response")
            return False
            
    except Exception as e:
        print(f"❌ Error in image-to-video test: {e}")
        return False

def test_upscale(livepeer):
    """Test the image upscale endpoint"""
    print("Testing image upscaling...")
    
    # Get default model for upscale
    default_model = get_default_model("upscale")
    print(f"Using model: {default_model or 'Default model'}")
    
    image = get_test_image()
    if not image:
        return False
    
    try:
        res = livepeer.generate.upscale(request={
            "prompt": "High-quality detailed image",
            "image": image,
            "model_id": default_model,
            "safety_check": True,
            "num_inference_steps": 75,
        })
        
        if res.image_response and res.image_response.images:
            print("✅ Image upscale test successful!")
            for i, img in enumerate(res.image_response.images):
                print(f"  Image {i+1}: {img.url}")
            return True
        else:
            print("❌ No images returned in response")
            return False
            
    except Exception as e:
        print(f"❌ Error in image upscale test: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Livepeer AI API endpoints")
    parser.add_argument("endpoint", nargs="?", choices=["t2i", "i2i", "i2v", "upscale", "all"], 
                      default="all", help="Endpoint to test (default: all)")
    args = parser.parse_args()
    
    # Get API key
    api_key = get_api_key()
    if not api_key:
        print("No API key found. Please check your config.json file.")
        return 1
    
    print(f"Using Livepeer API with key: {api_key[:8]}...")
    
    # Test selected endpoint(s)
    try:
        with Livepeer(http_bearer=api_key) as livepeer:
            results = {}
            
            if args.endpoint in ["t2i", "all"]:
                results["t2i"] = test_text_to_image(livepeer)
                
            if args.endpoint in ["i2i", "all"]:
                results["i2i"] = test_image_to_image(livepeer)
                
            if args.endpoint in ["i2v", "all"]:
                results["i2v"] = test_image_to_video(livepeer)
                
            if args.endpoint in ["upscale", "all"]:
                results["upscale"] = test_upscale(livepeer)
            
            # Print summary
            if args.endpoint == "all":
                print("\n====== Test Summary ======")
                for endpoint, success in results.items():
                    status = "✅ PASS" if success else "❌ FAIL"
                    print(f"{endpoint}: {status}")
                    
            # Return success if all tests passed
            return 0 if all(results.values()) else 1
                
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 