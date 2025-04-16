# WORK IN PROGRESS











# ComfyUI Livepeer Integration

This extension provides ComfyUI nodes for Livepeer AI generation endpoints with built-in retry functionality.

## Features

- **Text to Image (T2I)** - Generate images from text prompts
- **Image to Image (I2I)** - Apply transformations to images
- **Image to Video (I2V)** - Generate videos from still images
- **Upscale** - Increase image resolution
- **Image to Text** - Generate captions and descriptions from images

All nodes include:
- Automatic retries with configurable retry count and delay
- Exponential backoff for reliability
- Proper tensor format handling (BHWC for ComfyUI)

## Requirements

```
pip install livepeer_ai requests Pillow numpy torch
```

## API Key

You can use the included demo API key for testing, but for production use, obtain your own API key from [Livepeer](https://livepeer.studio).

## Usage

The nodes can be found in the "livepeer" category in the ComfyUI node menu.

### Common Parameters

All nodes share these parameters:
- `api_key` - Your Livepeer API key
- `max_retries` - Number of times to retry failed API calls
- `retry_delay` - Initial delay between retries (seconds)

### Models

The nodes support various model options. Some common models include:

- Text to Image: `ByteDance/SDXL-Lightning`, `stabilityai/stable-diffusion-xl-base-1.0`
- Image to Image: `stabilityai/stable-diffusion-2-1`
- Image to Video: `stabilityai/stable-video-diffusion-img2vid-xt`
- Upscale: `stabilityai/stable-diffusion-x4-upscaler`
- Image to Text: `Salesforce/blip-image-captioning-large`

## Example Workflows

### Basic Text to Image
1. Add a "Livepeer T2I" node
2. Connect to a "Preview Image" node
3. Enter your prompt and adjust parameters

### Image to Video
1. Create an image using any method
2. Feed the image to a "Livepeer I2V" node
3. The node will return a video URL that can be viewed in a browser

## Troubleshooting

If you encounter API errors:
1. Check your API key
2. Verify your internet connection
3. Increase the `max_retries` parameter
4. Check the Livepeer status page for service issues 