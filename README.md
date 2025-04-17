# WORK IN PROGRESS











# ComfyUI-Livepeer

A ComfyUI extension that provides integration with [Livepeer](https://livepeer.org/)'s AI services allowing for both sync and async generation. 

## Features

This extension provides ComfyUI nodes for:

- Text-to-Image generation (LivepeerT2I)
- Image-to-Image transformations (LivepeerI2I)
- Image-to-Video generation (LivepeerI2V)
- Image-to-Text (captioning) (LivepeerI2T)
- Audio-to-Text (transcription) (LivepeerA2T)
- Text-to-Speech synthesis (LivepeerT2S)
- Image Upscaling (LivepeerUpscale)
- Image Segmentation (LivepeerSegment)
- Large Language Model interface (LivepeerLLM)
- Live Video-to-Video transformations (LivepeerLive2Video)

The extension also provides getter nodes to retrieve and process the results:

- LivepeerImageJobGetter - Gets image output from relevant jobs
- LivepeerVideoJobGetter - Gets video output from relevant jobs
- LivepeerTextJobGetter - Gets text output from relevant jobs
- LivepeerAudioJobGetter - Gets audio output from relevant jobs

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory:
```
cd ComfyUI/custom_nodes
git clone https://github.com/your-username/ComfyUI-Livepeer.git
```

2. Install the dependencies:
```
pip install livepeer-ai
```

3. Restart ComfyUI

## Configuration

This extension uses a configuration file to manage settings. The default configuration file is located at `custom_nodes/ComfyUI-Livepeer/config.json` and is created automatically when the extension is first loaded.

You can customize the following settings:

### API Key
```json
"api_key": "your-api-key-here"
```
Set your Livepeer API key. This will be used as the default if not specified in the node.

### Logging
```json
"log_level": "INFO"
```
Set the log level. Options are: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

### Error Handling
```json
"error_handling": {
    "throw_errors": false,
    "max_error_logs": 10,
    "log_errors_to_file": true
}
```
- `throw_errors` - If true, errors will be raised instead of just logged. This can help with debugging but may interrupt workflow execution.
- `max_error_logs` - Maximum number of error log files to keep
- `log_errors_to_file` - Whether to log errors to a file (in `custom_nodes/ComfyUI-Livepeer/logs/`)

### Output Paths
```json
"output_paths": {
    "images": "output/livepeer/images",
    "videos": "output/livepeer/videos",
    "audio": "output/livepeer/audio"
}
```
Configure where downloaded media files will be saved. Paths are relative to the ComfyUI root directory.

### Retry Settings
```json
"default_retry_settings": {
    "max_retries": 3,
    "retry_delay": 2.0
},
"default_timeout": 120.0
```
- `max_retries` - Number of times to retry failed API calls
- `retry_delay` - Initial delay between retries in seconds (exponential backoff is applied)
- `default_timeout` - Default timeout for API calls in seconds

## Usage

### Basic Workflow Example

1. **Text-to-Image Generation**:
   - Add a LivepeerT2I node
   - Connect the output to a LivepeerImageJobGetter node
   - The getter node will output the generated image

2. **Image-to-Video Generation**:
   - Add a LivepeerI2V node
   - Connect an image input (e.g., from LoadImage)
   - Connect the output to a LivepeerVideoJobGetter node
   - The getter will provide the video URL and can download the video

### Asynchronous Processing

All nodes support asynchronous processing, which prevents ComfyUI from freezing during long-running operations:

1. Set the `run_async` parameter to `true` on any node
2. The node will return immediately with a job ID
3. Use the appropriate getter node to check the status and retrieve results

### Error Handling

Errors can be handled in several ways:

1. View errors in the ComfyUI console
2. Check error logs in `custom_nodes/ComfyUI-Livepeer/logs/`
3. The getter nodes provide status and error message outputs

## License

[MIT](LICENSE) 