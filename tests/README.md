# Livepeer API Testing

This directory contains a test script for validating Livepeer API endpoints. These scripts are WIP


## Running Tests

The test script can run tests for individual endpoints or all supported endpoints at once.

```bash
# Test all endpoints
python test_livepeer.py all

# Test specific endpoints
python test_livepeer.py t2i  # Text-to-image
python test_livepeer.py i2i  # Image-to-image
python test_livepeer.py i2v  # Image-to-video
python test_livepeer.py upscale  # Image upscaling
```

## What the Tests Do

Each test makes a basic API call to the corresponding Livepeer endpoint:

- **t2i**: Generates an image from a text prompt
- **i2i**: Transforms a test image using a text prompt
- **i2v**: Creates a video from a test image
- **upscale**: Upscales a test image to higher resolution

The script will automatically download a test image if needed for the image-based tests.

## Understanding Test Results

The script provides clear feedback about each test:

- ✅ indicates a successful test
- ❌ indicates a failed test

For failed tests, error messages are displayed to help diagnose the issue.

## Troubleshooting

If tests are failing, check the following:

1. Verify your API key in the `config.json` file is valid and has the necessary permissions
2. Check your internet connection
3. Confirm Livepeer services are online
4. For image-based tests, ensure the script can download or access the test image 