# YouTube Video Downloader

Download a YouTube video as a single MP4 that always contains **both video and
audio**. It grabs the best video and best audio streams separately and muxes
them with ffmpeg, then verifies the result with ffprobe.

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/) available on your `PATH` (provides `ffprobe` too)
- Python deps: `pip install -r requirements.txt`

## Usage

```bash
# Best available quality
python download.py "https://www.youtube.com/watch?v=x_eRIU8qk7s"

# Into a custom folder
python download.py "<url>" -o my_videos

# Cap the resolution (e.g. 720p)
python download.py "<url>" --quality 720
```

The script prints the final file path and confirms that both a video and an
audio stream are present. It exits non-zero if either is missing.
