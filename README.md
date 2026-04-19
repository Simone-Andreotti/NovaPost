# NovaPost

NovaPost is a local automation app that turns one media pack into platform-ready content for:

- X post text + image plan
- TikTok video with generated voice-over + burned subtitles
- LinkedIn square carousel PDF

It can be run via terminal or via GUI

The pipeline is fully local except LLM/TTS APIs you enable in `.env`.

The app have only being tested on Windows 11

## What This Project Produces

After a run, generated files are written in `output/`:

- `x_post.txt`
- `generated_copy.json`
- `platform_copy.json`
- `voiceover.wav`
- `subtitles.srt`
- `tiktok_final.mp4`
- `linkedin_carousel.pdf`

## Project Structure

```text
NovaPost/
|-- .env.example
|-- setup.bat
|-- run_gui.bat
|-- requirements.txt
|-- generate.py
|-- web_panel.py
|-- logo.png
|-- assets/               # your local input media (ignored by git)
|   `-- .gitkeep
|-- output/               # generated output (ignored by git)
|   `-- .gitkeep
|-- bin/                  # local ffmpeg.exe + ffprobe.exe used at runtime
|-- ffmpeg-build/         # optional local FFmpeg bundle source
|-- fonts/                # project fonts used for overlays/subtitles
|   |-- nexokora.otf
|   |-- Syncopate-Bold.ttf
|   `-- Syncopate-Regular.ttf
`-- models/
    `-- kokoro/           # local fallback model cache (ignored by git)
```

## Prerequisites

- Python 3.11+ (3.12 recommended)
- Windows, macOS, or Linux
- NVIDIA NIM API key (required)
- Optional: ElevenLabs key for premium TTS

FFmpeg is used from local binaries in `bin/`.
`setup.bat` tries to copy `ffmpeg.exe` and `ffprobe.exe` from `ffmpeg-build/bin/` automatically when available.

## Install FFmpeg Locally (Required)

FFmpeg files are intentionally git-ignored. Each machine must add them locally.

1. Download a Windows FFmpeg build (zip) that includes `ffmpeg.exe` and `ffprobe.exe`.
2. Extract the zip anywhere temporary.
3. Copy `ffmpeg.exe` and `ffprobe.exe` from the extracted `bin/` into `bin/`.
4. Optional alternative: copy the extracted folder into `ffmpeg-build/` and run `setup.bat` to auto-copy binaries to `bin/`.
5. Verify with:

```powershell
Set-Location "c:\path\to\NovaPost"
.\bin\ffmpeg.exe -version
.\bin\ffprobe.exe -version
```

## Quick Start (Windows)

```powershell
Set-Location "c:\path\to\NovaPost"
Copy-Item .env.example .env
# edit .env and set NVIDIA_NIM_API_KEY

.\setup.bat
.\venv\Scripts\python .\generate.py
```

To start GUI panel:

```powershell
.\run_gui.bat
```

## Quick Start (macOS / Linux)

```bash
cd /path/to/NovaPost
cp .env.example .env
# edit .env and set NVIDIA_NIM_API_KEY

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python generate.py
```

GUI:

```bash
python web_panel.py
```

## Configure Environment

Create `.env` from `.env.example`.

Minimum required variable:

```env
NVIDIA_NIM_API_KEY=your_key_here
```

Most useful optional variables:

```env
TTS_BACKEND=auto
TTS_EDGE_VOICE=en-US-ChristopherNeural
TTS_SUBTITLE_FONT_NAME=Syncopate
TTS_SUBTITLE_FONT_SIZE=34
ELEVENLABS_API_KEY=
```

## Prepare Inputs

Put your media in `assets/`:

1. Exactly 1 `.mp4` file
2. 1 or more images (`.png`, `.jpg`, `.jpeg`)
3. One `prompt.md`

Naming images with prefixes is recommended for stable order:

- `01_cover.png`
- `02_scene.jpg`
- `03_perf.png`

### `prompt.md` template

```md
context:
General project context and technical positioning.

video description:
the script for the video voice over ...

slides:
slide 1: 01_cover.png, title1, description1
slide 2: 02_scene.jpg, title2, description2.
slide 3: background color, title3, description3.
```

## Run Modes

### CLI full pipeline

```powershell
.\venv\Scripts\python .\generate.py
```

### GUI partial modes

`web_panel.py` supports:

- Full Script
- Only LinkedIn Slider
- Only X
- Only TikTok Video
- Only TikTok Voice Over (Audio)

## Customization Guide

### 1) Brand colors and visual identity

Edit color constants in `generate.py`:

- `BRAND_BLACK`
- `BRAND_RED`
- `BRAND_CYAN`
- `BRAND_WHITE`

These drive LinkedIn slide visuals and subtitle styling choices.

### 2) Fonts

Fonts are loaded from `fonts/`.

- Title font: `Syncopate-Bold.ttf`
- Body font: `Syncopate-Regular.ttf`
- Subtitle font name: controlled by `.env` with `TTS_SUBTITLE_FONT_NAME`

If you change fonts:

1. Add new font files to `fonts/`
2. Update `_title_font` / `_body_font` in `generate.py`
3. Update `TTS_SUBTITLE_FONT_NAME` in `.env`

### 3) Voice and subtitle behavior

Tune in `.env`:

- `TTS_BACKEND` (`auto`, `elevenlabs`, `edge`, `kokoro`)
- `TTS_EDGE_VOICE`, `TTS_EDGE_RATE`, `TTS_EDGE_PITCH`
- `TTS_SUBTITLE_WORDS_PER_CHUNK`
- `TTS_SUBTITLE_MARGIN_V`, `TTS_SUBTITLE_MARGIN_L`
- `TTS_SUBTITLE_FONT_SIZE`, `TTS_SUBTITLE_ALIGNMENT`

### 4) LinkedIn slide layout

In `generate.py`, adjust:

- `SQUARE_SIZE`
- `TOP_IMAGE_AREA_HEIGHT`
- text wrapping and sizing in `_draw_linkedin_text_block`

### 5) Copy generation behavior

Prompt section logic:

- `context` -> X + LinkedIn content
- `video description` -> TikTok narration sequencing
- `slides` -> LinkedIn slide mapping and text intent

## Make It Portable For Any Machine

For each new machine:

1. Clone repo
2. Create `.env` from `.env.example`
3. Install dependencies (`setup.bat` on Windows, manual venv on macOS/Linux)
4. Ensure local FFmpeg binaries exist in `bin/`
5. Add your own assets into `assets/`
6. Run `generate.py` or `web_panel.py`

No global Python packages are required if you use the local `venv`.

## Git Hygiene For Public Repo

Root `.gitignore` already excludes:

- `.env` secrets
- `assets/` user content
- `output/` generated artifacts
- local model cache binaries
- temp/cache files

Only templates and source code stay tracked.

## Common Errors

- `Expected exactly 1 mp4 in assets/, found 0.`
  - Add one `.mp4` into `assets/`.

- `Missing required binaries in bin`
  - Add `ffmpeg.exe` and `ffprobe.exe` to `bin/`.

- `NVIDIA_NIM_API_KEY is missing in .env`
  - Set key in `.env`.

- Subtitles not visible or too high/low
  - Tune `TTS_SUBTITLE_MARGIN_V`, `TTS_SUBTITLE_MARGIN_L`, `TTS_SUBTITLE_FONT_SIZE`.
