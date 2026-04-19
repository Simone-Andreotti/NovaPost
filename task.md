# Task Log - Social Media Content Automation Pipeline

## Objective
Implement all tasks described in `implementation.md` by creating a local Python project under `social-engine/` with modular automation for X, TikTok, and LinkedIn outputs.

## Considerations
- Keep tooling local to the project (no global installs).
- Bind FFmpeg from local `social-engine/bin/` at runtime.
- Preserve strict JSON schema handling for AI output.
- Handle missing assets and missing fonts with explicit errors/fallbacks.
- Keep output deterministic where possible (sorted asset discovery, ordered slides).
- Keep LinkedIn PDF size controlled by downscaling and JPEG compression before merge.
- Use relative FFmpeg subtitle filter paths to avoid Windows drive-letter escaping issues.
- Keep TikTok subtitles in lower-third, compact, and scene-timed to avoid text covering visuals.
- Keep LinkedIn slides square and readable with concise text overlays.
- Make X output include explicit image selection guidance and a more professional tone.
- Keep plain LinkedIn summary slides on dark solid background (no gradients).
- Never trim final TikTok output duration below input video duration.
- Use `prompt.md` sections (`context`, `video description`, `slides`) as authoritative content source.
- Keep LinkedIn image slides unzoomed and placed at top of square canvas.

## Decisions
- Project root for implementation: `social-engine/`.
- Keep `task.md` at workspace root to separate planning/progress from implementation files.
- Use NVIDIA OpenAI-compatible endpoint via `openai` SDK.
- Initial iteration replaced `edge-tts` with local `kokoro-onnx` male voice synthesis (`am_michael`).
- Reintroduce `edge-tts` as primary neural TTS backend with `kokoro-onnx` fallback for reliability.
- Create `setup.bat` to standardize local environment bootstrapping on Windows.
- Add local `ffprobe.exe` into `social-engine/bin/` because `ffmpeg.probe(...)` requires it.
- Use image analysis (vision + fallback heuristics) before generating copy so text is mapped to visuals.
- Auto-download Kokoro model assets into `social-engine/models/kokoro/` when missing.
- Cap LinkedIn output to `image_count + 1` slides (all image slides + one plain summary slide).
- Switch LinkedIn typography to Syncopate family only: bold title + regular body.
- Build TikTok subtitles from prompt-derived beat sequence rather than pure transcription.
- Build TikTok subtitles from Edge TTS word-boundary timestamps to keep speech/subtitle sync.
- Add mandatory script reread pass for TikTok and text reread pass for LinkedIn slides before rendering.
- Add optional ElevenLabs premium backend for higher-quality warm professional voice when key is configured.

## Progress
- [x] Reviewed `implementation.md` requirements.
- [x] Inspected workspace and available font assets.
- [x] Scaffolded `social-engine/` with `bin/`, `assets/`, and `output/`.
- [x] Copied `ffmpeg.exe` into local `social-engine/bin/`.
- [x] Added `requirements.txt`, `.env`, `setup.bat`, and starter `assets/prompt.md`.
- [x] Implemented full modular `generate.py` pipeline with all required steps.
- [x] Installed dependencies inside local `venv` and validated syntax (`python -m py_compile generate.py`).
- [x] Executed end-to-end run with real media assets.
- [x] Validated final outputs (`x_post.txt`, `tiktok_final.mp4`, `linkedin_carousel.pdf`).
- [x] Applied quality iteration after user feedback (TikTok subtitle layout/voice, LinkedIn square redesign, X professionalism + media plan).
- [x] Applied second quality iteration (Kokoro TTS migration, technical LinkedIn copy hardening, subtitle chunking, and fixed video-length preservation).
- [x] Applied third quality iteration (prompt-section parsing, prompt-driven voiceover/slide text, unzoomed top-image LinkedIn layout, Syncopate-only typography).
- [x] Applied fourth quality iteration (TikTok script QC reread, Edge TTS primary + Kokoro fallback, word-boundary subtitle timing, LinkedIn final-slide text QC + overflow-safe rendering).
- [x] Applied fifth stabilization pass (video-script label alias parsing, sentence-boundary subtitle fallback, adaptive voiceover duration fit, plain-slide rewrite lock).
- [x] Applied sixth polish pass (normal-speed narration, no-dot speech normalization, lower-left subtitle placement, video-timeline script study, premium voice option).
- [x] Applied seventh polish pass (male voice defaults, subtitle restore + placement fix, strict context/video source routing, background-slide title layout fix).
- [x] Applied eighth stabilization pass (markdown prompt-section parser compatibility, split generation prompts, video-script-only TikTok routing verified).
- [x] Applied ninth stabilization pass (strict video-script-only voiceover generation, crisp male voice default update, subtitle visibility restore).

## Execution Notes
- `venv` dependencies installed successfully from `requirements.txt`.
- Runtime check reached Step 1 and failed as designed because `assets/` currently has no `.mp4`.
- Error observed: `Expected exactly 1 mp4 in assets/, found 0.`
- Added `social-engine/README.md` with full folder layout and exact media/text input placement instructions.
- During real run, first failure was `[WinError 2]` due missing `ffprobe.exe`; fixed by copying `ffprobe.exe` to `social-engine/bin/`.
- Second failure was FFmpeg subtitles path parsing (`Unable to open ... subtitles.srt`) on Windows absolute paths; fixed by using relative filter paths.
- Final rerun completed all 5 steps successfully and generated all expected artifacts.
- After quality feedback, `generate.py` was refactored to:
	- Generate image-aware plans for X/LinkedIn with explicit filename mapping.
	- Write compact lower-third TikTok subtitles with scene-aware timing and segment splitting.
	- Render LinkedIn slides as 1080x1080 square compositions with reduced text density.
	- Enforce more professional X copy style and include recommended image list in `x_post.txt`.
- Second refinement added:
	- Kokoro local TTS backend (`kokoro-onnx` + `soundfile`) with male voice default.
	- TikTok output duration now matches source video duration (no trimming to voiceover length).
	- Subtitles now split into shorter phrases to reduce on-screen text density.
	- LinkedIn plain slide switched to solid dark background without gradients.
	- LinkedIn slide bodies forced toward technical implementation framing.
- Third refinement added:
	- `prompt.md` parser introduced for `context`, `video description`, and explicit slide lines.
	- TikTok narration and subtitle beats now follow prompt-provided video flow.
	- LinkedIn slides now consume prompt-provided slide intent first (not image-caption style output).
	- Image slides use non-zoomed top placement with dark lower text panel.
	- Fonts set to Syncopate Bold (title) and Syncopate Regular (body).
- Fourth refinement added:
	- TikTok script now goes through a mandatory QC reread pass before TTS.
	- TTS now uses `edge-tts` (`en-US-AndrewMultilingualNeural`) first, then falls back to Kokoro.
	- Subtitles are generated from Edge TTS word boundaries for direct speech sync.
	- LinkedIn slide text now gets a final reread pass and plain-slide intent is no longer replaced by generic copy.
	- LinkedIn text rendering now auto-scales and trims to avoid final-slide overflow artifacts.
- Fifth refinement added:
	- Prompt parser now accepts `video script description:` as alias for `video description:`.
	- Edge subtitle extractor now supports `SentenceBoundary` events when `WordBoundary` is unavailable.
	- Edge voiceover duration is adaptively speed-corrected only when narration is too long.
	- Plain LinkedIn summary slide is protected from reread-pass CTA drift and forced to technical summary tone.
- Sixth refinement added:
	- TTS speech script now strips punctuation pauses, converts `three.js` -> `threejs`, and reads domains as `dot` words.
	- TikTok script QC pass now includes detected video timeline notes (duration + scene-cut markers).
	- Subtitle layout moved to lower-left around 2/3 screen height, with configurable chunk sizing and margins.
	- Audio chain tuned with compression + loudness normalization for clearer, steadier narration.
	- Added optional ElevenLabs backend (`eleven_v3` with fallback to `eleven_multilingual_v2`) ahead of Edge/Kokoro in `auto` mode.
- Seventh refinement added:
	- Edge default voice switched to male (`en-US-AndrewMultilingualNeural`), with male ElevenLabs default voice ID.
	- Subtitle generation now hard-fails over to script-derived timing when boundary extraction is empty (restores missing subtitles).
	- Subtitle defaults set to lower-left with larger text and 2/3 vertical placement target.
	- Generation flow split so `context` is used for X + LinkedIn only, while TikTok narration uses `video script` + video-study notes.
	- Plain/background slide title forced to a stable single-line layout to avoid broken accent-line appearance.
- Eighth refinement added:
	- Prompt parser now supports markdown-style section markers like `# context:`, `# video script:`, and `# slides:`.
	- Generation requests split into posts/slides and TikTok-only phases to enforce source separation.
	- TikTok narration quality rerun validated with male voice at normal speed and active subtitle output.
- Ninth refinement added:
	- TikTok voiceover no longer uses any extra data generation path; it now follows the `video script` text directly with minimal cleanup only.
	- Removed duration-forced script extension loop to avoid altering author-provided narrative.
	- Default male voice switched to `en-US-ChristopherNeural` for crisper delivery.
	- Subtitle styling switched to boxed lower-left rendering with larger text to ensure visibility.

## Open Inputs Needed
- None.
