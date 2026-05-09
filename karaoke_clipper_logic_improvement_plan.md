# karaoke-clipper Logic Improvement Plan

## Goal

Improve music and singing clip detection so exported clips do not end too early, especially in karaoke, concert, and livestream VODs.

The main architectural change is:

> Treat `inaSpeechSegmenter` as a rough proposal generator, not as the final boundary authority.

The final boundary should come from a safer combination of:

```text
inaSpeechSegmenter raw timeline
+ short-gap bridging
+ pre/post padding
+ optional energy boundary protection
+ Chromaprint confirmation
+ safe WhisperX refinement
+ manifest review scoring
```

---

## Current Pipeline

```text
Input video / YouTube URL / local MP4
  ↓
Extract working WAV
  ↓
Detect music-like regions with inaSpeechSegmenter
  ↓
Fallback to energy segmentation if unavailable
  ↓
Merge candidate segments
  ↓
Match candidates with local Chromaprint library
  ↓
Optionally refine boundaries with WhisperX
  ↓
Export clips with FFmpeg
  ↓
Write manifest.json / manifest.csv
```

---

## Target Pipeline

```text
Input video / YouTube URL / local MP4
  ↓
Extract working WAV
  ↓
Generate raw audio-label timeline
  ↓
Create rough music/singing proposals
  ↓
Bridge short non-music gaps
  ↓
Apply pre-roll and post-roll padding
  ↓
Run Chromaprint matching
  ↓
Apply safe boundary refinement
  ↓
Score candidate confidence
  ↓
Mark risky clips for review
  ↓
Export clips with FFmpeg
  ↓
Write richer manifest
```

---

# 1. Add Pre-Roll and Post-Roll Padding

## Problem

Clips sometimes crop too early because the detected music region ends before the real song tail.

This commonly happens when:

- the detector briefly labels the outro as `noise`;
- the song fades out gradually;
- singing ends before instrumental outro;
- WhisperX trims to vocal/transcript boundaries;
- inaSpeechSegmenter creates a short break inside a real song.

## New CLI Flags

Add:

```bash
--pre-roll <seconds>
--post-roll <seconds>
```

Recommended defaults:

```text
pre_roll = 0.5
post_roll = 2.0
```

## Config Fields

Add to runtime config:

```python
pre_roll_sec: float = 0.5
post_roll_sec: float = 2.0
```

## Implementation

Create helper:

```python
def apply_padding(segments, audio_duration, pre_roll_sec, post_roll_sec):
    padded = []

    for segment in segments:
        padded.append(
            Segment(
                start=max(0.0, segment.start - pre_roll_sec),
                end=min(audio_duration, segment.end + post_roll_sec),
                label=segment.label,
                score=segment.score,
            )
        )

    return padded
```

## Acceptance Criteria

- Every final exported clip has padding applied.
- Padding never produces negative start times.
- Padding never exceeds the source audio duration.
- `post_roll` reduces early-crop issues.
- Manifest records the applied `pre_roll_sec` and `post_roll_sec`.

---

# 2. Preserve Raw inaSpeechSegmenter Timeline

## Problem

If the pipeline immediately keeps only `music`, it loses useful context.

Example raw output:

```text
music   10.0–40.0
noise   40.0–40.5
music   40.5–95.0
```

This should usually become one music clip, not two.

## New Data Model

Add a raw segment type:

```python
from dataclasses import dataclass

@dataclass
class RawAudioSegment:
    label: str
    start: float
    end: float
```

Keep all labels returned by the detector, for example:

```text
music
speech
male
female
noise
```

## Implementation Rule

Do not immediately filter only `music`.

Use:

```python
raw_segments = get_ina_segments(audio_path)
music_candidates = build_music_candidates_from_raw_timeline(raw_segments)
```

Instead of:

```python
if label == "music":
    keep_segment()
```

## Acceptance Criteria

- Raw segmentation output can be inspected and debugged.
- Non-music gaps between music regions can be bridged intelligently.
- Manifest can show how many raw segments were merged.
- The pipeline can distinguish between short noise gaps and long speech interruptions.

---

# 3. Bridge Short Non-Music Gaps

## Problem

Music/singing may be briefly classified as noise or speech. This creates artificial cuts.

## New CLI Flags

```bash
--bridge-noise-gap <seconds>
--bridge-speech-gap <seconds>
```

Recommended defaults:

```text
bridge_noise_gap = 2.0
bridge_speech_gap = 1.0
```

## Logic

Merge two music regions if the gap between them is short.

Rules:

```text
music + short noise + music  => merge
music + short speech + music => merge cautiously
music + long speech + music  => do not merge
```

## Pseudocode

```python
def build_music_candidates_from_raw_timeline(
    raw_segments,
    bridge_noise_gap_sec=2.0,
    bridge_speech_gap_sec=1.0,
):
    candidates = []

    current_start = None
    current_end = None
    pending_gap_label = None

    for seg in raw_segments:
        if seg.label == "music":
            if current_start is None:
                current_start = seg.start
                current_end = seg.end
                pending_gap_label = None
                continue

            gap_duration = seg.start - current_end
            should_bridge = False

            if pending_gap_label == "noise" and gap_duration <= bridge_noise_gap_sec:
                should_bridge = True

            if pending_gap_label in {"speech", "male", "female"} and gap_duration <= bridge_speech_gap_sec:
                should_bridge = True

            if should_bridge:
                current_end = seg.end
            else:
                candidates.append(Segment(current_start, current_end, "music", 1.0))
                current_start = seg.start
                current_end = seg.end

            pending_gap_label = None

        else:
            if current_start is not None:
                pending_gap_label = seg.label

    if current_start is not None:
        candidates.append(Segment(current_start, current_end, "music", 1.0))

    return candidates
```

## Acceptance Criteria

- Short gaps inside songs no longer split clips.
- Long speech sections still separate clips.
- Gap-bridging behavior is configurable.
- Manifest records total bridged gap duration and merge count.

---

# 4. Use Song-Level Minimum Duration

## Problem

A value like `min_duration = 0.8s` is useful for generic sound-event detection, but it is too short for karaoke/song clipping.

For song clipping, very short music detections are usually false positives or incomplete fragments.

## New Default

For karaoke/song mode:

```text
min_segment = 20.0 or 30.0 seconds
```

## Suggested Profiles

```text
karaoke:
  min_segment = 30
  max_segment = 420
  merge_gap = 1.5
  pre_roll = 0.5
  post_roll = 2.0

concert:
  min_segment = 45
  max_segment = 600
  merge_gap = 2.5
  pre_roll = 1.0
  post_roll = 3.0

mixed_stream:
  min_segment = 20
  max_segment = 420
  merge_gap = 1.0
  pre_roll = 0.5
  post_roll = 2.0
```

## Acceptance Criteria

- Tiny false-positive clips are filtered out.
- Normal karaoke songs remain.
- Users can override the default.
- Profile defaults are documented.

---

# 5. Make WhisperX Refinement Safe

## Problem

WhisperX is speech-oriented. It may ignore:

- instrumental intro;
- instrumental outro;
- applause;
- humming;
- singing with low transcription confidence;
- music-only sections.

Therefore WhisperX should not be allowed to aggressively shrink music clips.

## New CLI Flag

```bash
--whisperx-boundary-mode off|metadata|safe
```

Default:

```text
safe
```

## Modes

### `off`

Do not use WhisperX for boundary refinement.

### `metadata`

Run WhisperX only to generate transcript/review metadata. Do not modify clip boundaries.

### `safe`

Allow WhisperX to adjust boundaries only within strict limits.

## Safe Refinement Logic

```python
def safe_refine_music_boundary(
    original_start,
    original_end,
    whisper_start,
    whisper_end,
    max_start_shrink_sec=0.5,
    max_end_shrink_sec=0.5,
    post_roll_sec=2.0,
    audio_duration=None,
):
    # Protect intro
    if whisper_start > original_start + max_start_shrink_sec:
        final_start = original_start
    else:
        final_start = min(original_start, whisper_start)

    # Protect outro
    if whisper_end < original_end - max_end_shrink_sec:
        final_end = original_end
    else:
        final_end = max(original_end, whisper_end)

    final_end += post_roll_sec

    if audio_duration is not None:
        final_end = min(final_end, audio_duration)

    return final_start, final_end
```

## Acceptance Criteria

- WhisperX cannot remove more than `max_start_shrink_sec` from the intro.
- WhisperX cannot remove more than `max_end_shrink_sec` from the outro.
- Music clips do not become vocal-only clips.
- Boundary method is recorded in the manifest.

---

# 6. Avoid Hard Splitting Long Songs

## Problem

Current max-segment logic may cut a long song at an arbitrary timestamp.

Bad example:

```text
song: 0–360
max_segment: 240

output:
0–240
240–360
```

This can cut in the middle of a song.

## Better Behavior

If a segment exceeds `max_segment`, use one of these behaviors:

```text
Option A: keep full segment and mark needs_review
Option B: split only at low-energy valley
Option C: split at internal long non-music gap
```

## Recommended MVP

Use Option A first:

```python
if duration > max_segment_sec:
    segment.needs_review = True
    segment.review_reason = "segment_exceeds_max_duration"
```

Do not hard split unless the user explicitly enables:

```bash
--allow-hard-split true
```

Default:

```text
allow_hard_split = false
```

## Acceptance Criteria

- Long songs are not cut arbitrarily by default.
- Long clips are marked for review.
- Hard splitting is opt-in.
- Manifest records `review_reason = "segment_exceeds_max_duration"`.

---

# 7. Improve Energy Fallback Resolution

## Problem

A 1-second energy frame is too coarse for accurate boundaries.

If fallback uses:

```python
window_size = sample_rate
```

then each decision window is about 1 second.

## New Parameters

```bash
--energy-frame-ms 100
--energy-min-active-ms 500
--energy-min-silence-ms 1200
```

## Recommended Defaults

```text
energy_frame_ms = 100
energy_min_active_ms = 500
energy_min_silence_ms = 1200
```

## Implementation

```python
frame_size = int(sample_rate * energy_frame_ms / 1000)
```

Use frame-level RMS or dB energy.

Suggested logic:

```text
start when energy is above threshold for energy_min_active_ms
stop only after energy stays below threshold for energy_min_silence_ms
```

## Acceptance Criteria

- Energy fallback has better boundary precision.
- Short energy dips do not cause early crop.
- Fallback behavior is configurable.
- Unit tests cover short silence gaps.

---

# 8. Make Chromaprint Confirmation More Conservative

## Problem

Text-style similarity on Chromaprint strings can produce misleading scores.

False positives are worse than unknown matches.

## Rule

If proper Chromaprint comparison is unavailable:

```python
confidence = 0.0
backend = "chromaprint_unavailable"
needs_review = True
review_reason = "fingerprint_compare_unavailable"
```

Avoid using `SequenceMatcher` as a real confidence score for Chromaprint fingerprints.

## Acceptance Criteria

- Bad fingerprint fallback does not produce fake high-confidence matches.
- Unknown songs are marked for review instead of incorrectly labeled.
- Manifest explains why matching failed.

---

# 9. Add Candidate Scoring

## Problem

A clip should not be accepted only because one detector found it.

Use a combined score.

## Suggested Score

```python
score = (
    0.40 * music_ratio
    + 0.35 * fingerprint_confidence
    + 0.15 * duration_score
    + 0.10 * boundary_quality_score
)
```

## Fields

```python
music_ratio: float
fingerprint_confidence: float
duration_score: float
boundary_quality_score: float
final_score: float
```

## Review Rule

```python
needs_review = final_score < 0.65
```

## Suggested Subscores

### `music_ratio`

Ratio of raw timeline duration inside the final clip that is labeled `music`.

```python
music_ratio = music_labeled_duration / clip_duration
```

### `duration_score`

Reward durations that look like song clips.

Example:

```python
if 60 <= duration <= 360:
    duration_score = 1.0
elif 30 <= duration < 60 or 360 < duration <= 600:
    duration_score = 0.7
else:
    duration_score = 0.4
```

### `boundary_quality_score`

Penalize clips that were heavily modified, hard split, or have too much non-music content.

## Acceptance Criteria

- Low-confidence clips are still exported if desired, but marked for review.
- Manifest shows why a clip is trustworthy or risky.
- Users can sort clips by confidence.

---

# 10. Add Richer Manifest Fields

## Problem

Debugging early-crop issues requires knowing how the boundary was produced.

## Add Fields

```json
{
  "source_video": "...",
  "video_id": "...",
  "song": "...",
  "artist": "...",

  "raw_start_sec": 102.3,
  "raw_end_sec": 210.7,

  "start_sec": 101.8,
  "end_sec": 212.9,
  "duration_sec": 111.1,

  "start_tc": "00:01:41.800",
  "end_tc": "00:03:32.900",

  "pre_roll_sec": 0.5,
  "post_roll_sec": 2.0,

  "boundary_method": "ina_bridge_padding",
  "refinement_method": "whisperx_safe",

  "music_ratio": 0.92,
  "fingerprint_confidence": 0.87,
  "duration_score": 0.95,
  "boundary_quality_score": 0.80,
  "final_score": 0.89,

  "merge_count": 3,
  "bridged_gap_total_sec": 1.4,

  "clip_path": "...",
  "audio_path": "...",

  "backend": "chromaprint",
  "needs_review": false,
  "review_reason": null
}
```

## Acceptance Criteria

- Manifest clearly shows raw and final boundaries.
- Manifest shows whether padding/refinement was applied.
- Review tools can filter by `needs_review`.
- The Streamlit UI can display `final_score`, `boundary_method`, and `review_reason`.

---

# 11. Add Preset Profiles

## Problem

Users should not manually tune every parameter.

## New CLI Flag

```bash
--profile karaoke|concert|mixed_stream|strict
```

## Profiles

```python
PROFILES = {
    "karaoke": {
        "min_segment": 30,
        "max_segment": 420,
        "merge_gap": 1.5,
        "bridge_noise_gap": 2.0,
        "bridge_speech_gap": 1.0,
        "pre_roll": 0.5,
        "post_roll": 2.0,
        "whisperx_boundary_mode": "safe",
    },
    "concert": {
        "min_segment": 45,
        "max_segment": 600,
        "merge_gap": 2.5,
        "bridge_noise_gap": 3.0,
        "bridge_speech_gap": 1.5,
        "pre_roll": 1.0,
        "post_roll": 3.0,
        "whisperx_boundary_mode": "safe",
    },
    "mixed_stream": {
        "min_segment": 20,
        "max_segment": 420,
        "merge_gap": 1.0,
        "bridge_noise_gap": 1.5,
        "bridge_speech_gap": 0.8,
        "pre_roll": 0.5,
        "post_roll": 2.0,
        "whisperx_boundary_mode": "metadata",
    },
    "strict": {
        "min_segment": 45,
        "max_segment": 360,
        "merge_gap": 0.8,
        "bridge_noise_gap": 0.8,
        "bridge_speech_gap": 0.3,
        "pre_roll": 0.3,
        "post_roll": 1.0,
        "whisperx_boundary_mode": "off",
    },
}
```

## Acceptance Criteria

- `--profile karaoke` works out of the box.
- Explicit CLI flags override profile values.
- README documents profile behavior.
- Streamlit UI exposes profile selection.

---

# 12. Suggested File Changes

## `app/config.py`

Add config fields:

```python
pre_roll_sec
post_roll_sec
bridge_noise_gap_sec
bridge_speech_gap_sec
whisperx_boundary_mode
allow_hard_split
energy_frame_ms
energy_min_active_ms
energy_min_silence_ms
profile
review_score_threshold
```

---

## `app/main.py`

Update orchestration order:

```text
detect raw timeline
build candidates
bridge gaps
merge/coalesce
apply padding
match
safe refine
score
export
write manifest
```

---

## `app/segment/music_segments.py`

Add:

```python
RawAudioSegment
get_raw_ina_timeline()
build_music_candidates_from_raw_timeline()
bridge_short_gaps()
apply_padding()
calculate_music_ratio()
```

Modify:

```python
merge_segments()
coalesce_to_expected_count()
energy_fallback()
```

---

## `app/align/whisperx_align.py`

Add safe modes:

```text
off
metadata
safe
```

Add:

```python
safe_refine_music_boundary()
```

Prevent WhisperX from aggressively shrinking music clips.

---

## `app/identify/chromaprint_match.py`

Remove or weaken `SequenceMatcher` fallback.

If proper fingerprint comparison fails:

```python
confidence = 0.0
needs_review = True
review_reason = "fingerprint_compare_unavailable"
```

---

## `app/output/manifest.py`

Add richer fields:

```python
raw_start_sec
raw_end_sec
duration_sec
boundary_method
refinement_method
music_ratio
fingerprint_confidence
final_score
merge_count
bridged_gap_total_sec
needs_review
review_reason
```

---

## `app/ui/streamlit_app.py`

Expose new controls:

```text
profile
pre_roll_sec
post_roll_sec
bridge_noise_gap_sec
bridge_speech_gap_sec
whisperx_boundary_mode
review_score_threshold
```

Display new manifest fields:

```text
final_score
needs_review
review_reason
boundary_method
refinement_method
music_ratio
```

---

# 13. Tests To Add

## Padding Tests

```text
test_apply_padding_does_not_go_below_zero
test_apply_padding_does_not_exceed_audio_duration
test_apply_padding_adds_expected_tail
```

## Gap Bridging Tests

```text
test_bridge_short_noise_gap_between_music
test_bridge_short_speech_gap_between_music
test_do_not_bridge_long_speech_gap
```

## WhisperX Safe Refinement Tests

```text
test_whisperx_cannot_shrink_intro_too_much
test_whisperx_cannot_shrink_outro_too_much
test_whisperx_can_expand_boundary
test_whisperx_metadata_mode_does_not_change_boundaries
test_whisperx_off_mode_does_not_run_refinement
```

## Long Segment Handling Tests

```text
test_long_segment_marked_for_review
test_long_segment_not_hard_split_by_default
test_hard_split_only_when_enabled
```

## Manifest Tests

```text
test_manifest_contains_raw_and_final_boundaries
test_manifest_contains_review_reason
test_manifest_contains_boundary_method
test_manifest_contains_final_score
```

## Energy Fallback Tests

```text
test_energy_fallback_uses_configurable_frame_size
test_energy_fallback_bridges_short_silence
test_energy_fallback_stops_after_min_silence
```

---

# 14. Recommended Default Values

```text
profile = karaoke

min_segment = 30
max_segment = 420
merge_gap = 1.5

bridge_noise_gap = 2.0
bridge_speech_gap = 1.0

pre_roll = 0.5
post_roll = 2.0

whisperx_boundary_mode = safe

allow_hard_split = false

energy_frame_ms = 100
energy_min_active_ms = 500
energy_min_silence_ms = 1200

review_score_threshold = 0.65
```

---

# 15. Example Behavior

## Before

```text
music 10.0–40.0
noise 40.0–40.5
music 40.5–95.0

Output:
10.0–40.0
40.5–95.0
```

## After

```text
music 10.0–40.0
noise 40.0–40.5
music 40.5–95.0

Bridge short noise gap
Apply pre/post roll

Output:
9.5–97.0
```

---

# 16. Priority Implementation Order

1. Add `pre_roll` and `post_roll`.
2. Make WhisperX boundary refinement safe.
3. Preserve raw inaSpeechSegmenter timeline.
4. Bridge short noise/speech gaps.
5. Add richer manifest fields.
6. Add review flags.
7. Improve energy fallback frame size.
8. Add preset profiles.
9. Remove unsafe fingerprint fallback.
10. Improve long-segment handling.
11. Add Streamlit controls for the new options.
12. Add tests for padding, bridging, refinement, and manifest output.

---

# 17. Definition of Done

The improvement is complete when:

- Music/singing clips no longer crop early in common cases.
- Short noise gaps inside songs are bridged.
- WhisperX cannot remove instrumental intro/outro aggressively.
- Final clips include configurable padding.
- Manifest explains how each boundary was produced.
- Risky clips are marked with `needs_review`.
- Unit tests cover padding, bridging, safe refinement, long segment handling, and manifest output.
- README documents the new profiles and tuning parameters.

---

# 18. Notes for Coding Agent

Do not rewrite the whole project.

Implement this as an incremental refactor:

1. Keep the existing pipeline working.
2. Add new config fields with safe defaults.
3. Add padding first.
4. Add safe WhisperX behavior second.
5. Add raw timeline preservation and bridging third.
6. Add manifest/debug fields after behavior is stable.
7. Add tests after each logic change.

The highest-priority bug to fix is early crop. Therefore, prioritize:

```text
post_roll padding
safe WhisperX refinement
short-gap bridging
```

before lower-priority features like scoring and profiles.
