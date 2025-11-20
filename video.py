"""
video1.py  (Smart Lyrical Video Generator – cleaned version, single-line lyrics)

- Two tabs: Audio & Mood, Video Settings & Output
- Load audio (YouTube/file), trim, Whisper lyrics, simple mood detection
- Video render: Black / Mood Video / Mood Image / Custom Background
- Bigger lyrics, text color from ColorPicker, font preview, custom bg drag&drop
- Lyrics on video: one line at a time (segments split into chunks)

Run: python video1.py
"""

import os
import uuid
import shutil
import random
import traceback
import subprocess
import math
from pathlib import Path
from typing import List, Dict, Any

import gradio as gr
import yt_dlp
import whisper
import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
from moviepy.editor import (
    AudioFileClip, ImageClip, TextClip, CompositeVideoClip, VideoFileClip,
    concatenate_videoclips, ColorClip
)
import moviepy.video.fx.all as vfx

# ----------------- Configuration & Folders -----------------
BASE = Path.cwd()
DOWNLOADS = BASE / "downloads"
PROCESSED = BASE / "processed"
LYRICS = BASE / "lyrics"
ASSETS = BASE / "assets"
ASSET_IMAGES = ASSETS / "images"
ASSET_VIDEOS = ASSETS / "videos"
ASSET_CUSTOM = ASSETS / "custom"
FONT_DIR = ASSETS / "fonts"

for d in (DOWNLOADS, PROCESSED, LYRICS, ASSETS, ASSET_IMAGES, ASSET_VIDEOS, ASSET_CUSTOM, FONT_DIR):
    d.mkdir(exist_ok=True, parents=True)

MOODS = ["energetic", "calm", "happy", "sad"]
for mood in MOODS:
    (ASSET_IMAGES / mood).mkdir(exist_ok=True, parents=True)
    (ASSET_VIDEOS / mood).mkdir(exist_ok=True, parents=True)

WHISPER_MODEL = "small"
FPS = 24

DEFAULT_TEXT_COLOR = "#FFFFFF"
DEFAULT_ANIM = "Fade In"
DEFAULT_PLATFORM = "Reels/Shorts (9:16)"

# ----------------- Font utilities -----------------
def available_fonts_in_assets() -> List[str]:
    files = []
    if FONT_DIR.exists():
        for p in sorted(FONT_DIR.iterdir()):
            if p.suffix.lower() in (".ttf", ".otf"):
                files.append(p.name)
    return files

def find_font_path(font_choice: str) -> str:
    p = Path(font_choice)
    if p.exists() and p.suffix.lower() in (".ttf", ".otf"):
        return str(p.resolve())
    candidate = FONT_DIR / font_choice
    if candidate.exists():
        return str(candidate.resolve())
    for ext in (".ttf", ".otf"):
        cand2 = FONT_DIR / (font_choice + ext)
        if cand2.exists():
            return str(cand2.resolve())
    return font_choice

FONT_CHOICES = available_fonts_in_assets()
if not FONT_CHOICES:
    FONT_CHOICES = ["Arial", "Times New Roman", "Georgia"]

# ----------------- Basic helpers -----------------
def is_youtube_url(url: str) -> bool:
    return bool(url and ("youtube.com" in url or "youtu.be" in url))

def download_youtube_audio(url: str, out_dir: Path = DOWNLOADS) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "%(title)s.%(ext)s")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": template,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "quiet": True,
        "no_warnings": True,
        "prefer_ffmpeg": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)
    files = list(out_dir.glob("*.mp3"))
    if not files:
        raise RuntimeError("Failed to download audio")
    files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files_sorted[0].resolve())

def ensure_preview_audio(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg"):
        return str(p.resolve())
    out = DOWNLOADS / f"preview_{uuid.uuid4().hex}.mp3"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(p), "-y", str(out)], check=True)
    return str(out.resolve())

def convert_to_wav_for_whisper(input_file: str, output_file: str) -> str:
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(input_file),
        "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "-y", str(output_file)
    ], check=True)
    return str(Path(output_file).resolve())

def trim_audio_segment_file(input_file: str, start_sec: float, duration_sec: float, output_file: str) -> str:
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start_sec), "-i", str(input_file),
        "-t", str(duration_sec), "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "-y", str(output_file)
    ], check=True)
    return str(Path(output_file).resolve())

# ----------------- Whisper -----------------
def load_whisper_model(name=WHISPER_MODEL):
    global _WHISPER_MODEL_CACHE
    try:
        _WHISPER_MODEL_CACHE
    except NameError:
        _WHISPER_MODEL_CACHE = whisper.load_model(name)
    return _WHISPER_MODEL_CACHE

def transcribe_with_whisper_file(audio_path: str) -> Dict[str, Any]:
    model = load_whisper_model()
    result = model.transcribe(audio_path, fp16=False)
    segments = []
    for seg in result.get("segments", []):
        segments.append({"start": round(seg["start"], 2), "end": round(seg["end"], 2), "text": seg["text"].strip()})
    return {"text": result.get("text", ""), "segments": segments}

def segments_to_html(segments: List[Dict]) -> str:
    lines = []
    for seg in segments:
        start = round(seg.get("start", 0.0), 2)
        txt = seg.get("text", "").replace("\n", " ").strip()
        if txt:
            lines.append(f'<p data-start="{start}" style="margin:6px 0;">{txt}</p>')
    return "\n".join(lines)

def create_lyric_html_with_sync(segments: List[Dict]) -> str:
    lines_html = segments_to_html(segments)
    js = """
<script>
(function(){
  function sync() {
    const audio = document.getElementById('preview_audio');
    const cont = document.getElementById('preview_lyrics_container');
    if(!audio || !cont){ requestAnimationFrame(sync); return; }
    const lines = cont.querySelectorAll('p');
    function update(){
      const t = audio.currentTime;
      lines.forEach((line,i)=>{
        const start = parseFloat(line.dataset.start);
        const next = i<lines.length-1 ? parseFloat(lines[i+1].dataset.start) : 1e9;
        if(t>=start && t<next){
          line.style.color = '#ffcc00';
          line.style.fontWeight = '700';
          line.style.fontSize = '18px';
        } else {
          line.style.color = '#ddd';
          line.style.fontWeight = '400';
          line.style.fontSize = '16px';
        }
      });
      requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
  }
  requestAnimationFrame(sync);
})();
</script>
"""
    container = f'<div id="preview_lyrics_container" style="max-height:320px; overflow-y:auto; font-family:Poppins, sans-serif; color:#ddd; padding:8px;">{lines_html}</div>'
    return container + js

# ----------------- Mood heuristic -----------------
def detect_mood_from_audio_file(audio_path: str) -> str:
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        sc = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        if tempo >= 120 and rms > 0.05 and zcr > 0.1:
            return "Energetic ⚡"
        if tempo < 80 and sc < 2000 and rms < 0.03:
            return "Calm 🌙"
        if sc > 3000 and rms > 0.04:
            return "Happy 😄"
        if rms < 0.02 and zcr < 0.05:
            return "Sad 😢"
        return "Calm 🌙"
    except Exception as e:
        print("detect_mood error:", e)
        return "Calm 🌙"

def mood_label_to_key(mood_label: str) -> str:
    if not mood_label:
        return "calm"
    label = mood_label.lower()
    if "energetic" in label or "⚡" in label:
        return "energetic"
    if "calm" in label or "🌙" in label:
        return "calm"
    if "happy" in label or "😄" in label:
        return "happy"
    if "sad" in label or "😢" in label:
        return "sad"
    return "calm"

def list_media_files(folder: Path, exts=(".mp4", ".mov", ".webm", ".mkv", ".avi")):
    if not folder.exists():
        return []
    return [str(p.resolve()) for p in folder.iterdir() if p.suffix.lower() in exts]

def list_image_files(folder: Path):
    if not folder.exists():
        return []
    return [str(p.resolve()) for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]

def get_mood_video_path(mood_key: str) -> str:
    files = list_media_files(ASSET_VIDEOS / mood_key)
    return random.choice(files) if files else None

def get_mood_image_path(mood_key: str) -> str:
    files = list_image_files(ASSET_IMAGES / mood_key)
    return random.choice(files) if files else None

# ----------------- Safe TextClip -----------------
def safe_textclip(txt: str, fontsize: int, fontname_or_path: str, color: str, max_width: int):
    font_resolved = find_font_path(fontname_or_path)
    try:
        tc = TextClip(
            txt,
            fontsize=fontsize,
            font=font_resolved,
            color=color,
            stroke_color="black",
            stroke_width=3,
            method="caption",
            size=(max_width, None),
            align="center"
        )
        return tc
    except Exception:
        try:
            pil_font = None
            if isinstance(font_resolved, str) and Path(font_resolved).exists():
                try:
                    pil_font = ImageFont.truetype(font_resolved, fontsize)
                except Exception:
                    pil_font = None
            if pil_font is None:
                for candidate in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
                    try:
                        pil_font = ImageFont.truetype(candidate, fontsize)
                        break
                    except Exception:
                        pil_font = None
            if pil_font is None:
                pil_font = ImageFont.load_default()

            img_w = max(400, max_width)
            words = txt.split(" ")
            lines, cur = [], ""
            dummy = Image.new("RGBA", (img_w, 1000), (0,0,0,0))
            draw = ImageDraw.Draw(dummy)
            for w in words:
                test = (cur + " " + w).strip()
                bbox = draw.textbbox((0,0), test, font=pil_font)
                tw = bbox[2] - bbox[0]
                if tw <= img_w - 40:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)

            img_h = (len(lines) + 1) * (fontsize + 12)
            img = Image.new("RGBA", (img_w, img_h), (0,0,0,0))
            draw = ImageDraw.Draw(img)
            try:
                fill = tuple(int(color.lstrip("#")[i:i+2],16) for i in (0,2,4))
            except Exception:
                fill = (255,255,255)
            y = 0
            stroke = (0,0,0)
            for line in lines:
                offsets = [(1,1), (-1,1), (1,-1), (-1,-1)]
                for dx, dy in offsets:
                    draw.text((10+dx, y+dy), line, font=pil_font, fill=stroke)
                draw.text((10, y), line, font=pil_font, fill=fill)
                y += fontsize + 12
            tmp = PROCESSED / f"txt_{uuid.uuid4().hex}.png"
            img.save(tmp)
            ic = ImageClip(str(tmp)).set_duration(1)
            ic.tmp_path = tmp
            return ic
        except Exception as e:
            print("safe_textclip fallback failed:", e)
            raise

# ----------------- Presets & font size -----------------
PRESET_DEFS = {
    "Bold Cinematic": {
        "scale": 1.0,
        "weight": 800,
        "stroke_ratio": 0.12,
        "line_spacing": 1.05,
        "shadow": True
    },
    "Minimal Clean": {
        "scale": 0.85,
        "weight": 700,
        "stroke_ratio": 0.05,
        "line_spacing": 1.1,
        "shadow": False
    },
    "Big Chorus": {
        "scale": 1.25,
        "weight": 900,
        "stroke_ratio": 0.14,
        "line_spacing": 1.02,
        "shadow": True
    },
    "Vibe Mode": {
        "scale": 0.95,
        "weight": 700,
        "stroke_ratio": 0.06,
        "line_spacing": 1.12,
        "shadow": True
    },
    "Story Mode": {
        "scale": 0.78,
        "weight": 600,
        "stroke_ratio": 0.04,
        "line_spacing": 1.25,
        "shadow": False
    },
    "Title Mode": {
        "scale": 1.4,
        "weight": 900,
        "stroke_ratio": 0.15,
        "line_spacing": 1.0,
        "shadow": True
    }
}

def _calc_fontsize_by_width(resolution: tuple, text_line: str, max_width_percent: float, preset_name: str):
    w, h = resolution
    base = int(h * 0.06)
    preset = PRESET_DEFS.get(preset_name, PRESET_DEFS["Bold Cinematic"])
    scale = preset["scale"]
    target_w = int(w * (max_width_percent / 100.0))
    avg_char = max(6, len(text_line))

    estimated = int(target_w / (0.42 * avg_char))
    estimated = max(base, min(estimated, int(h * 0.33)))

    final = int(estimated * scale)
    final = max(28, min(final, 260))
    return final

# ----------------- Background helper -----------------
def create_mood_clip(mood_key: str, resolution: tuple, audio_duration: float, dim_factor: float = 0.85):
    try:
        w, h = resolution
        files = list_media_files(ASSET_VIDEOS / mood_key)
        if not files:
            return ColorClip(size=(w, h), color=(0,0,0)).set_duration(audio_duration).set_fps(FPS)
        chosen = random.choice(files)
        bg = VideoFileClip(chosen)
        bg = bg.resize(height=h)
        if bg.w < w:
            bg = bg.resize(width=w)
        if bg.w != w or bg.h != h:
            try:
                x1 = max(0, int((bg.w - w) / 2))
                y1 = max(0, int((bg.h - h) / 2))
                bg = bg.crop(x1=x1, y1=y1, x2=x1 + w, y2=y1 + h)
            except Exception:
                bg = bg.resize(newsize=(w, h))
        if bg.duration < audio_duration:
            loops = int(audio_duration // bg.duration) + 1
            bg = concatenate_videoclips([bg] * loops)
        bg = bg.subclip(0, audio_duration)
        try:
            bg = vfx.colorx(bg, dim_factor)
        except Exception:
            overlay = ColorClip(size=(w,h), color=(0,0,0)).set_duration(audio_duration).set_opacity(0.15)
            bg = CompositeVideoClip([bg, overlay]).set_duration(audio_duration)
        return bg.set_fps(FPS)
    except Exception as e:
        print("create_mood_clip error:", e)
        traceback.print_exc()
        w, h = resolution
        return ColorClip(size=(w, h), color=(0,0,0)).set_duration(audio_duration).set_fps(FPS)

# ----------------- SINGLE-LINE helper -----------------
MAX_CHARS_PER_LINE = 35

def split_into_chunks(text: str, max_chars: int = MAX_CHARS_PER_LINE):
    words = text.split()
    chunks, current, length = [], [], 0
    for w_ in words:
        extra = 1 if current else 0
        if length + len(w_) + extra > max_chars:
            if current:
                chunks.append(" ".join(current))
            current = [w_]
            length = len(w_)
        else:
            current.append(w_)
            length += len(w_) + extra
    if current:
        chunks.append(" ".join(current))
    return chunks

# ----------------- Video creation (SINGLE-LINE) -----------------
def create_lyrical_video_file(
    audio_path: str,
    segments: List[Dict],
    out_path: str,
    resolution=(1080,1920),
    fps=FPS,
    text_color="#FFFFFF",
    font=FONT_CHOICES[0],
    animation=DEFAULT_ANIM,
    mode="black",
    background_path: str = None,
    mood_key: str = None,
    max_width_percent: float = 80,
    preset_name: str = "Bold Cinematic"
):
    try:
        max_width_percent = float(max_width_percent)

        if not text_color:
            text_color = DEFAULT_TEXT_COLOR
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        w, h = resolution

        # background
        if mode == "video":
            bg_clip = None
            if background_path:
                try:
                    bg_clip = VideoFileClip(background_path).resize(newsize=resolution).set_duration(duration).set_fps(fps)
                except Exception:
                    bg_clip = None
            if bg_clip is None:
                mk = mood_key or "calm"
                bg_clip = create_mood_clip(mk, resolution, duration)
        elif mode == "image":
            if background_path:
                try:
                    bg_clip = ImageClip(background_path).resize(newsize=resolution).set_duration(duration).set_fps(fps)
                except Exception:
                    bg_clip = ColorClip(size=resolution, color=(0,0,0)).set_duration(duration).set_fps(fps)
            else:
                bg_clip = ColorClip(size=resolution, color=(0,0,0)).set_duration(duration).set_fps(fps)
        else:
            bg_clip = ColorClip(size=resolution, color=(0,0,0)).set_duration(duration).set_fps(fps)

        text_clips = []
        anim_key = (animation or DEFAULT_ANIM).strip().lower()
        max_w_px = int(w * (max_width_percent / 100.0))

        # cleaned segments
        cleaned = []
        cur_end = 0.0
        for seg in segments:
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
            txt = seg.get("text", "").replace("\n", " ").strip()
            if not txt:
                continue
            gap = 0.0
            s = max(s, cur_end + gap)
            if e <= s:
                e = s + 0.1
            cleaned.append({"start": s, "end": e, "text": txt})
            cur_end = e

        for seg in cleaned:
            s, e, txt = seg["start"], seg["end"], seg["text"]
            start = max(0, s)
            end = min(duration, e)
            dur = end - start
            if dur <= 0:
                continue

            # split into chunks -> one chunk on screen at a time
            chunks = split_into_chunks(txt, MAX_CHARS_PER_LINE)
            if not chunks:
                continue
            chunk_dur = dur / len(chunks)

            fontsize = _calc_fontsize_by_width(resolution, txt, max_width_percent, preset_name)
            preset = PRESET_DEFS.get(preset_name, PRESET_DEFS["Bold Cinematic"])
            stroke_w = max(1, int(fontsize * preset.get("stroke_ratio", 0.08)))
            font_resolved = find_font_path(font)
            base_y = int(h * 0.5)

            for i, chunk in enumerate(chunks):
                c_start = start + i * chunk_dur

                try:
                    tc = TextClip(
                        chunk,
                        fontsize=fontsize,
                        font=font_resolved,
                        color=text_color,
                        stroke_color="black",
                        stroke_width=stroke_w,
                        method="caption",
                        size=(max_w_px, None),
                        align="center"
                    )
                except Exception:
                    tc = safe_textclip(chunk, fontsize, font_resolved, text_color, max_w_px)

                tc = tc.set_start(c_start).set_duration(chunk_dur).set_position(("center", base_y))

                if "fade" in anim_key:
                    tc = tc.crossfadein(min(0.25, chunk_dur/6))
                    text_clips.append(tc)
                elif "pop" in anim_key:
                    tc = tc.set_position(
                        lambda t, s0=c_start: (
                            "center",
                            int(base_y - (1 - max(0, min(1, (t - s0)/max(1e-6, chunk_dur)))) * 40)
                        )
                    )
                    text_clips.append(tc)
                elif "slide up" in anim_key or "slideup" in anim_key:
                    tc = tc.set_position(
                        lambda t, s0=c_start: (
                            "center",
                            int(base_y + (1 - max(0, min(1, (t - s0)/max(1e-6, chunk_dur)))) * 180)
                        )
                    )
                    text_clips.append(tc)
                elif "slide left" in anim_key:
                    tc = tc.set_position(
                        lambda t, s0=c_start: (
                            int(w/2 - (1 - max(0, min(1, (t - s0)/max(1e-6, chunk_dur)))) * 360),
                            base_y
                        )
                    )
                    text_clips.append(tc)
                elif "slide right" in anim_key:
                    tc = tc.set_position(
                        lambda t, s0=c_start: (
                            int(w/2 + (1 - max(0, min(1, (t - s0)/max(1e-6, chunk_dur)))) * 360),
                            base_y
                        )
                    )
                    text_clips.append(tc)
                elif "typewriter" in anim_key or "type" in anim_key:
                    chars = list(chunk)
                    total_chars = len(chars)
                    per_char = max(0.03, chunk_dur / max(1, total_chars))
                    accum = ""
                    for j, ch in enumerate(chars):
                        accum += ch
                        try:
                            cclip = TextClip(
                                accum,
                                fontsize=fontsize,
                                font=font_resolved,
                                color=text_color,
                                stroke_color="black",
                                stroke_width=max(1, int(stroke_w*0.6)),
                                method="caption",
                                size=(max_w_px, None),
                                align="center"
                            )
                        except Exception:
                            cclip = safe_textclip(accum, fontsize, font_resolved, text_color, max_w_px)
                        cclip = cclip.set_start(c_start + j * per_char).set_duration(
                            max(0.05, chunk_dur - j * per_char)
                        ).set_position(("center", base_y))
                        text_clips.append(cclip)
                elif "bounce" in anim_key:
                    tc = tc.set_position(
                        lambda tt, s0=c_start: (
                            "center",
                            int(base_y - abs(math.sin((tt - s0) * 6)) * 40 if tt >= s0 else base_y)
                        )
                    )
                    text_clips.append(tc)
                else:
                    text_clips.append(tc)

        if not text_clips:
            fallback = TextClip(
                "Lyrics not available",
                fontsize=36,
                font=find_font_path(FONT_CHOICES[0]),
                color=text_color,
                stroke_color="black",
                stroke_width=3,
                method="caption",
                size=(max_w_px, None),
                align="center"
            )
            fallback = fallback.set_start(0).set_duration(duration).set_position(("center","center"))
            text_clips = [fallback]

        final = CompositeVideoClip([bg_clip, *text_clips]).set_duration(duration).set_audio(audio_clip).set_fps(fps)
        final.write_videofile(
            str(out_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(PROCESSED/"temp-audio.m4a"),
            remove_temp=True,
            fps=fps,
            threads=4,
            preset="medium"
        )
        for clip in text_clips:
            tmp = getattr(clip, "tmp_path", None)
            if tmp and Path(tmp).exists():
                try:
                    Path(tmp).unlink()
                except Exception:
                    pass
        return str(Path(out_path).resolve())
    except Exception as e:
        print("ERROR in create_lyrical_video_file:", e)
        traceback.print_exc()
        return None

# ----------------- Simple font preview -----------------
PREVIEW_SAMPLE_TEXT = "Let your emotions sing"

def build_font_preview_html(font_choice: str):
    if not font_choice:
        font_choice = FONT_CHOICES[0]
    sample = PREVIEW_SAMPLE_TEXT
    html = f"""
<div style="
    background:#111;
    padding:12px 16px;
    border-radius:8px;
    text-align:center;
    color:#ffffff;
    ">
  <span style="font-family:'{font_choice}', sans-serif; font-size:28px;">
    {sample}
  </span>
</div>
"""
    return html

def cb_font_preview(font_choice: str):
    return build_font_preview_html(font_choice)

# ----------------- Callbacks -----------------
def cb_load_audio(youtube_link: str, upload_file, state: dict):
    try:
        source_path, title = None, "Unknown"
        if youtube_link and is_youtube_url(youtube_link):
            source_path = download_youtube_audio(youtube_link)
            title = Path(source_path).stem
        elif upload_file:
            src = Path(upload_file.name) if hasattr(upload_file, "name") else Path(upload_file)
            dest = DOWNLOADS / f"{uuid.uuid4().hex}_{src.name}"
            shutil.copy(str(src), str(dest))
            source_path = str(dest.resolve())
            title = dest.stem
        else:
            return None, "No source", "0.00", "<div style='color:orangered'>Provide YouTube link or upload</div>", state

        preview = ensure_preview_audio(source_path)
        duration = librosa.get_duration(filename=preview)
        duration_text = f"{duration:.2f} sec"
        mood_label = detect_mood_from_audio_file(preview)
        mood_html = f"<div style='padding:10px;border-radius:8px;background:#111;color:#ffcc00;font-weight:700;text-align:center'>{mood_label}</div>"

        state = state or {}
        state["source_path"] = preview
        state["duration"] = float(duration)
        state["title"] = title
        state["segments"] = []
        state["trimmed_path"] = None
        state["mood_label"] = mood_label
        state["mood_key"] = mood_label_to_key(mood_label)

        return preview, title, duration_text, mood_html, state
    except Exception as e:
        print("ERROR in cb_load_audio:", e)
        traceback.print_exc()
        return None, "Error", "0.00", f"<div style='color:orangered'>Error: {e}</div>", state

def cb_trim_and_transcribe(start_time: float, end_time: float, state: dict, youtube_link: str, upload_file):
    try:
        if not state:
            state = {}
        source = state.get("source_path")
        if not source and youtube_link and is_youtube_url(youtube_link):
            source = download_youtube_audio(youtube_link)
            source = ensure_preview_audio(source)
        if not source and upload_file:
            src = Path(upload_file.name) if hasattr(upload_file, "name") else Path(upload_file)
            dest = DOWNLOADS / f"{uuid.uuid4().hex}_{src.name}"
            shutil.copy(str(src), str(dest))
            source = ensure_preview_audio(str(dest.resolve()))
        if not source:
            return None, "<div style='color:orangered'>No source audio loaded.</div>", "<div style='color:orangered'>No mood</div>", None, state

        start = max(0.0, float(start_time))
        end = max(start + 0.01, float(end_time))
        duration = end - start

        converted = PROCESSED / f"converted_{uuid.uuid4().hex}.wav"
        convert_to_wav_for_whisper(source, str(converted))

        trimmed = PROCESSED / f"trimmed_{uuid.uuid4().hex}.wav"
        trim_audio_segment_file(str(converted), start, duration, str(trimmed))

        trans = transcribe_with_whisper_file(str(trimmed))
        segments = trans.get("segments", [])

        lyric_html = create_lyric_html_with_sync(segments)
        mood_label = detect_mood_from_audio_file(str(trimmed))
        mood_html = f"<div style='padding:10px;border-radius:8px;background:#111;color:#ffcc00;font-weight:700;text-align:center'>{mood_label}</div>"

        lrc_path = LYRICS / f"lyrics_{uuid.uuid4().hex}.lrc"
        with open(lrc_path, "w", encoding="utf-8") as f:
            for seg in segments:
                s = seg.get("start", 0.0)
                m = int(s // 60)
                s_rem = s - m * 60
                ts = f"[{m:02d}:{s_rem:05.2f}]"
                f.write(f"{ts} {seg.get('text','')}\n")

        state["trimmed_path"] = str(trimmed.resolve())
        state["segments"] = segments
        state["trim_start"] = start
        state["trim_end"] = end
        state["mood_label"] = mood_label
        state["mood_key"] = mood_label_to_key(mood_label)

        return str(trimmed.resolve()), lyric_html, mood_html, str(lrc_path.resolve()), state
    except Exception as e:
        print("ERROR in cb_trim_and_transcribe:", e)
        traceback.print_exc()
        return None, f"<div style='color:orangered'>Error: {e}</div>", None, None, state

def cb_render_final(
    state: dict,
    selected_template: str,
    custom_bg_file,        # for Custom Background
    text_color: str,
    font_choice: str,
    animation_choice: str,
    platform_choice: str,
    max_width_percent: float,
    preset_choice: str
):
    try:
        if not state:
            return None
        trimmed = state.get("trimmed_path") or state.get("source_path")
        segments = state.get("segments", [])
        if not trimmed or not segments:
            return None

        if not text_color:
            text_color = DEFAULT_TEXT_COLOR

        max_width_percent = float(max_width_percent)

        res_map = {
            "Reels/Shorts (9:16)": (1080, 1920),
            "Instagram Post (1:1)": (1080, 1080),
            "YouTube (16:9)": (1920, 1080)
        }
        resolution = res_map.get(platform_choice, (1080, 1920))
        mood_key = state.get("mood_key", "calm")

        bg_path = None
        mode = "black"

        if selected_template == "Black":
            mode = "black"

        elif selected_template == "Mood Video":
            mode = "video"
            bg_path = state.get("preview_video_path") or get_mood_video_path(mood_key)

        elif selected_template == "Mood Image":
            mode = "image"
            bg_path = state.get("preview_image_path") or get_mood_image_path(mood_key)

        elif selected_template == "Custom Background":
            if custom_bg_file:
                src = Path(custom_bg_file.name) if hasattr(custom_bg_file, "name") else Path(custom_bg_file)
                dest = ASSET_CUSTOM / f"{uuid.uuid4().hex}_{src.name}"
                shutil.copy(str(src), str(dest))
                bg_path = str(dest.resolve())
                ext = dest.suffix.lower()
                if ext in [".mp4",".mov",".mkv",".webm",".avi"]:
                    mode = "video"
                else:
                    mode = "image"
            else:
                mode = "black"
                bg_path = None

        out_video = PROCESSED / f"final_{uuid.uuid4().hex}.mp4"
        final_path = create_lyrical_video_file(
            str(trimmed),
            segments,
            str(out_video),
            resolution=resolution,
            text_color=text_color,
            font=font_choice,
            animation=animation_choice,
            mode=mode,
            background_path=bg_path,
            mood_key=mood_key,
            max_width_percent=max_width_percent,
            preset_name=preset_choice
        )
        return final_path
    except Exception as e:
        print("ERROR in cb_render_final:", e)
        traceback.print_exc()
        return None

# ----------------- Gradio UI -----------------
css = """
body, .gradio-container { background: #0f0f10; color: #ffffff; font-family: Poppins, sans-serif; }
.gr-button { border-radius: 10px !important; padding: 8px 14px !important; }
.gr-row { gap: 16px; }
#audio_preview audio { height: 50px !important; }
"""

with gr.Blocks(title="Smart Lyrical Video Generator", css=css) as demo:
    gr.Markdown("<h2 style='text-align:center; color:#ffcc00'>Smart Lyrical Video Generator</h2>")

    state = gr.State({})

    # ---------- Tab 1: Audio & Mood ----------
    with gr.Tab("Audio & Mood"):
        with gr.Row():
            with gr.Column(scale=2):
                youtube = gr.Textbox(label="YouTube link", placeholder="https://youtu.be/...")
                upload = gr.File(label="Upload audio (.mp3/.wav/.m4a)", file_count="single")
                load_btn = gr.Button("Load Audio & Preview")
                audio_preview = gr.Audio(label="Audio Preview")
                title_box = gr.Textbox(label="Title", interactive=False)
                duration_box = gr.Textbox(label="Duration", interactive=False)
            with gr.Column(scale=1):
                gr.Markdown("### Trim & Extract")
                start_input = gr.Number(value=0.0, label="Start (sec)", precision=2)
                end_input = gr.Number(value=30.0, label="End (sec)", precision=2)
                trim_btn = gr.Button("Trim & Extract Lyrics")
                gr.Markdown("### Mood")
                mood_card = gr.HTML(
                    "<div style='padding:10px;border-radius:8px;background:#111;color:#ffcc00;font-weight:700;text-align:center'>Mood will appear here</div>"
                )
        gr.Markdown("### Lyrics (synchronized while preview plays)")
        lyrics_html = gr.HTML(
            "<div id='lyrics-container' style='max-height:260px; overflow:auto; color:#ddd'></div>"
        )
        gr.Markdown("### Trimmed audio")
        trimmed_audio = gr.Audio(label="Trimmed audio (plays trimmed segment)")
        lrc_file = gr.File(label="Download LRC file (trimmed)")

    # ---------- Tab 2: Video Settings & Output ----------
    with gr.Tab("Video Settings & Output"):
        with gr.Row():
            with gr.Column(scale=1):
                resolution_dropdown = gr.Dropdown(
                    choices=["Reels/Shorts (9:16)", "Instagram Post (1:1)", "YouTube (16:9)"],
                    value=DEFAULT_PLATFORM,
                    label="Platform / Resolution"
                )
                template_radio = gr.Radio(
                    choices=["Black", "Mood Video", "Mood Image", "Custom Background"],
                    value="Black",
                    label="Choose Final Template"
                )
                font_dropdown_final = gr.Dropdown(
                    choices=FONT_CHOICES,
                    value=FONT_CHOICES[0],
                    label="Font (assets/fonts)"
                )
                color_picker_final = gr.ColorPicker(
                    value=DEFAULT_TEXT_COLOR,
                    label="Text color (over background)"
                )
                animation_dropdown = gr.Dropdown(
                    choices=["Fade In", "Pop Up", "Slide Up", "Slide Left",
                             "Slide Right", "Typewriter", "Bounce", "None"],
                    value=DEFAULT_ANIM,
                    label="Animation Style"
                )
                preset_dropdown = gr.Dropdown(
                    choices=list(PRESET_DEFS.keys()),
                    value="Bold Cinematic",
                    label="Lyric Style Preset"
                )
                max_width_slider = gr.Slider(
                    minimum=40,
                    maximum=95,
                    step=5,
                    value=80,
                    label="Max Text Width (%)"
                )

                custom_bg_upload = gr.File(
                    label="Upload custom background (image or video)",
                    file_count="single",
                    visible=False
                )

                font_preview_html = gr.HTML(
                    value=build_font_preview_html(FONT_CHOICES[0]),
                    label="Font Preview"
                )

                generate_btn = gr.Button("Render Final Video")
            with gr.Column(scale=1):
                gr.Markdown("### Final Video Output")
                final_video = gr.Video(label="Final Video (rendered)", height=600)

    # ---------- Wiring ----------
    load_btn.click(
        fn=cb_load_audio,
        inputs=[youtube, upload, state],
        outputs=[audio_preview, title_box, duration_box, mood_card, state],
        show_progress=True
    )

    trim_btn.click(
        fn=cb_trim_and_transcribe,
        inputs=[start_input, end_input, state, youtube, upload],
        outputs=[trimmed_audio, lyrics_html, mood_card, lrc_file, state],
        show_progress=True
    )

    font_dropdown_final.change(
        fn=cb_font_preview,
        inputs=font_dropdown_final,
        outputs=font_preview_html,
    )

    def toggle_custom_bg_visibility(template_choice: str):
        if template_choice == "Custom Background":
            return gr.update(visible=True)
        return gr.update(visible=False)

    template_radio.change(
        fn=toggle_custom_bg_visibility,
        inputs=template_radio,
        outputs=custom_bg_upload,
    )

    generate_btn.click(
        fn=cb_render_final,
        inputs=[
            state,               # state
            template_radio,      # selected_template
            custom_bg_upload,    # custom_bg_file
            color_picker_final,  # text_color
            font_dropdown_final, # font_choice
            animation_dropdown,  # animation_choice
            resolution_dropdown, # platform_choice
            max_width_slider,    # max_width_percent
            preset_dropdown      # preset_choice
        ],
        outputs=final_video,
        show_progress=True
    )

demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
