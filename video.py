"""
video1.py  (Smart Lyrical Video Generator – cleaned version, single-line lyrics)

- Two tabs: Audio & Mood, Video Settings & Output
- Load audio (YouTube/file), trim, Whisper lyrics, simple mood detection
- Video render: Black / Mood Video / Mood Image / Custom Background
- Bigger lyrics, text color from ColorPicker, font preview, custom bg drag&drop
- Lyrics on video: one line at a time (segments split into chunks)

Run: python video1.py
"""
import base64
from io import BytesIO
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
        # Load audio (only 30 seconds to speed it up)
        y, sr = librosa.load(audio_path, sr=16000, duration=30.0)
        
        # 1. Extract Features & FORCE FLOAT CONVERSION
        # librosa returns numpy arrays, so we must wrap them in float() 
        # to use them in standard math/printing without errors.
        tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_raw) 
        
        rms = float(librosa.feature.rms(y=y).mean())
        sc = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y).mean())
        
        # 2. Normalize features
        norm_tempo = (tempo - 60) / 120  
        norm_rms = (rms - 0.02) / 0.08
        norm_sc = (sc - 1000) / 3000
        
        # 3. Calculate Scores
        energy_score = (0.5 * norm_rms) + (0.3 * norm_tempo) + (0.2 * zcr)
        brightness_score = (0.6 * norm_sc) + (0.4 * norm_tempo)
        
        # 4. Decision Logic
        print(f"Debug Mood: Energy={energy_score:.2f}, Brightness={brightness_score:.2f}, Tempo={tempo:.2f}")
        
        if energy_score > 0.35:
            if brightness_score > 0.4:
                return "Energetic ⚡"
            else:
                return "Happy 😄"
        
        elif energy_score < 0.15:
            return "Sad 😢"
            
        else:
            if brightness_score > 0.3:
                return "Happy 😄"
            else:
                return "Calm 🌙"
                
    except Exception as e:
        print("detect_mood error:", e)
        # Fallback to Calm if anything breaks, so the app doesn't crash
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

def format_animated_mood_html(mood_label: str) -> str:
    """
    Returns high-end HTML with animations and colors based on the mood.
    """
    if not mood_label:
        return ""
        
    # Default (Purple)
    color = "#a855f7" 
    bg_gradient = "linear-gradient(135deg, rgba(168, 85, 247, 0.1), rgba(0,0,0,0))"
    emoji = "🎵"
    text = mood_label
    
    # 1. Determine Color & Emoji based on text
    lower = mood_label.lower()
    if "energetic" in lower:
        color = "#facc15" # Yellow/Gold
        bg_gradient = "linear-gradient(135deg, rgba(250, 204, 21, 0.15), rgba(0,0,0,0))"
        emoji = "⚡"
    elif "happy" in lower:
        color = "#4ade80" # Green
        bg_gradient = "linear-gradient(135deg, rgba(74, 222, 128, 0.15), rgba(0,0,0,0))"
        emoji = "😄"
    elif "sad" in lower:
        color = "#60a5fa" # Blue
        bg_gradient = "linear-gradient(135deg, rgba(96, 165, 250, 0.15), rgba(0,0,0,0))"
        emoji = "😢"
    elif "calm" in lower:
        color = "#818cf8" # Indigo
        bg_gradient = "linear-gradient(135deg, rgba(129, 140, 248, 0.15), rgba(0,0,0,0))"
        emoji = "🌙"
        
    # Remove old emojis from text if present to avoid double emoji
    clean_text = mood_label.replace("⚡","").replace("😄","").replace("😢","").replace("🌙","").strip()

    # 2. Build the HTML
    html = f"""
    <div class="mood-card-animated" style="
        background: {bg_gradient};
        border: 2px solid {color};
        color: {color};
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 4px 30px rgba(0,0,0,0.5);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px;
        position: relative;
        overflow: hidden;
    ">
        <div class="mood-emoji" style="font-size: 4rem; filter: drop-shadow(0 0 10px {color});">
            {emoji}
        </div>
        
        <div style="
            font-size: 1.8rem; 
            font-weight: 800; 
            text-transform: uppercase; 
            letter-spacing: 2px;
            text-shadow: 0 0 15px {color};
        ">
            {clean_text}
        </div>
        
        <div style="font-size: 0.8rem; opacity: 0.7; font-weight: 400; color: #cbd5e1;">
            AI Detected Emotion
        </div>
        
        <div style="
            position: absolute;
            inset: 0;
            box-shadow: inset 0 0 30px {color};
            border-radius: 20px;
            opacity: 0.3;
            pointer-events: none;
            animation: pulseGlow 3s infinite;
            color: {color}; 
        "></div>
    </div>
    """
    return html

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
# ----------------- REPLACEMENT 1: FORCE COLOR WITH PIL -----------------
def safe_textclip(txt: str, fontsize: int, fontname_or_path: str, color: str, max_width: int):
    # 1. Setup Color (Convert Hex #FF0000 to RGB Tuple)
    try:
        from PIL import ImageColor
        # If color is missing, default to white
        if not color: 
            fill_color = (255, 255, 255)
        else:
            # This converts "#FF0000" -> (255, 0, 0)
            fill_color = ImageColor.getrgb(str(color))
    except Exception as e:
        print(f"Color Error: {e}")
        fill_color = (255, 255, 255)

    # 2. Setup Font
    font_resolved = find_font_path(fontname_or_path)
    try:
        pil_font = ImageFont.truetype(font_resolved, fontsize)
    except:
        try:
            pil_font = ImageFont.truetype("arial.ttf", fontsize)
        except:
            pil_font = ImageFont.load_default()

    # 3. Create Image Manually (Bypassing MoviePy TextClip)
    img_w = max(400, int(max_width))
    
    # Wrap text
    words = txt.split(" ")
    lines, cur = [], ""
    dummy = Image.new("RGBA", (img_w, 1000), (0,0,0,0))
    draw = ImageDraw.Draw(dummy)
    for w in words:
        test = (cur + " " + w).strip()
        bbox = draw.textbbox((0,0), test, font=pil_font)
        tw = bbox[2] - bbox[0]
        if tw <= img_w - 40: cur = test
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)

    # Calculate Height
    img_h = (len(lines) + 1) * (fontsize + 15)
    img = Image.new("RGBA", (img_w, img_h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    
    y = 5
    stroke_color = (0,0,0) # Black Outline
    
    for line in lines:
        # Draw Outline (Thick)
        for dx, dy in [(2,2), (-2,2), (2,-2), (-2,-2), (0,2), (0,-2), (2,0), (-2,0)]:
            draw.text((10+dx, y+dy), line, font=pil_font, fill=stroke_color)
        
        # Draw Text with ACTUAL COLOR
        draw.text((10, y), line, font=pil_font, fill=fill_color)
        y += fontsize + 12
        
    # Save and Load
    tmp = PROCESSED / f"txt_{uuid.uuid4().hex}.png"
    img.save(tmp)
    ic = ImageClip(str(tmp)).set_duration(1)
    ic.tmp_path = tmp
    return ic
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
    text_color="#FFFFFF",   # <--- Used dynamically now
    font=FONT_CHOICES[0],
    animation=DEFAULT_ANIM,
    mode="black",
    background_path: str = None,
    mood_key: str = None,
    max_width_percent: float = 80,
    preset_name: str = "Bold Cinematic",
    render_lyrics: bool = True # <--- NEW PARAMETER
):
    try:
        max_width_percent = float(max_width_percent)
        if not text_color: text_color = DEFAULT_TEXT_COLOR
        
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        w, h = resolution

        # --- Background Setup ---
        bg_clip = None
        if mode == "video":
            if background_path:
                try:
                    bg_clip = VideoFileClip(background_path).resize(newsize=resolution).set_duration(duration).set_fps(fps)
                except: bg_clip = None
            if bg_clip is None:
                mk = mood_key or "calm"
                bg_clip = create_mood_clip(mk, resolution, duration)
        elif mode == "image":
            if background_path:
                try:
                    bg_clip = ImageClip(background_path).resize(newsize=resolution).set_duration(duration).set_fps(fps)
                except: bg_clip = ColorClip(size=resolution, color=(0,0,0)).set_duration(duration).set_fps(fps)
            else:
                bg_clip = ColorClip(size=resolution, color=(0,0,0)).set_duration(duration).set_fps(fps)
        else:
            bg_clip = ColorClip(size=resolution, color=(0,0,0)).set_duration(duration).set_fps(fps)

        text_clips = []

        # =========================================================
        #   LOGIC: ONLY GENERATE TEXT IF "render_lyrics" IS TRUE
        # =========================================================
        if render_lyrics:
            # --- Helper: Hindi Detection ---
            def is_hindi(text):
                return any(0x0900 <= ord(c) <= 0x097F for c in text)

            # --- Helper: Font Paths ---
            base_font_path = find_font_path(font)
            hindi_font_path = str((FONT_DIR / "hindi.ttf").resolve()) if (FONT_DIR / "hindi.ttf").exists() else base_font_path

            # A. Strict Sorting & Cleaning
            segments = sorted(segments, key=lambda x: x.get("start", 0.0))
            cleaned = []
            for i in range(len(segments)):
                seg = segments[i]
                txt = seg.get("text", "").replace("\n", " ").strip()
                if not txt: continue
                s = float(seg.get("start", 0.0))
                e = float(seg.get("end", 0.0))
                if i < len(segments) - 1:
                    next_s = float(segments[i+1].get("start", 0.0))
                    if next_s < e: e = next_s
                if e - s < 0.3: e = s + 0.3
                e = min(duration, e)
                if e > s: cleaned.append({"start": s, "end": e, "text": txt})

            # B. Generate Clips
            anim_key = (animation or DEFAULT_ANIM).strip().lower()
            max_w_px = int(w * (max_width_percent / 100.0))
            base_y = int(h * 0.5)

            for seg in cleaned:
                s, e, txt = seg["start"], seg["end"], seg["text"]
                dur = e - s
                chunks = split_into_chunks(txt, MAX_CHARS_PER_LINE)
                if not chunks: continue
                chunk_dur = dur / len(chunks)
                
                fontsize = _calc_fontsize_by_width(resolution, txt, max_width_percent, preset_name)
                preset = PRESET_DEFS.get(preset_name, PRESET_DEFS["Bold Cinematic"])
                stroke_w = max(1, int(fontsize * preset.get("stroke_ratio", 0.08)))

                for i, chunk in enumerate(chunks):
                    c_start = s + i * chunk_dur
                    
                    # --- Typewriter Logic ---
                    if "typewriter" in anim_key or "type" in anim_key:
                        chars = list(chunk)
                        total = len(chars)
                        step = chunk_dur / max(1, total) 
                        accum = ""
                        for j, char in enumerate(chars):
                            accum += char
                            if is_hindi(accum):
                                sub_tc = safe_textclip(accum, fontsize, hindi_font_path, text_color, max_w_px)
                            else:
                                try:
                                    sub_tc = TextClip(accum, fontsize=fontsize, font=base_font_path, color=text_color, stroke_color="black", stroke_width=stroke_w, method="caption", size=(max_w_px, None), align="center")
                                except:
                                    sub_tc = safe_textclip(accum, fontsize, base_font_path, text_color, max_w_px)
                            sub_start = c_start + (j * step)
                            sub_dur = step
                            if j == total - 1: sub_dur = max(step, chunk_dur - (j * step))
                            sub_tc = sub_tc.set_start(sub_start).set_duration(sub_dur).set_position(("center", base_y))
                            text_clips.append(sub_tc)
                        continue

                    # --- Standard Logic ---
                    if is_hindi(chunk):
                        tc = safe_textclip(chunk, fontsize, hindi_font_path, text_color, max_w_px)
                    else:
                        try:
                            tc = TextClip(chunk, fontsize=fontsize, font=base_font_path, color=text_color, stroke_color="black", stroke_width=stroke_w, method="caption", size=(max_w_px, None), align="center")
                        except:
                            tc = safe_textclip(chunk, fontsize, base_font_path, text_color, max_w_px)

                    # --- Animations ---
                    tc = tc.set_start(c_start).set_duration(chunk_dur).set_position(("center", base_y))
                    if "zoom" in anim_key:
                        tc = tc.resize(lambda t: 1 + 0.3 * (t / chunk_dur)).set_position(("center", "center"))
                    elif "glitch" in anim_key:
                        import random
                        tc = tc.set_position(lambda t: ("center", base_y + random.randint(-4, 4)))
                    elif "slide" in anim_key:
                        tc = tc.set_position(lambda t: ("center", int(base_y + (1 - min(1, (t / 0.3))) * 120)))
                    elif "pop" in anim_key:
                        tc = tc.set_position(lambda t: ("center", int(base_y + (1 - math.sin(min(1, t/0.4)*math.pi*1.5)) * 50)))
                    elif "wave" in anim_key:
                         tc = tc.set_position(lambda t: ("center", base_y + 10 * math.sin(2 * math.pi * t)))
                    elif "fade" in anim_key:
                        if chunk_dur > 0.5: tc = tc.crossfadein(0.2)
                    
                    text_clips.append(tc)

        # Final Composition
        final_comp = [bg_clip, *text_clips]
        final = CompositeVideoClip(final_comp).set_duration(duration).set_audio(audio_clip).set_fps(fps)
        
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
                try: Path(tmp).unlink()
                except: pass
                
        return str(Path(out_path).resolve())

    except Exception as e:
        print("ERROR in create_lyrical_video_file:", e)
        traceback.print_exc()
        return None

# ----------------- Simple font preview -----------------
PREVIEW_SAMPLE_TEXT = "Let your emotions sing"

# ----------------- REPLACEMENT 2: FIX PREVIEW -----------------
def cb_font_preview(font_choice: str, color_hex: str):
    # Handle defaults
    if not font_choice: font_choice = "Arial"
    if not color_hex: color_hex = "#FFFFFF"

    # Resolve Font
    font_path = find_font_path(font_choice)
    W, H = 800, 150
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype(font_path, 80)
    except:
        font = ImageFont.load_default()

    text = "Let your emotions sing"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (W - text_w) // 2
    y = (H - text_h) // 2
    
    # Draw Shadow
    draw.text((x+4, y+4), text, font=font, fill="black")
    
    # Draw Text with User Color
    try:
        from PIL import ImageColor
        rgb = ImageColor.getrgb(color_hex)
        draw.text((x, y), text, font=font, fill=rgb)
    except:
        draw.text((x, y), text, font=font, fill="white")
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"""<div style="display:flex; justify-content:center; align-items:center; background:#1e1b4b; border-radius:12px; padding:10px; border:1px dashed #4c1d95;">
        <img src="data:image/png;base64,{img_str}" style="max-width:100%;"></div>"""

# ----------------- Callbacks -----------------

def cb_load_audio(youtube_link: str, upload_file, state: dict):
    # --- Define Default/Reset Values for clearing the screen ---
    reset_trimmed = None
    reset_lyrics = "<div id='lyrics-container' style='height:200px; overflow-y:auto; font-size:14px; color:#cbd5e1; padding:10px;'>Lyrics will scroll here...</div>"
    reset_mood = "<div style='background:#1a1a20; padding:15px; border-radius:10px; color:#a855f7; font-weight:bold; text-align:center; border:1px dashed #333'>Waiting for analysis...</div>"
    reset_lrc = None
    reset_video = None
    
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
            # Return error state (clearing everything else)
            return (None, "No source", "0.00", state, "-", "-", 
                    "<div style='color:orangered'>Provide YouTube link or upload</div>",
                    reset_trimmed, reset_lyrics, reset_mood, reset_lrc, reset_video)

        preview = ensure_preview_audio(source_path)
        duration = librosa.get_duration(filename=preview)
        
        # Audio Stats
        sr_text = "44100 Hz"
        chan_text = "Stereo"
        summary_html = "<div style='color:#64748b; font-size:13px; text-align:center;'>Could not analyze audio</div>"
        
        try:
            y_temp, sr_temp = librosa.load(preview, sr=None, duration=1.0)
            sr_text = f"{sr_temp} Hz"
            is_stereo = len(y_temp.shape) > 1
            chan_text = f"Stereo ({y_temp.shape[0]})" if is_stereo else "Mono (1)"
            
            if sr_temp >= 44100 and is_stereo:
                summary_html = f"<div style='background:rgba(34, 197, 94, 0.1); border:1px solid #22c55e; color:#22c55e; padding:8px; border-radius:6px; text-align:center; font-size:13px; margin-top:10px;'>✅ <b>Studio Quality:</b> High-res stereo.</div>"
            elif sr_temp >= 44100:
                summary_html = f"<div style='background:rgba(234, 179, 8, 0.1); border:1px solid #eab308; color:#eab308; padding:8px; border-radius:6px; text-align:center; font-size:13px; margin-top:10px;'>⚠️ <b>Good Quality (Mono):</b> Clear but lacks stereo depth.</div>"
            else:
                summary_html = f"<div style='background:rgba(239, 68, 68, 0.1); border:1px solid #ef4444; color:#ef4444; padding:8px; border-radius:6px; text-align:center; font-size:13px; margin-top:10px;'>❌ <b>Low Quality:</b> {sr_temp}Hz.</div>"
        except Exception:
            pass

        duration_text = f"{duration:.2f} sec"

        # Update State (Clear previous segments/trimmed path)
        state = state or {}
        state["source_path"] = preview
        state["duration"] = float(duration)
        state["title"] = title
        state["segments"] = []      # CLEAR OLD SEGMENTS
        state["trimmed_path"] = None # CLEAR OLD TRIM
        state["mood_label"] = None
        state["mood_key"] = "calm"

        return (
            preview, title, duration_text, state, sr_text, chan_text, summary_html,
            reset_trimmed, reset_lyrics, reset_mood, reset_lrc, reset_video
        )
        
    except Exception as e:
        print("ERROR in cb_load_audio:", e)
        traceback.print_exc()
        return (
            None, "Error", "0.00", state, "-", "-", f"<div style='color:orangered'>Error: {e}</div>",
            reset_trimmed, reset_lyrics, reset_mood, reset_lrc, reset_video
        )

def cb_trim_and_transcribe(start_time, end_time, state, youtube_link, upload_file, model_size_choice, language_choice):
    """
    Updated to handle dynamic Model Size AND Language selection.
    """
    try:
        if not state:
            state = {}
            
        # 1. Handle Audio Source
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

        # 2. Trim Audio
        start = max(0.0, float(start_time))
        end = max(start + 0.01, float(end_time))
        duration = end - start

        converted = PROCESSED / f"converted_{uuid.uuid4().hex}.wav"
        convert_to_wav_for_whisper(source, str(converted))

        trimmed = PROCESSED / f"trimmed_{uuid.uuid4().hex}.wav"
        trim_audio_segment_file(str(converted), start, duration, str(trimmed))

        # 3. Load Selected Whisper Model
        target_model = "small"
        if "tiny" in model_size_choice.lower(): target_model = "tiny"
        elif "base" in model_size_choice.lower(): target_model = "base"
        elif "small" in model_size_choice.lower(): target_model = "small"
            
        global _WHISPER_MODEL_CACHE
        _WHISPER_MODEL_CACHE = whisper.load_model(target_model)
        
        # 4. --- NEW: DETECT LANGUAGE SELECTION ---
        # Map UI Text -> Whisper Language Code
        lang_code = None # Default is None (Auto-Detect)
        
        if "english" in language_choice.lower(): lang_code = "en"
        elif "hindi" in language_choice.lower(): lang_code = "hi"
        elif "spanish" in language_choice.lower(): lang_code = "es"
        elif "french" in language_choice.lower(): lang_code = "fr"
        # If "Auto-Detect", lang_code remains None
        
        # 5. Transcribe with Language Force
        # We pass 'language=lang_code' to the transcribe function
        result = _WHISPER_MODEL_CACHE.transcribe(str(trimmed), fp16=False, language=lang_code)
        segments = result.get("segments", [])
        
        # 6. Format Results
        lyric_html = create_lyric_html_with_sync(segments)
        
        # Add Keyword Cloud Logic (if you kept that feature)
        # ... (Your existing keyword cloud logic goes here if you have it) ...

        mood_label = detect_mood_from_audio_file(str(trimmed))
        mood_html = format_animated_mood_html(mood_label)

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

        # If you added the 'Visualizer' or 'Tags' features, ensure your return outputs match your button outputs!
        # Assuming standard setup:
        return str(trimmed.resolve()), lyric_html, mood_html, str(lrc_path.resolve()), state
        
    except Exception as e:
        print("ERROR in cb_trim_and_transcribe:", e)
        traceback.print_exc()
        return None, f"<div style='color:orangered'>Error: {e}</div>", None, None, state

def cb_render_final(
    state: dict,
    selected_template: str,
    custom_bg_file,        
    text_color: str,       # <--- NEW INPUT
    render_mode: str,      # <--- NEW INPUT ("With Lyrics", "Video Only")
    font_choice: str,
    animation_choice: str,
    platform_choice: str,
    max_width_percent: float,
    preset_choice: str
):
    try:
        if not state: return None
        trimmed = state.get("trimmed_path") or state.get("source_path")
        segments = state.get("segments", [])
        if not trimmed: return None

        max_width_percent = float(max_width_percent)

        res_map = {
            "Reels/Shorts (9:16)": (1080, 1920),
            "Instagram Post (1:1)": (1080, 1080),
            "YouTube (16:9)": (1920, 1080)
        }
        resolution = res_map.get(platform_choice, (1080, 1920))
        mood_key = state.get("mood_key", "calm")

        # Determine Background
        bg_path, mode = None, "black"
        if selected_template == "Black": mode = "black"
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
                mode = "video" if dest.suffix.lower() in [".mp4",".mov",".webm",".avi"] else "image"
            else:
                mode = "black"

        # Determine Lyrics Visibility
        should_render_lyrics = (render_mode == "With Lyrics")

        out_video = PROCESSED / f"final_{uuid.uuid4().hex}.mp4"
        final_path = create_lyrical_video_file(
            str(trimmed),
            segments,
            str(out_video),
            resolution=resolution,
            text_color=text_color,  # <--- Passing the color
            font=font_choice,
            animation=animation_choice,
            mode=mode,
            background_path=bg_path,
            mood_key=mood_key,
            max_width_percent=max_width_percent,
            preset_name=preset_choice,
            render_lyrics=should_render_lyrics # <--- Passing the toggle
        )
        return final_path
    except Exception as e:
        print("ERROR in cb_render_final:", e)
        traceback.print_exc()
        return None

# =========================================================
#            MODERN UI SECTION (Home Page + Studio)
# =========================================================

# 1. Define the Theme (Forcing Black/Transparent)
theme = gr.themes.Base(
    primary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui'],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#000000",       
    block_background_fill="transparent",   # <--- FIX: Removes gray block background
    block_border_color="#333",
    block_border_width="1px",
    input_background_fill="#050505",       # Almost black inputs
    button_primary_background_fill="linear-gradient(90deg, #7c3aed 0%, #d946ef 100%)", 
    button_primary_text_color="#FFFFFF",
    slider_color="#a855f7",               
    body_text_color="#e2e8f0"
)

# 2. Custom CSS (Pure Black Glass)
custom_css = """
/* Background Glow */
.gradio-container { 
    background: radial-gradient(circle at 20% 0%, #1e1b4b 0%, #000000 50%) !important; 
    min-height: 100vh;
}

/* Animations */
@keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes float { 0% { transform: translateY(0px); } 50% { transform: translateY(-10px); } 100% { transform: translateY(0px); } }

.hero-container { text-align: center; padding: 60px 20px; animation: fadeIn 0.8s ease-out; }
.hero-title { font-size: 4rem; font-weight: 900; background: linear-gradient(to right, #fff, #a855f7, #d946ef); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }
.hero-subtitle { font-size: 1.2rem; color: #94a3b8; max-width: 600px; margin: 0 auto 40px auto; line-height: 1.6; }

/* Feature Cards - Pure Black Transparent */
.glass-card { 
    background: rgba(0, 0, 0, 0.4); /* FIX: Black instead of Gray */
    border: 1px solid rgba(255, 255, 255, 0.1); 
    border-radius: 16px; 
    padding: 30px; 
    transition: transform 0.3s, border-color 0.3s;
}
.glass-card:hover { transform: translateY(-5px); border-color: #a855f7; }

/* Studio Panels - Pure Black Transparent */
.glass-panel { 
    background: rgba(0, 0, 0, 0.6) !important; /* FIX: Darker Black Transparency */
    backdrop-filter: blur(15px); 
    border: 1px solid rgba(255, 255, 255, 0.1); 
    border-radius: 16px !important; 
    padding: 20px !important; 
    box-shadow: 0 4px 20px rgba(0,0,0,0.6); 
}

/* Buttons & Tabs */
#action-btn { background: linear-gradient(90deg, #7c3aed 0%, #db2777 100%) !important; border: none; box-shadow: 0 0 15px rgba(124, 58, 237, 0.5); font-size: 1.1rem; font-weight: 700; height: 60px; transition: transform 0.2s; }
#action-btn:hover { transform: scale(1.02); box-shadow: 0 0 25px rgba(124, 58, 237, 0.7); }
#get-started-btn { font-size: 1.3rem; padding: 10px 40px; height: 60px; box-shadow: 0 0 20px rgba(124, 58, 237, 0.3); }

/* Remove Gray from Tabs */
.tabs { background: transparent !important; border: none !important; }
.tabs > button { border: none !important; background: transparent !important; color: #94a3b8 !important; font-weight: 600; font-size: 16px; }
.tabs > button.selected { color: #d8b4fe !important; border-bottom: 2px solid #d8b4fe !important; background: transparent !important; }

/* Inputs */
.search-bar textarea { background: #050505 !important; border: 1px solid #333 !important; border-radius: 10px !important; }
#lyrics-box { background: #050505 !important; border: 1px solid #333; border-radius: 12px; }
"""

with gr.Blocks(theme=theme, css=custom_css, title="Lyrical Gen Ultra") as demo:
    
    state = gr.State({})

    # ================= TOP NAVIGATION TABS =================
    with gr.Tabs(elem_id="main-tabs") as main_nav:

        # ---------------- TAB 1: HOME ----------------
        with gr.TabItem("🏠 Home", id="tab_home"):
            
            # Hero Section
            gr.HTML("""
            <div class="hero-container">
                <div style="font-size: 5rem; margin-bottom: 10px; animation: float 4s ease-in-out infinite;">🎵</div>
                <h1 class="hero-title">Lyrical Gen <span style="font-style:italic">Ultra</span></h1>
                <p class="hero-subtitle">
                    Create stunning lyrical videos in seconds.<br>
                    AI Transcription • Mood Detection • Cinematic Animations.
                </p>
            </div>
            """)
            
            # CTA Button
            with gr.Row():
                with gr.Column(scale=1): pass
                with gr.Column(scale=1):
                    start_btn = gr.Button("✨ Start Creating Now", variant="primary", elem_id="get-started-btn")
                with gr.Column(scale=1): pass

            # Feature Grid
            gr.HTML("""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; padding: 40px 20px; max-width: 1000px; margin: 0 auto;">
                <div class="glass-card">
                    <div style="font-size: 2rem; margin-bottom: 10px;">🎙️</div>
                    <h3 style="color:white; font-weight:bold; margin-bottom:5px;">Whisper AI</h3>
                    <p style="color:#94a3b8; font-size:0.9rem;">Auto-transcribes audio with support for English, Hindi, and more.</p>
                </div>
                <div class="glass-card">
                    <div style="font-size: 2rem; margin-bottom: 10px;">🧠</div>
                    <h3 style="color:white; font-weight:bold; margin-bottom:5px;">Mood Intelligence</h3>
                    <p style="color:#94a3b8; font-size:0.9rem;">Detects if your song is Happy, Sad, or Energetic and adapts visuals.</p>
                </div>
                <div class="glass-card">
                    <div style="font-size: 2rem; margin-bottom: 10px;">🎨</div>
                    <h3 style="color:white; font-weight:bold; margin-bottom:5px;">Cinematic FX</h3>
                    <p style="color:#94a3b8; font-size:0.9rem;">Typewriter, Glitch, Zoom, and Wave animations built-in.</p>
                </div>
            </div>
            """)

        # ---------------- TAB 2: CREATOR STUDIO ----------------
        with gr.TabItem("🎨 Creator Studio", id="tab_create"):
            
            gr.Markdown("<br>") # Spacer

            # --- STUDIO TABS ---
            with gr.Tabs():

                # === SUB-TAB 1: AUDIO ===
                with gr.Tab("Audio & Analysis"):
                    with gr.Row():
                        with gr.Column(scale=3, elem_classes=["glass-panel"]):
                            gr.Markdown("### 📂 Source Media")
                            with gr.Row():
                                youtube = gr.Textbox(label="YouTube Link", placeholder="https://...", show_label=False, elem_classes=["search-bar"], scale=3)
                                upload = gr.File(label="", file_count="single", type="filepath", scale=1, min_width=50)
                            
                            load_btn = gr.Button("Load Source", variant="secondary", size="sm")
                            
                            gr.Markdown("### ✂️ Trim Audio", elem_id="trim-header")
                            with gr.Row(equal_height=True):
                                start_input = gr.Number(value=0.0, label="Start", show_label=False, container=False, scale=4)
                                with gr.Column(scale=1, min_width=10): 
                                    gr.Markdown("<div style='text-align:center; margin-top:10px'>-</div>")
                                end_input = gr.Number(value=30.0, label="End", show_label=False, container=False, scale=4)
                            
                            with gr.Row(visible=True):
                                title_box = gr.Textbox(label="Track", interactive=False, container=False, text_align="right")
                                duration_box = gr.Textbox(label="Dur", interactive=False, container=False)

                        with gr.Column(scale=2, elem_classes=["glass-panel"]):
                            gr.Markdown("### ⚡ Process")
                            trim_btn = gr.Button("Transcribe & Detect Mood", elem_id="action-btn")
                            
                            gr.Markdown("### ⚙️ Engine Configuration")
                            with gr.Group():
                                with gr.Row():
                                    model_selector = gr.Dropdown(["Tiny", "Base", "Small"], value="Small", label="Model Size", interactive=True, scale=1)
                                    language_selector = gr.Dropdown(["Auto-Detect", "English", "Hindi"], value="Auto-Detect", label="Language", interactive=True, scale=1)

                            gr.Markdown("### 📊 Audio Intelligence")
                            with gr.Group():
                                with gr.Row():
                                    sr_box = gr.Textbox(label="Sample Rate", value="-", interactive=False, elem_id="stat-card")
                                    channels_box = gr.Textbox(label="Channels", value="-", interactive=False, elem_id="stat-card")
                            
                            quality_summary = gr.HTML(value="<div style='color:#64748b; font-size:13px; text-align:center; margin-top:8px;'>Waiting for analysis...</div>")

                    with gr.Row():
                        with gr.Column(scale=3, elem_classes=["glass-panel"]):
                            gr.Markdown("### 🎵 Audio Preview")
                            audio_preview = gr.Audio(label="Original", visible=True) 
                            trimmed_audio = gr.Audio(label="Processed Clip", interactive=False, elem_id="main-audio")
                            lrc_file = gr.File(label="Download LRC", visible=True)

                        with gr.Column(scale=2, elem_classes=["glass-panel"]):
                            gr.Markdown("### 🧠 Detected Mood")
                            mood_card = gr.HTML("<div style='background:#1a1a20; padding:15px; border-radius:10px; color:#a855f7; font-weight:bold; text-align:center; border:1px dashed #333'>Waiting for analysis...</div>")
                            gr.Markdown("### 📝 Synced Lyrics")
                            lyrics_html = gr.HTML("<div id='lyrics-container' style='height:200px; overflow-y:auto; font-size:14px; color:#cbd5e1; padding:10px;'>Lyrics will scroll here...</div>", elem_id="lyrics-box")

                # === SUB-TAB 2: VIDEO ===
                with gr.Tab("Video Settings"):
                    with gr.Row():
                        # Left Sidebar
                        with gr.Column(scale=1, elem_classes=["glass-panel"]):
                            gr.Markdown("### ⚙️ Output Settings")
                            
                            # Platform & Render Mode
                            gr.Markdown("**Format & Content**")
                            resolution_dropdown = gr.Radio(["Reels/Shorts (9:16)", "Instagram Post (1:1)", "YouTube (16:9)"], value="Reels/Shorts (9:16)", label="Aspect Ratio", container=True)
                            render_mode_radio = gr.Radio(["With Lyrics", "Video Only"], value="With Lyrics", label="Render Content", container=True)

                            gr.Markdown("---")
                            
                            # Background
                            gr.Markdown("**Background Template**")
                            template_radio = gr.Dropdown(["Mood Video", "Mood Image", "Black", "Custom Background"], value="Mood Video", label="", container=False)
                            custom_bg_upload = gr.File(label="Upload Background", visible=False)

                            gr.Markdown("---")
                            
                            # Text
                            gr.Markdown("**Lyric Appearance**")
                            with gr.Row(equal_height=True):
                                font_dropdown_final = gr.Dropdown(FONT_CHOICES, value=FONT_CHOICES[0], label="Font Family", scale=2, container=True)
                                # Color Picker
                                color_picker_final = gr.ColorPicker(value="#FFFFFF", label="Color", scale=1, container=True)
                            
                            preset_dropdown = gr.Dropdown(list(PRESET_DEFS.keys()), value="Bold Cinematic", label="Effect Preset")
                            
                            animation_dropdown = gr.Dropdown(
                                choices=["Fade In", "Typewriter", "Cinematic Zoom", "Nervous Glitch", "Float Wave", "Slide Up", "Pop Up"], 
                                value="Fade In", 
                                label="Animation"
                            )

                            max_width_slider = gr.Slider(minimum=40, maximum=95, value=80, label="Text Width %")
                            
                            gr.Markdown("<br>")
                            generate_btn = gr.Button("Render Video 🚀", variant="primary")

                        # Right Stage
                        with gr.Column(scale=2, elem_classes=["glass-panel"]):
                            gr.Markdown("### 🎬 Preview Monitor")
                            font_preview_html = gr.HTML(label="Font Style Check")
                            final_video = gr.Video(label="Final Output", height=500, interactive=False)

    # ---------------- LOGIC WIRING ----------------
    
    # 0. SWITCH TAB LOGIC
    def switch_to_creator():
        return gr.Tabs(selected="tab_create")
        
    start_btn.click(fn=switch_to_creator, inputs=None, outputs=main_nav)

    # 1. Load Audio
    load_btn.click(
        fn=cb_load_audio,
        inputs=[youtube, upload, state],
        outputs=[audio_preview, title_box, duration_box, state, sr_box, channels_box, quality_summary, trimmed_audio, lyrics_html, mood_card, lrc_file, final_video],
        show_progress=True
    )

    # 2. Trim & Transcribe
    trim_btn.click(
        fn=cb_trim_and_transcribe,
        inputs=[start_input, end_input, state, youtube, upload, model_selector, language_selector],
        outputs=[trimmed_audio, lyrics_html, mood_card, lrc_file, state],
        show_progress=True
    )

    # 3. Dynamic UI updates
    # ----------------- REPLACEMENT 3: CORRECT WIRING -----------------
    
    # 1. Update Preview when FONT changes (Send Font AND Color)
    font_dropdown_final.change(
        fn=cb_font_preview, 
        inputs=[font_dropdown_final, color_picker_final], 
        outputs=font_preview_html
    )
    
    # 2. Update Preview when COLOR changes (Send Font AND Color)
    color_picker_final.change(
        fn=cb_font_preview, 
        inputs=[font_dropdown_final, color_picker_final], 
        outputs=font_preview_html
    )

    # 3. Toggle Background Upload
    def toggle_custom_bg_visibility(template_choice):
        return gr.update(visible=(template_choice == "Custom Background"))
    template_radio.change(fn=toggle_custom_bg_visibility, inputs=template_radio, outputs=custom_bg_upload)

    # 4. Generate Video (Sending COLOR correctly)
    generate_btn.click(
        fn=cb_render_final,
        inputs=[
            state,                  
            template_radio,         
            custom_bg_upload,       
            color_picker_final,     # <--- CRITICAL: Passing Color Here
            render_mode_radio,      
            font_dropdown_final,    
            animation_dropdown,     
            resolution_dropdown,    
            max_width_slider,       
            preset_dropdown         
        ],
        outputs=final_video,
        show_progress=True
    )
    
demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
