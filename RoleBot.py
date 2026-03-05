import discord
import os
import io

# Load .env so DISCORD_TOKEN and GEMINI_API_KEY are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import asyncio
from discord.ext import commands, tasks
from datetime import datetime, timezone
from PIL import Image
import requests

intents = discord.Intents.default()
intents.members = True
intents.message_content = True  
bot = commands.Bot(command_prefix='!', intents=intents)

GUILD_ID = 1014861739780755519
PROMOTION_CHANNEL_ID = 1020680595757596802
COOKING_CHANNEL_ID = 1451575975035801744  # Channel where pickle images are BANNED

# Pickle detection (WhatAPickle)
# Long-term: zero-shot alone is weak; train a head with !pickle_teach + finetune, set WHATAPICKLE_HEAD_PATH.
CLIP_ENABLED = True
PICKLE_CONFIDENCE_THRESHOLD = 0.55  # 0.5 is fine once you use a fine-tuned head
WHATAPICKLE_HEAD_PATH = "WhatAPickle/output/pickle_head_logreg.pkl"

# Save images for future offline training (builds dataset for re-training)
SAVE_IMAGES_FOR_TRAINING = True
TRAINING_SAVE_DIR = "WhatAPickle/data/discord_collected"
TRAINING_SAVE_MIN_CONFIDENCE = 0.85  # set to 0 to save every image (then curate mislabels by hand)

# Global for lazy-loaded detector
pickle_detector = None

# Umor emoji ping: when :Umor: is used, ping this user in the same channel
UMOR_EMOJI_ID = "1472933034398318614"
UMOR_USER_ID = 359397805682589698

CHECKED_ROLE_IDS = {
    1020682173759627265,  
    1055504940019232870,  
    1037479415472459886,  
}

ROLE_THRESHOLDS = [
    (0,   1382042066468737164),  # Recruit - 1 Month
    (90,   1382044635681259640),  # Squire - 3 Months
    (180,  1382047496343388240),  # Knight - 6 Months
    (365,  1382047626417143898),  # Champion - 1 Year
    (730,  1382061051142602822),  # Veteran - 2 Years
    (1095, 1382058816878678116),  # Warlord - 3 Years
    (1460, 1382066528421281802),  # Elder - 4 Years
    (1825, 1382071830797615114),  # Legend - 5 Years
]

START_DATE = datetime(2023, 12, 30, tzinfo=timezone.utc)


# ========== WHATAPICKLE DETECTOR ==========

def get_pickle_detector():
    """Lazy load WhatAPickle detector."""
    global pickle_detector
    if pickle_detector is not None:
        return pickle_detector
    try:
        from pickle_detector.detector import PickleDetector
        print("🔄 Loading WhatAPickle detector...")
        pickle_detector = PickleDetector(
            threshold=PICKLE_CONFIDENCE_THRESHOLD,
            head_path=WHATAPICKLE_HEAD_PATH or None,
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        )
        print(f"✅ WhatAPickle detector loaded on {pickle_detector.device} ({pickle_detector.mode})")
        return pickle_detector
    except ImportError as e:
        print(f"❌ WhatAPickle not installed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading detector: {e}")
        return None


def unload_pickle_detector():
    """Unload detector to free memory."""
    global pickle_detector
    if pickle_detector is not None:
        pickle_detector = None
        print("🗑️ Pickle detector unloaded")


def _save_frame_for_training_sync(frame_pil, is_pickle: bool, score: float, save_dir: str):
    """Run in thread: save one image to pickle/ or non_pickle/ with unique filename."""
    import hashlib
    from datetime import datetime
    subdir = "pickle" if is_pickle else "non_pickle"
    out_dir = os.path.join(save_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    raw = io.BytesIO()
    frame_pil.save(raw, format="JPEG", quality=90)
    raw = raw.getvalue()
    h = hashlib.sha256(raw).hexdigest()[:8]
    name = f"discord_{ts}_{h}.jpg"
    path = os.path.join(out_dir, name)
    with open(path, "wb") as f:
        f.write(raw)
    print(f"💾 Saved for training: {path}")


def _get_first_frame_from_url_sync(url: str):
    """Run in thread: download URL (image or video) and return first frame as PIL, or None on failure."""
    import tempfile
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"⚠️ Teach: failed to download: {e}")
        return None
    is_video = url.lower().endswith(('.mp4', '.webm', '.mov'))
    if is_video:
        try:
            import cv2
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            ret, frame = cap.read()
            cap.release()
            os.unlink(tmp_path)
            if not ret or frame is None:
                return None
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
        except Exception as e:
            print(f"⚠️ Teach: video frame extraction failed: {e}")
            return None
    try:
        image_data = io.BytesIO(response.content)
        image = Image.open(image_data).convert("RGB")
        return image
    except Exception as e:
        print(f"⚠️ Teach: image open failed: {e}")
        return None


async def check_image_for_pickle(image_url: str) -> tuple[bool, float]:
    """
    Check if an image contains a pickle using CLIP.
    For GIFs, checks multiple frames.
    Returns: (has_pickle: bool, confidence: float)
    """
    if not CLIP_ENABLED:
        return True, 1.0  # Skip check if disabled
    
    try:
        # Download media
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Check if it's a video (MP4/WebM - Discord converts GIFs to MP4)
        is_video = image_url.lower().endswith(('.mp4', '.webm', '.mov'))
        frames_to_check = []
        
        if is_video:
            # For videos (Discord GIFs converted to MP4), extract frames
            print(f"🎥 Analyzing video/GIF (MP4 format)...")
            try:
                import cv2
                import numpy as np
                import tempfile
                
                # Save video to temp file (opencv needs file path)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(response.content)
                    tmp_path = tmp_file.name
                
                # Open video with opencv
                cap = cv2.VideoCapture(tmp_path)
                
                # Extract first 5 frames
                frame_count = 0
                max_frames = 5
                
                while frame_count < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert BGR (opencv) to RGB (PIL)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(frame_rgb)
                    frames_to_check.append(pil_frame)
                    frame_count += 1
                
                cap.release()
                
                # Clean up temp file
                import os
                os.unlink(tmp_path)
                
                print(f"🎥 Extracted {frame_count} frames from video")
                
                if not frames_to_check:
                    print(f"⚠️ No frames extracted from video")
                    return True, 1.0
                    
            except ImportError:
                print(f"⚠️ opencv-python not installed - run: pip install opencv-python")
                return True, 1.0  # Allow if opencv not available
            except Exception as e:
                print(f"⚠️ Error extracting video frames: {e}")
                return True, 1.0
        else:
            # Image (GIF, PNG, JPG, etc.)
            image_data = io.BytesIO(response.content)
            image = Image.open(image_data)
            
            # Check if it's a GIF (animated)
            is_gif = (image.format == "GIF" and getattr(image, "is_animated", False)) or \
                     (image_url.lower().endswith('.gif') and getattr(image, "n_frames", 1) > 1)
            
            if is_gif:
                # For GIFs, extract first 5 frames (or all if less than 5)
                try:
                    frame_count = min(5, getattr(image, "n_frames", 1))
                    print(f"🎞️ Analyzing GIF with {frame_count} frames...")
                    
                    for frame_idx in range(frame_count):
                        image.seek(frame_idx)
                        frame = image.convert("RGB")
                        frames_to_check.append(frame)
                except Exception as e:
                    print(f"⚠️ Error extracting GIF frames: {e}, checking first frame only")
                    frames_to_check = [image.convert("RGB")]
            else:
                # Static image
                frames_to_check = [image.convert("RGB")]
        
        # Load detector in a thread so HF download + model load don't block the event loop
        detector = await asyncio.to_thread(get_pickle_detector)
        if detector is None:
            return True, 1.0

        max_pickle_confidence = 0.0
        best_frame = None
        best_result = None
        is_multi = len(frames_to_check) > 1

        for frame_idx, frame in enumerate(frames_to_check):
            result = detector.detect(frame)
            if result.pickle_score > max_pickle_confidence:
                max_pickle_confidence = result.pickle_score
                best_frame = frame
                best_result = result
            if is_multi:
                print(f"🔍 Frame {frame_idx + 1}: {result.pickle_score:.2%} pickle confidence")
            if result.is_pickle:
                break

        has_pickle = max_pickle_confidence >= detector.threshold
        if is_multi:
            print(f"🎞️ Analysis complete: Max confidence {max_pickle_confidence:.2%} | Has pickle: {has_pickle}")
        else:
            if best_result and getattr(best_result, "zero_shot_score", None) is not None and getattr(best_result, "finetuned_score", None) is not None:
                print(f"🔍 Pickle: zero_shot={best_result.zero_shot_score:.2%} finetuned={best_result.finetuned_score:.2%} combined={max_pickle_confidence:.2%} | Has pickle: {has_pickle}")
            else:
                print(f"🔍 Pickle detection: {max_pickle_confidence:.2%} confidence | Has pickle: {has_pickle}")

        if SAVE_IMAGES_FOR_TRAINING and best_frame is not None:
            do_save = False
            if TRAINING_SAVE_MIN_CONFIDENCE <= 0:
                do_save = True
            elif has_pickle and max_pickle_confidence >= TRAINING_SAVE_MIN_CONFIDENCE:
                do_save = True
            elif not has_pickle and max_pickle_confidence <= (1 - TRAINING_SAVE_MIN_CONFIDENCE):
                do_save = True
            if do_save:
                asyncio.create_task(asyncio.to_thread(
                    _save_frame_for_training_sync,
                    best_frame,
                    has_pickle,
                    max_pickle_confidence,
                    TRAINING_SAVE_DIR,
                ))

        return has_pickle, max_pickle_confidence
        
    except requests.RequestException as e:
        print(f"⚠️ Failed to download image: {e}")
        return True, 1.0  # Allow if download fails
    except Exception as e:
        print(f"⚠️ Error checking image: {e}")
        return True, 1.0  # Allow if check fails


async def extract_media_from_embeds(message):
    """Extract image/video URLs from Discord embeds (GIF picker, Tenor, Giphy, etc.)"""
    media_urls = []
    
    # Check message embeds (from GIF picker)
    if message.embeds:
        print(f"🔗 Found {len(message.embeds)} embed(s)")
        for embed in message.embeds:
            # Check embed image
            if embed.image and embed.image.url:
                print(f"   - Embed image: {embed.image.url}")
                media_urls.append(("embed_image", embed.image.url))
            
            # Check embed video
            if embed.video and embed.video.url:
                print(f"   - Embed video: {embed.video.url}")
                media_urls.append(("embed_video", embed.video.url))
            
            # Check embed thumbnail
            if embed.thumbnail and embed.thumbnail.url:
                print(f"   - Embed thumbnail: {embed.thumbnail.url}")
                media_urls.append(("embed_thumbnail", embed.thumbnail.url))
    
    return media_urls


async def process_message_attachments(message):
    """Process all image/video attachments in a message for pickle detection"""
    results = []
    all_have_pickles = True
    
    # Check file attachments
    if message.attachments:
        print(f"📎 Found {len(message.attachments)} attachment(s)")
        for att in message.attachments:
            print(f"   - {att.filename} | Type: {att.content_type} | URL: {att.url}")
        
        # Include both images and videos (Discord converts GIFs to MP4)
        image_attachments = [
            att for att in message.attachments 
            if att.content_type and (
                att.content_type.startswith('image/') or 
                att.content_type.startswith('video/')
            )
        ]
        
        if image_attachments:
            print(f"✅ Processing {len(image_attachments)} file attachment(s)")
            
            for attachment in image_attachments:
                has_pickle, confidence = await check_image_for_pickle(attachment.url)
                results.append((attachment.filename, has_pickle, confidence))
                
                if not has_pickle:
                    all_have_pickles = False
    
    # Check embeds (GIF picker, Tenor, Giphy)
    embed_media = await extract_media_from_embeds(message)
    if embed_media:
        print(f"✅ Processing {len(embed_media)} embed media")
        
        for media_type, media_url in embed_media:
            has_pickle, confidence = await check_image_for_pickle(media_url)
            results.append((f"{media_type}", has_pickle, confidence))
            
            if not has_pickle:
                all_have_pickles = False
    
    # If no media found at all
    if not results:
        return True, []
    
    return all_have_pickles, results


@bot.event
async def on_ready():
    print(f'{bot.user} is online and watching for promotions.')
    # Set bot nickname to SBbot on the main guild
    guild = bot.get_guild(GUILD_ID)
    if guild:
        try:
            await guild.me.edit(nick="SBbot")
        except discord.Forbidden:
            print("❌ Missing permission to change bot nickname.")
        except Exception as e:
            print(f"⚠️ Error setting bot nickname: {e}")
    
    # Start update_roles task only if not already running
    if not update_roles.is_running():
        update_roles.start()

    # Preload pickle detector in background so first image doesn't block the bot
    if CLIP_ENABLED:
        async def _preload_detector():
            try:
                await asyncio.to_thread(get_pickle_detector)
            except Exception as e:
                print(f"⚠️ Preload detector: {e}")
        asyncio.create_task(_preload_detector())

@bot.event
async def on_member_join(member):
    guild = bot.get_guild(GUILD_ID)
    recruit_role = guild.get_role(1382042066468737164)
    if recruit_role:
        await member.add_roles(recruit_role)

@bot.event
async def on_message(message):
    # Ignore messages from bots
    if message.author.bot:
        await bot.process_commands(message)
        return

    # When :Umor: emoji is used, ping Umor in this channel
    if UMOR_EMOJI_ID in message.content:
        try:
            await message.channel.send(f"<@{UMOR_USER_ID}> GET OVER HERE")
        except discord.Forbidden:
            print(f"❌ Missing permissions to send message in #{message.channel.name}")
        except Exception as e:
            print(f"⚠️ Error sending Umor ping: {e}")
    
    # Check if message is in the cooking channel (applies to ALL users)
    if message.channel.id == COOKING_CHANNEL_ID:
        # Check messages with attachments OR embeds (GIF picker)
        if (message.attachments or message.embeds) and CLIP_ENABLED:
            has_pickle_image = False
            image_results = []
            
            try:
                # Show processing indicator
                async with message.channel.typing():
                    has_pickle_image, image_results = await process_message_attachments(message)
            except Exception as e:
                print(f"⚠️ Error processing media: {e}")
                has_pickle_image = False
            
            # DELETE message if pickle is detected in image
            if has_pickle_image:
                try:
                    await message.delete()
                    
                    # Create funny warning message
                    warning_text = (
                        f"🚫 {message.author.mention} tried to sneak in some pickles! "
                        f"Not on my watch! This is a pickle-free zone! 🥒❌"
                    )
                    
                    await message.channel.send(warning_text)
                    
                except discord.Forbidden:
                    print(f"❌ Missing permissions to delete message from {message.author.name}")
                except Exception as e:
                    print(f"⚠️ Error deleting pickle message: {e}")
    
    # Process commands (important for bot commands to still work)
    await bot.process_commands(message)

@tasks.loop(hours=24)
async def update_roles():
    guild = bot.get_guild(GUILD_ID)
    if not guild:
        return

    channel = guild.get_channel(PROMOTION_CHANNEL_ID)
    if not channel:
        return

    now = datetime.now(timezone.utc)
    days_since_start = (now - START_DATE).days

    checked_roles = [guild.get_role(rid) for rid in CHECKED_ROLE_IDS if guild.get_role(rid) is not None]
    promotion_roles = [guild.get_role(rid) for _, rid in ROLE_THRESHOLDS if guild.get_role(rid) is not None]

    for member in guild.members:
        if member.bot:
            continue

        has_any_checked_role = any(role in member.roles for role in checked_roles)

        if not has_any_checked_role:
            roles_to_remove = [r for r in promotion_roles if r in member.roles]
            if roles_to_remove:
                print(f"Attempting to remove these roles from {member.name}: {[r.name for r in roles_to_remove]}")
                try:
                    await member.remove_roles(*roles_to_remove)
                    print(f"Successfully removed promotion roles from {member.name} because they lack required roles.")
                except discord.Forbidden:
                    print(f"❌ Missing permissions to remove roles from {member.name}")
                except Exception as e:
                    print(f"⚠️ Error removing roles from {member.name}: {e}")
            else:
                print(f"No promotion roles to remove from {member.name}")
            continue

        now = datetime.now(timezone.utc)
        member_joined_at = member.joined_at or now
        effective_start = max(member_joined_at, START_DATE)
        days_since_effective_start = (now - effective_start).days

        role_to_assign = None
        for threshold_days, role_id in reversed(ROLE_THRESHOLDS):
            if days_since_effective_start >= threshold_days:
                role_to_assign = guild.get_role(role_id)
                break

        if role_to_assign is None:
            continue

        roles_to_remove = [r for r in promotion_roles if r in member.roles and r != role_to_assign]

        try:
            if roles_to_remove:
                await member.remove_roles(*roles_to_remove)
            if role_to_assign not in member.roles:
                await member.add_roles(role_to_assign)
                await channel.send(f"🎉 {member.mention} has been promoted to **{role_to_assign.name}**! You are less of a scrub now , BOBER KURWA!")
        except discord.Forbidden:
            print(f"❌ Missing permissions to assign {role_to_assign.name} to {member.name}")
        except Exception as e:
            print(f"⚠️ Error assigning {role_to_assign.name} to {member.name}: {e}")

@bot.command()
async def test_promotion(ctx):
    channel = bot.get_channel(PROMOTION_CHANNEL_ID)
    if channel:
        try:
            await channel.send("✅ Promotion message test successful! <:bobrkurva:1319803021974310942>")
            await ctx.send("✅ Sent test message to promotion channel.")
        except Exception as e:
            await ctx.send(f"❌ Failed to send message: {e}")
    else:
        await ctx.send("❌ Could not find the promotion channel.")


@bot.command()
@commands.has_permissions(administrator=True)
async def clip_status(ctx):
    """Check pickle detector status"""
    global pickle_detector

    status = "✅ Loaded in memory" if pickle_detector is not None else "💤 Not loaded (will load on first image)"
    enabled = "✅ Enabled" if CLIP_ENABLED else "❌ Disabled"
    mode = getattr(pickle_detector, "mode", "—") if pickle_detector else "—"
    save_status = "✅ Saving" if SAVE_IMAGES_FOR_TRAINING else "❌ Off"

    await ctx.send(
        f"**Pickle Detection Status:**\n"
        f"• Status: {status}\n"
        f"• Mode: `{mode}`\n"
        f"• Enabled: {enabled}\n"
        f"• Threshold: `{PICKLE_CONFIDENCE_THRESHOLD:.0%}`\n"
        f"• Save for training: {save_status} → `{TRAINING_SAVE_DIR}`\n"
        f"• **BAN pickles** (deletes images with pickles)\n"
        f"• Target Channel: <#{COOKING_CHANNEL_ID}> (ALL users)"
    )


@bot.command()
@commands.has_permissions(administrator=True)
async def clip_unload(ctx):
    """Unload detector to free memory"""
    global pickle_detector

    if pickle_detector is None:
        await ctx.send("ℹ️ Detector is not loaded.")
        return

    unload_pickle_detector()
    await ctx.send("✅ Detector unloaded. It will reload on next image.")


@bot.command()
@commands.has_permissions(administrator=True)
async def clip_preload(ctx):
    """Preload pickle detector"""
    global pickle_detector

    if pickle_detector is not None:
        await ctx.send("ℹ️ Detector is already loaded.")
        return

    await ctx.send("🔄 Loading detector...")
    det = await asyncio.to_thread(get_pickle_detector)
    if det is not None:
        await ctx.send(f"✅ Detector loaded on `{det.device}` ({det.mode})!")
    else:
        await ctx.send("❌ Failed to load detector. Check console for errors.")


def _get_first_media_url_from_message(message) -> str | None:
    """Get first image/video URL from a message (attachments or embeds)."""
    for att in (message.attachments or []):
        if att.content_type and (
            att.content_type.startswith("image/") or att.content_type.startswith("video/")
        ):
            return att.url
    for embed in (message.embeds or []):
        if embed.image and embed.image.url:
            return embed.image.url
        if embed.thumbnail and embed.thumbnail.url:
            return embed.thumbnail.url
        if embed.video and embed.video.url:
            return embed.video.url
    return None


@bot.command()
@commands.has_permissions(administrator=True)
async def pickle_teach(ctx, label: str):
    """Save a labeled image for training. Reply to a message that contains an image, then: !pickle_teach pickle or !pickle_teach non_pickle"""
    if label.lower() not in ("pickle", "non_pickle"):
        await ctx.send("❌ Use `!pickle_teach pickle` or `!pickle_teach non_pickle`")
        return
    ref = ctx.message.reference
    if not ref:
        await ctx.send("❌ Reply to a message that has an image (or video), then run this command.")
        return
    try:
        replied = await ctx.channel.fetch_message(ref.message_id)
    except Exception as e:
        await ctx.send(f"❌ Could not load the replied message: {e}")
        return
    url = _get_first_media_url_from_message(replied)
    if not url:
        await ctx.send("❌ That message has no image or video.")
        return
    await ctx.send("🔄 Downloading and saving for training...")
    frame = await asyncio.to_thread(_get_first_frame_from_url_sync, url)
    if frame is None:
        await ctx.send("❌ Failed to download or decode the image/video.")
        return
    is_pickle = label.lower() == "pickle"
    await asyncio.to_thread(
        _save_frame_for_training_sync,
        frame,
        is_pickle,
        0.0,
        TRAINING_SAVE_DIR,
    )
    await ctx.send(f"✅ Saved as **{label}** to `{TRAINING_SAVE_DIR}`. Run finetune then set `WHATAPICKLE_HEAD_PATH` and restart.")


@bot.command()
async def test_pickle(ctx):
    """Test pickle detection on attached images"""
    if not ctx.message.attachments:
        await ctx.send("❌ Please attach an image to test!")
        return
    
    await ctx.send("🔍 Analyzing image(s)...")
    
    async with ctx.channel.typing():
        has_pickle, results = await process_message_attachments(ctx.message)
    
    if not results:
        await ctx.send("❌ No valid images found.")
        return
    
    response = "**Pickle Detection Results:**\n"
    for filename, has_p, confidence in results:
        emoji = "🚫" if has_p else "✅"
        status = "WOULD BE DELETED" if has_p else "Safe"
        response += f"{emoji} `{filename}`: {confidence:.1%} pickle confidence - **{status}**\n"
    
    response += f"\n**Overall:** {'Pickles detected! Would be DELETED 🚫' if has_pickle else 'No pickles found! Safe to post ✅'}"
    await ctx.send(response)


bot.run(os.environ['DISCORD_TOKEN'])