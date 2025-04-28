import mediapy as media
import imageio.v2 as imageio
import glob
import os

# Path to your frames
frame_dir = os.path.join(os.getcwd(), 'videoFrames')
frames_path = os.path.join(frame_dir, os.path.join('frames_ppo_vel', 'frames_ppo_vel'))
# Get list of frames in order
frame_files = sorted(glob.glob(os.path.join(frames_path, "frame_*.png")))

frames = []
i = 0
for f in frame_files:
    img = imageio.imread(f)
    i += 1
    if img.shape[-1] == 4:  # Check if the image has an alpha channel (RGBA)
        img = img[..., :3]  # Keep only the RGB channels
    frames.append(img)
print(f"Number of frames: {len(frames)}")

# Create a video from the image sequence
media.write_video("ppo_velocity.mp4", frames, fps=25)
