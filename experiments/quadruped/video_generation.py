import mediapy as media
import imageio.v2 as imageio
import glob
import os

# Path to your frames
frames_path = os.path.join(os.getcwd(), os.path.join("frames_tacosppo5", "frames_tacosppo"))
# Get list of frames in order
frame_files = sorted(glob.glob(os.path.join(frames_path, "frame_*.png")))

frames = []
for f in frame_files:
    img = imageio.imread(f)
    if img.shape[-1] == 4:  # Check if the image has an alpha channel (RGBA)
        img = img[..., :3]  # Keep only the RGB channels
    frames.append(img)
print(f"Number of frames: {len(frames)}")

# Create a video from the image sequence (30 fps)
media.write_video("tacosppo5_playground.mp4", frames, fps=30)
