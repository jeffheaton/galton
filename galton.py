import os
import random
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from scipy.io.wavfile import write as wav_write

# ───────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────
RESOLUTION = (512, 768)
SCREEN_WIDTH, SCREEN_HEIGHT = RESOLUTION

FPS = 60
LEVELS = 10
BALL_COUNT = 200
DROP_INTERVAL = 0.25  # seconds between drops

# relative sizes (fractions of SCREEN_WIDTH)
BALL_SIZE = 0.01
PEG_SIZE = 0.005

BALL_RADIUS = int(SCREEN_WIDTH * BALL_SIZE)
PEG_RADIUS = int(SCREEN_WIDTH * PEG_SIZE)
BALL_MASS = 10

GRAVITY = (0, 900)
SPACE_DAMPING = 0.99
FRICTION = 10.0

BALL_ELASTICITY = 0.0
PEG_ELASTICITY = 0.0
SEGMENT_ELASTICITY = 0.0

SPAWN_RANGE = 2

# Tick sound parameters
SAMPLE_RATE = 44100
TICK_SOUND_FREQ = 500
TICK_SOUND_DURATION = 0.01

VELOCITY_THRESHOLD = 50

# ───────────────────────────────────────────────────────────────
# FLOOR & SEPARATOR ENHANCEMENTS
# ───────────────────────────────────────────────────────────────
FLOOR_THICKNESS = 5  # “thickness” of floor/separator capsules
VISUAL_OFFSET = 10  # raise separators & floor up by this many pixels
SUBSTEPS = 3  # physics sub-steps per frame

# ───────────────────────────────────────────────────────────────
# PYGAME + PYMUNK SETUP
# ───────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

space = pymunk.Space()
space.gravity = GRAVITY
space.damping = SPACE_DAMPING
space.iterations = 30  # more solver iterations to reduce tunneling

draw_options = pymunk.pygame_util.DrawOptions(screen)

# prepare directory for frames
frame_dir = "galton_frames"
os.makedirs(frame_dir, exist_ok=True)
frames = []
collision_times = []


# Collision handling
def handle_collision(arbiter, space, data):
    ball_shape = (
        arbiter.shapes[0]
        if arbiter.shapes[0].collision_type == 1
        else arbiter.shapes[1]
    )
    velocity = ball_shape.body.velocity.y
    if abs(velocity) > VELOCITY_THRESHOLD:
        collision_time = (pygame.time.get_ticks() - start_time) / 1000.0
        collision_times.append(collision_time)
        pygame_tick_sound.play()
    return True


space.add_collision_handler(1, 0).begin = handle_collision


# Generate waveform once
def generate_tick_waveform():
    t = np.linspace(
        0, TICK_SOUND_DURATION, int(SAMPLE_RATE * TICK_SOUND_DURATION), False
    )
    waveform = np.sin(2 * np.pi * TICK_SOUND_FREQ * t) * np.hanning(len(t))
    return waveform.astype(np.float32)


tick_waveform = generate_tick_waveform()
pygame_tick_sound = pygame.mixer.Sound((tick_waveform * 32767).astype(np.int16))


# ───────────────────────────────────────────────────────────────
# HELPERS FOR PEGS & BALLS
# ───────────────────────────────────────────────────────────────
def add_peg(x, y):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = (x, y)
    shape = pymunk.Circle(body, PEG_RADIUS)
    shape.elasticity = PEG_ELASTICITY
    shape.friction = FRICTION
    space.add(body, shape)


def add_ball(x, y):
    body = pymunk.Body(BALL_MASS, pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS))
    body.position = (x, y)
    shape = pymunk.Circle(body, BALL_RADIUS)
    shape.elasticity = BALL_ELASTICITY
    shape.friction = FRICTION
    shape.collision_type = 1
    space.add(body, shape)

    # tiny random horizontal impulse so they don't stack perfectly
    # impulse = random.uniform(-BALL_RADIUS * 5, BALL_RADIUS * 5)
    # body.apply_impulse_at_local_point((impulse, 0))


# ───────────────────────────────────────────────────────────────
# LAYOUT PEGS
# ───────────────────────────────────────────────────────────────
offset_y = SCREEN_HEIGHT / (LEVELS + 4)
spacing_x = SCREEN_WIDTH / (LEVELS + 1)

max_pegs_even = int((SCREEN_WIDTH - 2 * PEG_RADIUS) / spacing_x) + 1
occupied_width = (max_pegs_even - 1) * spacing_x
left_margin = (SCREEN_WIDTH - occupied_width) / 2

for row in range(LEVELS):
    y = offset_y * (row + 2)
    if row % 2 == 0:
        pegs_in_row = max_pegs_even
        x_start = left_margin
    else:
        pegs_in_row = max_pegs_even - 1
        x_start = left_margin + spacing_x / 2

    for i in range(pegs_in_row):
        x = x_start + i * spacing_x
        add_peg(x, y)


# ───────────────────────────────────────────────────────────────
# DYNAMIC BIN SEPARATORS (thicker & raised)
# ───────────────────────────────────────────────────────────────
bottom_row = LEVELS - 1
if bottom_row % 2 == 0:
    bottom_count = max_pegs_even
    bottom_start = left_margin
else:
    bottom_count = max_pegs_even - 1
    bottom_start = left_margin + spacing_x / 2

bottom_centers = [bottom_start + i * spacing_x for i in range(bottom_count)]
walls_x = [0.0] + bottom_centers + [SCREEN_WIDTH]

margin = PEG_RADIUS // 2
y_bottom = offset_y * (bottom_row + 2)
wall_top_y = y_bottom - margin

# compute the “line” for the capsule so its interior sits at y=SCREEN_HEIGHT
floor_line_y = SCREEN_HEIGHT + FLOOR_THICKNESS - VISUAL_OFFSET
separator_top_y = wall_top_y + VISUAL_OFFSET

for x in walls_x:
    seg = pymunk.Segment(
        space.static_body, (x, SCREEN_HEIGHT), (x, wall_top_y), FLOOR_THICKNESS
    )
    seg.elasticity = 0.0  # matching SEGMENT_ELASTICITY
    seg.friction = 10.0  # matching FRICTION
    space.add(seg)

floor = pymunk.Segment(
    space.static_body,
    (0, SCREEN_HEIGHT),
    (SCREEN_WIDTH, SCREEN_HEIGHT),
    FLOOR_THICKNESS,
)
floor.elasticity = 0.0  # matching SEGMENT_ELASTICITY
floor.friction = 10.0  # matching FRICTION
space.add(floor)


# ───────────────────────────────────────────────────────────────
# SPAWN POINT & MAIN LOOP
# ───────────────────────────────────────────────────────────────

spawn_x = SCREEN_WIDTH / 2
spawn_y = offset_y - BALL_RADIUS
balls_dropped = 0
spawn_timer = 0.0
start_time = pygame.time.get_ticks()
running = True
last_ball_drop_time = None

running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    spawn_timer += dt

    if balls_dropped < BALL_COUNT and spawn_timer >= DROP_INTERVAL:
        add_ball(spawn_x + random.uniform(-SPAWN_RANGE, SPAWN_RANGE), spawn_y)
        balls_dropped += 1
        spawn_timer -= DROP_INTERVAL
        last_ball_drop_time = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))
    for _ in range(SUBSTEPS):
        space.step(dt / SUBSTEPS)
    space.debug_draw(pymunk.pygame_util.DrawOptions(screen))

    frame_path = os.path.join(frame_dir, f"frame_{pygame.time.get_ticks()}.png")
    pygame.image.save(screen, frame_path)
    frames.append(frame_path)

    pygame.display.flip()

    if last_ball_drop_time and (pygame.time.get_ticks() - last_ball_drop_time > 10000):
        running = False

pygame.quit()

# Video
video_clip = ImageSequenceClip(frames, fps=FPS)
video_clip.write_videofile("galtron_video.mp4", fps=FPS, codec="libx264", audio=False)

# Audio
video_duration = len(frames) / FPS
audio_samples = np.zeros(
    int(SAMPLE_RATE * video_duration) + len(tick_waveform), dtype=np.float32
)
for ct in collision_times:
    idx_start = int(ct * SAMPLE_RATE)
    idx_end = idx_start + len(tick_waveform)
    audio_samples[idx_start:idx_end] += tick_waveform

audio_samples = np.clip(audio_samples, -1.0, 1.0)
wav_audio = (audio_samples * 32767).astype(np.int16)
wav_write("galtron_audio.wav", SAMPLE_RATE, wav_audio)

# Final combined video/audio
audio_clip = AudioFileClip("galtron_audio.wav")
final_clip = video_clip.with_audio(audio_clip).with_duration(video_clip.duration)
final_clip.write_videofile(
    "galtron_combined.mp4", fps=FPS, codec="libx264", audio_codec="aac"
)
