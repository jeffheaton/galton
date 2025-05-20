import os
import random
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import imageio
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.audio.AudioClip import AudioClip

# Requirements:
# pygame==2.5.2
# pymunk==6.6.0
# imageio[ffmpeg]==2.34.1
# moviepy==2.1.2

# CONFIGURATION
RESOLUTION = (512, 768)
SCREEN_WIDTH, SCREEN_HEIGHT = RESOLUTION

FPS = 60
LEVELS = 10
BALL_COUNT = 10  # 200
DROP_INTERVAL = 0.5

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

VELOCITY_THRESHOLD = 50
SPAWN_RANGE = 2
FLOOR_THICKNESS = 5
VISUAL_OFFSET = 10
SUBSTEPS = 3

pygame.mixer.pre_init(44100, -16, 1, 512)
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

space = pymunk.Space()
space.gravity = GRAVITY
space.damping = SPACE_DAMPING
space.iterations = 30

draw_options = pymunk.pygame_util.DrawOptions(screen)

frame_dir = "galton_frames"
os.makedirs(frame_dir, exist_ok=True)
frames = []

collision_times = []


# Generate tick sound
def make_tick_sound():
    frequency = 500  # Hz
    duration = 0.0005  # seconds
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(frequency * 2 * np.pi * t) * 32767
    audio = audio.astype(np.int16)
    sound = pygame.mixer.Sound(audio)
    return sound


tick_sound = make_tick_sound()


def handle_collision(arbiter, space, data):
    ball_shape = (
        arbiter.shapes[0]
        if arbiter.shapes[0].collision_type == 1
        else arbiter.shapes[1]
    )
    ball_velocity = ball_shape.body.velocity

    # Only play sound if the ball has significant vertical velocity
    if abs(ball_velocity.y) > VELOCITY_THRESHOLD:
        collision_times.append(pygame.time.get_ticks() / 1000.0)
        tick_sound.play()

    return True


space.add_collision_handler(1, 0).begin = handle_collision


# HELPERS FOR PEGS & BALLS
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


# LAYOUT PEGS
offset_y = SCREEN_HEIGHT / (LEVELS + 4)
spacing_x = SCREEN_WIDTH / (LEVELS + 1)

max_pegs_even = int((SCREEN_WIDTH - 2 * PEG_RADIUS) / spacing_x) + 1
occupied_width = (max_pegs_even - 1) * spacing_x
left_margin = (SCREEN_WIDTH - occupied_width) / 2

for row in range(LEVELS):
    y = offset_y * (row + 2)
    pegs_in_row = max_pegs_even if row % 2 == 0 else max_pegs_even - 1
    x_start = left_margin if row % 2 == 0 else left_margin + spacing_x / 2

    for i in range(pegs_in_row):
        add_peg(x_start + i * spacing_x, y)

# ADD BINS (Floor setup)
bottom_row = LEVELS - 1
bottom_count = max_pegs_even if bottom_row % 2 == 0 else max_pegs_even - 1
bottom_start = left_margin if bottom_row % 2 == 0 else left_margin + spacing_x / 2
bottom_centers = [bottom_start + i * spacing_x for i in range(bottom_count)]
walls_x = [0.0] + bottom_centers + [SCREEN_WIDTH]

margin = PEG_RADIUS // 2
y_bottom = offset_y * (bottom_row + 2)
wall_top_y = y_bottom - margin

floor_line_y = SCREEN_HEIGHT + FLOOR_THICKNESS - VISUAL_OFFSET
separator_top_y = wall_top_y + VISUAL_OFFSET

for x in walls_x:
    seg = pymunk.Segment(
        space.static_body, (x, floor_line_y), (x, separator_top_y), FLOOR_THICKNESS
    )
    seg.elasticity = SEGMENT_ELASTICITY
    seg.friction = FRICTION
    space.add(seg)

floor = pymunk.Segment(
    space.static_body, (0, floor_line_y), (SCREEN_WIDTH, floor_line_y), FLOOR_THICKNESS
)
floor.elasticity = SEGMENT_ELASTICITY
floor.friction = FRICTION
space.add(floor)

spawn_x = SCREEN_WIDTH / 2
spawn_y = offset_y - BALL_RADIUS
balls_dropped = 0
spawn_timer = 0.0

running = True
last_ball_drop_time = None
while running:
    dt = clock.tick(FPS) / 1000.0
    spawn_timer += dt

    if balls_dropped < BALL_COUNT and spawn_timer >= DROP_INTERVAL:
        add_ball(spawn_x + random.uniform(-SPAWN_RANGE, SPAWN_RANGE), spawn_y)
        balls_dropped += 1
        spawn_timer -= DROP_INTERVAL
        last_ball_drop_time = pygame.time.get_ticks() / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    for _ in range(SUBSTEPS):
        space.step(dt / SUBSTEPS)

    space.debug_draw(draw_options)

    frame_path = os.path.join(frame_dir, f"frame_{pygame.time.get_ticks()}.png")
    pygame.image.save(screen, frame_path)
    frames.append(frame_path)

    pygame.display.flip()

    current_time = pygame.time.get_ticks() / 1000.0
    if last_ball_drop_time and current_time - last_ball_drop_time > 10:
        running = False


# Create the video clip from frames
# Create the video clip from frames
video_clip = ImageSequenceClip(frames, fps=FPS)


def audio_generator(t):
    frequency = 500

    # Ensure t is a numpy array
    scalar_input = False
    if np.isscalar(t):
        t = np.array([t])
        scalar_input = True

    audio = np.zeros_like(t, dtype=np.float32)
    for ct in collision_times:
        mask = np.abs(t - ct) < 0.01
        audio[mask] += np.sin(2 * np.pi * frequency * t[mask])

    audio = audio.reshape(-1, 1)  # Ensure correct shape (N, 1)

    if scalar_input:
        return audio[0]  # Return single sample if input was scalar
    return audio


audio_clip = AudioClip(audio_generator, duration=video_clip.duration, fps=44100)

# Set audio to video using with_audio
video_clip = video_clip.with_audio(audio_clip)

# Export MP4
video_clip.write_videofile(
    "galton_simulation.mp4", fps=FPS, codec="libx264", audio_codec="aac"
)


pygame.quit()
