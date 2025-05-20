import os
import random

import pygame
import pymunk
import pymunk.pygame_util
import imageio

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
BALL_MASS = 1

GRAVITY = (0, 900)
SPACE_DAMPING = 0.99
FRICTION = 0.6

BALL_ELASTICITY = 0.0
PEG_ELASTICITY = 0.0
SEGMENT_ELASTICITY = 0.0

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


# collision handler stub (no sound)
def handle_collision(arbiter, space, data):
    return True


space.add_collision_handler(1, 0).begin = handle_collision


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
    impulse = random.uniform(-BALL_RADIUS * 5, BALL_RADIUS * 5)
    body.apply_impulse_at_local_point((impulse, 0))


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
        space.static_body, (x, floor_line_y), (x, separator_top_y), FLOOR_THICKNESS
    )
    seg.elasticity = SEGMENT_ELASTICITY
    seg.friction = FRICTION
    space.add(seg)

# thick floor
floor = pymunk.Segment(
    space.static_body, (0, floor_line_y), (SCREEN_WIDTH, floor_line_y), FLOOR_THICKNESS
)
floor.elasticity = SEGMENT_ELASTICITY
floor.friction = FRICTION
space.add(floor)


# ───────────────────────────────────────────────────────────────
# SPAWN POINT & MAIN LOOP
# ───────────────────────────────────────────────────────────────
spawn_x = SCREEN_WIDTH / 2
spawn_y = offset_y - BALL_RADIUS
balls_dropped = 0
spawn_timer = 0.0

running = True
while running:
    dt = clock.tick(FPS) / 1000.0
    spawn_timer += dt

    if balls_dropped < BALL_COUNT and spawn_timer >= DROP_INTERVAL:
        add_ball(spawn_x, spawn_y)
        balls_dropped += 1
        spawn_timer -= DROP_INTERVAL

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))

    # physics with substeps to avoid tunneling
    for _ in range(SUBSTEPS):
        space.step(dt / SUBSTEPS)

    space.debug_draw(draw_options)

    # capture frame for video
    frame_path = os.path.join(frame_dir, f"frame_{pygame.time.get_ticks()}.png")
    pygame.image.save(screen, frame_path)
    frames.append(frame_path)

    pygame.display.flip()

    # stop when all balls are dropped and nearly still
    if balls_dropped == BALL_COUNT and all(
        abs(b.velocity.y) < 0.1 and abs(b.velocity.x) < 0.1
        for b in space.bodies
        if b.body_type == pymunk.Body.DYNAMIC
    ):
        running = False

pygame.quit()

# ───────────────────────────────────────────────────────────────
# COMPILE VIDEO & CLEANUP
# ───────────────────────────────────────────────────────────────
with imageio.get_writer("galton.mp4", fps=FPS) as writer:
    for f in frames:
        writer.append_data(imageio.imread(f))

for f in frames:
    os.remove(f)
os.rmdir(frame_dir)
