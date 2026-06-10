import genesis as gs
import numpy as np
import pygame
import sys

print("=== Genesis Teleop Marker (LOCKED IN) ===")

# ---------------------------------------------------------
# 1. SETUP GENESIS
# ---------------------------------------------------------
gs.init(backend=gs.vulkan)

scene = gs.Scene(show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())

# Keep fixed=True. Genesis allows us to manually teleport fixed objects.
# If it's False, the physics engine fights our teleport commands.
marker = scene.add_entity(
    morph=gs.morphs.Sphere(
        radius=0.03,
        pos=(0.4, 0.0, 1.0),
        fixed=True
    ),
    surface=gs.surfaces.Rough(
        color=(1.0, 0.0, 0.0)
    )
)

scene.build()

# ---------------------------------------------------------
# 2. SETUP INPUT CONTROLLER (PYGAME)
# ---------------------------------------------------------
pygame.init()
pygame.joystick.init()

# Create window. MUST BE CLICKED ON to capture keyboard.
screen = pygame.display.set_mode((400, 200))
pygame.display.set_caption(">>> CLICK ME TO MOVE <<<")
font = pygame.font.SysFont(None, 24)

# Detect Controller
controller = None
if pygame.joystick.get_count() > 0:
    controller = pygame.joystick.Joystick(0)
    controller.init()
    print(f"✅ Detected controller: {controller.get_name()}")
else:
    print("⚠️ No controller detected. Using Keyboard.")

# ---------------------------------------------------------
# 3. MAIN LOOP
# ---------------------------------------------------------
pos = np.array([0.4, 0.0, 1.0], dtype=np.float32)
speed = 0.015

running = True
while running:
    # --- THE CRITICAL WINDOWS 11 FIX ---
    # You MUST loop through event.get() to empty the OS event queue.
    # Otherwise, Windows marks the window as "Not Responding" and cuts off inputs.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    dx, dy, dz = 0.0, 0.0, 0.0

    # -- KEYBOARD INPUT --
    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        running = False

    # Using Arrow Keys (X/Y) and O/U (Z) to completely avoid Genesis WASD conflicts
    if keys[pygame.K_UP]: dx += speed  # Move Forward
    if keys[pygame.K_DOWN]: dx -= speed  # Move Back
    if keys[pygame.K_LEFT]: dy += speed  # Move Left
    if keys[pygame.K_RIGHT]: dy -= speed  # Move Right
    if keys[pygame.K_o]: dz += speed  # Move Up
    if keys[pygame.K_u]: dz -= speed  # Move Down

    # -- GAMEPAD INPUT --
    if controller:
        axis_y = controller.get_axis(0)
        axis_x = controller.get_axis(1)
        if abs(axis_x) > 0.1: dx -= axis_x * speed
        if abs(axis_y) > 0.1: dy -= axis_y * speed
        if controller.get_button(4): dz -= speed
        if controller.get_button(5): dz += speed

        # -- UPDATE GENESIS --
    if dx != 0 or dy != 0 or dz != 0:
        pos[0] += dx
        pos[1] += dy
        pos[2] += dz
        marker.set_pos(pos)

        # Update Pygame window
    screen.fill((30, 30, 30))
    text1 = font.render("1. CLICK THIS WINDOW FIRST", True, (255, 255, 0))
    text2 = font.render(f"XYZ: {pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}", True, (0, 255, 0))
    text3 = font.render("Arrows=Move | O/U=Up/Down", True, (255, 255, 255))

    screen.blit(text1, (20, 30))
    screen.blit(text2, (20, 80))
    screen.blit(text3, (20, 130))
    pygame.display.flip()

    # Step physics
    scene.step()

# Cleanup
pygame.quit()
sys.exit()