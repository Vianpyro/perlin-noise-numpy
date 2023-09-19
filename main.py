import pygame
import numpy as np
from settings import *

def generate_gradient_grid(width, height, seed=None):
    """Generate a grid of random gradient vectors with optional seed."""
    if seed is not None:
        np.random.seed(seed)  # Set the seed if provided
    
    grid = np.random.rand(width, height, 2) * 2 - 1
    grid /= np.linalg.norm(grid, axis=2, keepdims=True)
    return grid

def smoothstep(t):
    """Smoothstep function to smooth transitions."""
    return t * t * (3 - 2 * t)

def interpolate(p, grad00, grad01, grad10, grad11):
    """Interpolate between gradient vectors using cubic Hermite interpolation."""
    u = smoothstep(p[0])
    v = smoothstep(p[1])
    
    dot00 = np.dot(grad00, p)
    dot01 = np.dot(grad01, [p[0], p[1] - 1])
    dot10 = np.dot(grad10, [p[0] - 1, p[1]])
    dot11 = np.dot(grad11, [p[0] - 1, p[1] - 1])
    
    blend_x0 = dot00 + u * (dot10 - dot00)
    blend_x1 = dot01 + u * (dot11 - dot01)
    
    return blend_x0 + v * (blend_x1 - blend_x0)

def perlin_noise_chunk(width, height, scale, octaves, seed, x_start, y_start, chunk_size):
    """Generate a chunk of Perlin noise with specified parameters."""
    gradient_grid = generate_gradient_grid(width, height, seed)
    noise_chunk = np.zeros((chunk_size, chunk_size))
    
    frequencies = [2 ** octave for octave in range(octaves)]
    amplitudes = [0.5 ** octave for octave in range(octaves)]

    noise_chunk = np.zeros((chunk_size, chunk_size))

    for x in range(x_start, x_start + chunk_size):
        for y in range(y_start, y_start + chunk_size):
            noise_value = 0
            
            for octave in range(octaves):
                frequency = frequencies[octave]
                amplitude = amplitudes[octave]
                
                cell_x = x / scale * frequency
                cell_y = y / scale * frequency
                
                cell_x0, cell_x1 = int(cell_x), int(cell_x) + 1
                cell_y0, cell_y1 = int(cell_y), int(cell_y) + 1
                
                p = [cell_x - cell_x0, cell_y - cell_y0]
                
                grad00 = gradient_grid[cell_x0 % width, cell_y0 % height]
                grad01 = gradient_grid[cell_x0 % width, cell_y1 % height]
                grad10 = gradient_grid[cell_x1 % width, cell_y0 % height]
                grad11 = gradient_grid[cell_x1 % width, cell_y1 % height]
                
                noise_value += interpolate(p, grad00, grad01, grad10, grad11) * amplitude
            
            noise_chunk[x - x_start, y - y_start] = noise_value
    
    return noise_chunk

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Perlin Noise")

running = True
x_offset, y_offset = 0, 0
clock = pygame.time.Clock()

while running:
    clock.tick(30)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    if y_offset >= HEIGHT:
        pygame.display.set_caption(f"Perlin Noise | FPS: {clock.get_fps()}")
        continue
    
    # Generate a chunk of Perlin noise
    noise_chunk = perlin_noise_chunk(WIDTH, HEIGHT, SCALE, OCTAVES, SEED if SEED is not None else np.random.randint(0, 10000), x_offset, y_offset, CHUNK_SIZE)

    # Scale the noise values using global min and max
    noise_scaled = ((noise_chunk - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN) * 255).astype(np.uint8)
    
    # Create a Pygame surface from the noise chunk
    noise_surface = pygame.surfarray.make_surface(noise_scaled)
    
    # Display the Perlin noise chunk on the screen
    screen.blit(noise_surface, (x_offset, y_offset))
    
    # Update the chunk position for the next iteration
    if x_offset < WIDTH - CHUNK_SIZE:
        x_offset += CHUNK_SIZE
    elif y_offset < HEIGHT:
        x_offset = 0
        y_offset += CHUNK_SIZE

    # Display the Perlin noise surface
    pygame.display.flip()
    pygame.display.set_caption(f"Perlin Noise | FPS: {clock.get_fps()}")

pygame.quit()
