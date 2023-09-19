import math

# Window settings
WIDTH, HEIGHT = int(2560 * 0.65), int(1440 * 0.65)

# Noise settings
SCALE = 64
OCTAVES = 1
CHUNK_SIZE = math.gcd(WIDTH, HEIGHT)
SEED = 6667

# Height map
GLOBAL_MIN = -1
GLOBAL_MAX = 1

print("Seed used for generation:", SEED)