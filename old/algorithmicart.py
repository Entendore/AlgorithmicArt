import numpy as np
import matplotlib.pyplot as plt

def create_algorithmic_art(size=256, color=False, algorithm=None):
    """
    Generate algorithmic image using a user-supplied or built-in pattern generator.
    """
    y, x = np.indices((size, size))
    if algorithm:
        return algorithm(x, y, size, color)

    gradient = (x / size) * 255
    stripes = (((x // 16) % 2) * 255).astype(np.uint8)
    center = size // 2
    radius = np.sqrt((x - center)**2 + (y - center)**2)
    circles = ((np.sin(radius / 8) > 0) * 255).astype(np.uint8)

    if not color:
        img = 0.4 * gradient + 0.3 * stripes + 0.3 * circles
        return np.clip(img, 0, 255).astype(np.uint8)
    else:
        r = 0.5 * gradient + 0.3 * stripes
        g = 0.5 * np.roll(gradient, 10, axis=1) + 0.3 * circles
        b = 0.5 * np.roll(gradient, -10, axis=0) + 0.3 * stripes
        return np.clip(np.stack([r, g, b], axis=2), 0, 255).astype(np.uint8)

def spiral_art(x, y, size, color):
    cx, cy = size // 2, size // 2
    dx, dy = x - cx, y - cy
    angle = np.arctan2(dy, dx)
    radius = np.sqrt(dx**2 + dy**2)
    spiral = np.sin(radius / 4.0 + angle * 6) * 127 + 128
    return colorize(spiral, color)

def noise_art(x, y, size, color):
    np.random.seed(42)
    noise = np.random.rand(size, size) * 255
    return colorize(noise, color)

def maze_art(x, y, size, color):
    cell_size = 16
    grid_x = ((x // cell_size) % 2) * 255
    grid_y = ((y // cell_size) % 2) * 255
    maze = ((grid_x ^ grid_y) > 0) * 255
    return colorize(maze, color)

def mandelbrot_art(x, y, size, color):
    x = x / size * 3.5 - 2.5
    y = y / size * 2.0 - 1.0
    c = x + 1j * y
    z = np.zeros_like(c, dtype=complex)
    div_time = np.full(c.shape, 255, dtype=np.uint8)
    for i in range(255):
        z = z * z + c
        mask = (np.abs(z) > 2) & (div_time == 255)
        div_time[mask] = i
        z[mask] = 2
    return colorize(div_time, color)

def wave_art(x, y, size, color):
    dx = x - size // 2
    dy = y - size // 2
    radius = np.sqrt(dx**2 + dy**2)
    wave = np.sin(radius / 10) * 127 + 128
    return colorize(wave, color)

def checkerboard_art(x, y, size, color):
    tile_size = 32
    checker = ((x // tile_size + y // tile_size) % 2) * 255
    return colorize(checker, color)

def rings_art(x, y, size, color):
    cx, cy = size // 2, size // 2
    radius = np.sqrt((x - cx)**2 + (y - cy)**2)
    rings = ((np.sin(radius / 6) > 0) * 255).astype(np.uint8)
    return colorize(rings, color)


def colorize(base, color):
    """
    Convert a grayscale image to RGB color version if color=True.
    All values clipped to [0, 255] and returned as uint8.
    """
    base = np.asarray(base, dtype=np.float32)
    base = np.clip(base, 0, 255)

    if not color:
        return base.astype(np.uint8)

    r = base
    g = np.roll(base, shift=15, axis=0)
    b = np.roll(base, shift=-15, axis=1)
    img = np.stack([r, g, b], axis=2)
    return np.clip(img, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    size = 512
    color = True

    patterns = {
        "spiral": spiral_art,
        "noise": noise_art,
        "maze": maze_art,
        "mandelbrot": mandelbrot_art,
        "wave": wave_art,
        "checkerboard": checkerboard_art,
        "rings": rings_art
    }

    print("Choose a pattern:")
    for i, name in enumerate(patterns.keys()):
        print(f"  {i+1}. {name}")

    try:
        choice = int(input("Enter number: ")) - 1
        pattern_name = list(patterns.keys())[choice]
        art_fn = patterns[pattern_name]
    except Exception:
        print("Invalid choice. Defaulting to spiral.")
        art_fn = spiral_art
        pattern_name = "spiral"

    img = create_algorithmic_art(size=size, color=color, algorithm=art_fn)

    plt.imshow(img, cmap=None if color else 'gray')
    plt.title(f"Art: {pattern_name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
