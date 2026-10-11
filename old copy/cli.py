import argparse
import matplotlib.pyplot as plt
from art_generator import create_algorithmic_art, spiral_art, noise_art, maze_art, mandelbrot_art, wave_art, checkerboard_art, rings_art

patterns = {
    "spiral": spiral_art,
    "noise": noise_art,
    "maze": maze_art,
    "mandelbrot": mandelbrot_art,
    "wave": wave_art,
    "checkerboard": checkerboard_art,
    "rings": rings_art
}

def main():
    parser = argparse.ArgumentParser(description="Generate algorithmic art.")
    parser.add_argument("--pattern", choices=patterns.keys(), default="spiral", help="Pattern to generate")
    parser.add_argument("--size", type=int, default=512, help="Image size")
    parser.add_argument("--color", action="store_true", help="Use color output")
    args = parser.parse_args()

    art_fn = patterns[args.pattern]
    img = create_algorithmic_art(size=args.size, color=args.color, algorithm=art_fn)

    plt.imshow(img, cmap=None if args.color else 'gray')
    plt.title(f"Art: {args.pattern}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
