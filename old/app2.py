import streamlit as st
import matplotlib.pyplot as plt
from art_generator import create_algorithmic_art, spiral_art, noise_art, maze_art, mandelbrot_art, wave_art, checkerboard_art, rings_art

patterns = {
    "Spiral": spiral_art,
    "Noise": noise_art,
    "Maze": maze_art,
    "Mandelbrot": mandelbrot_art,
    "Wave": wave_art,
    "Checkerboard": checkerboard_art,
    "Rings": rings_art
}

st.title("🎨 Algorithmic Art Generator")

pattern_name = st.selectbox("Choose a pattern", list(patterns.keys()))
size = st.slider("Image size", 128, 1024, 512, step=64)
color = st.checkbox("Color Output", value=True)

art_fn = patterns[pattern_name]
img = create_algorithmic_art(size=size, color=color, algorithm=art_fn)

st.image(img, caption=f"{pattern_name} Pattern", use_column_width=True)
