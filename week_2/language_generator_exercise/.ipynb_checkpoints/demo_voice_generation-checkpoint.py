#!/usr/bin/env python3
"""
Train a Markov voice generator and generate sample text.

Usage:
    python demo_voice_generation.py --order n

Arguments:
    order   Markov chain order (default: 1). Higher = more coherent text,
            fewer branching choices per step.

The trained model is always saved to my_model.pkl. Run generate_markov_viz.py
afterward to explore it interactively in the browser.
"""
import argparse
import os
import sys
from pathlib import Path

# Add phoenixvoice to path
sys.path.insert(0, str(Path(__file__).parent / "phoenixvoice"))
from phoenixvoice.src.markov_voice_generator import MarkovVoiceGenerator
from phoenixvoice.src.fetch_blog_posts import fetch_blog_posts

MODEL_FILE = "my_model.pkl"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=1,
                        help="Markov chain order (default: 1)")
    order = parser.parse_args().order

    # Configuration
    blog_url = "https://chelseatroy.com"
    blog_file = "chelseatroy_blog_posts.txt"
    max_posts = 50

    # Step 1: Get or fetch blog posts
    if not os.path.exists(blog_file):
        fetch_blog_posts(base_url=blog_url, max_posts=max_posts)

    # Step 2: Train Markov model
    markov_generator = MarkovVoiceGenerator(order=order)
    markov_generator.train(blog_file)

    # Step 3: Save model
    markov_generator.save(MODEL_FILE)

    # Step 4: Generate sample text
    markov_text = markov_generator.generate_text(max_tokens=300)
    print(markov_text)

if __name__ == "__main__":
    main()
