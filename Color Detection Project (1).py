#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
from collections import Counter

def get_color_name(rgb_value, color_dict):
    """
    Get the closest color name from the provided color dictionary.
    """
    distances = {key: np.sqrt(np.sum((np.array(value) - np.array(rgb_value))**2)) for key, value in color_dict.items()}
    return min(distances, key=distances.get)

def get_dominant_colors(image, n_colors=5):
    """
    Get the n most dominant colors in an image.
    """
    # Reshape the image to a 2D array of pixels
    pixels = np.reshape(image, (-1, 3))

    # Get the most common colors
    common_colors = Counter(map(tuple, pixels))
    dominant_colors = [color for color, _ in common_colors.most_common(n_colors)]

    return dominant_colors

# Load the color dictionary
color_dict = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    # Add more colors as needed
}

# Load the image
image = cv2.imread('image.jpg')

# Get the dominant colors in the image
dominant_colors = get_dominant_colors(image, n_colors=5)

# Print the dominant colors and their names
for color in dominant_colors:
    color_name = get_color_name(color, color_dict)
    print(f"Dominant color: {color_name} - RGB: {color}")

