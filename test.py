# from PIL import Image

width = 1920
height = 1080

def get_color(w, h):
    r = h / (height - 1)
    g = w / (width - 1)
    b = 0
    ir = int(255.999 * r)
    ig = int(255.999 * g)
    ib = int(255.999 * b)
    return ir, ig, ib

def output_ppm(output_path):
    with open(output_path, 'wb') as f:
        f.write(f'P6\n{width} {height}\n255\n'.encode())
        for h in range(height):
            for w in range(width):
                # color = get_color(x, y)
                r, g, b = get_color(w, h)
                f.write(bytes([r, g, b]))
            # print(f"r{r} g{g} b{b}")

# Example usage:
# image_path = "path/to/your/image.jpg"  # Replace with your image path
output_path = "output.ppm"
output_ppm(output_path)