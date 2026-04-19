import os, sys, base64, tempfile
from pathlib import Path

filepath = {{ filepath | tojson }} or {{ path | tojson }}
if not filepath:
    context['output'] = "Error: No image path. Usage: /view_image path/to/image.png"
    sys.exit()

filepath = os.path.expanduser(filepath)
if not os.path.exists(filepath):
    context['output'] = f"Error: File not found: {filepath}"
    sys.exit()

original_path = filepath
ext = Path(filepath).suffix.lower()
supported = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
if ext not in supported:
    print(f"Warning: {ext} may not be supported")

# Compress large images
size_mb = os.path.getsize(filepath) / (1024 * 1024)
compressed_msg = ""
if size_mb > 1.0:
    try:
        from PIL import Image
        img = Image.open(filepath)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        max_dim = 2048
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.Resampling.LANCZOS)
        temp_path = tempfile.mktemp(suffix='.jpg')
        img.save(temp_path, 'JPEG', quality=90)
        new_mb = os.path.getsize(temp_path) / (1024 * 1024)
        compressed_msg = f" (compressed {round(size_mb, 1)}MB -> {round(new_mb, 1)}MB)"
        filepath = temp_path
    except Exception:
        pass

# Read and attach
with open(filepath, 'rb') as f:
    image_bytes = f.read()
b64_data = base64.b64encode(image_bytes).decode('utf-8')

mime_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.gif': 'image/gif', '.bmp': 'image/bmp', '.webp': 'image/webp'}
mime_type = mime_map.get(ext, 'image/png')

state = context.get('state')
if state:
    if not state.attachments:
        state.attachments = []
    state.attachments.append({'type': 'image', 'mime_type': mime_type, 'data': b64_data, 'path': filepath})

# Optional inline render for supported terminals
show_inline = {{ show_inline | tojson }}
render_msg = "Image attached (model can see it)"
if show_inline:
    term = os.environ.get('TERM', '')
    term_program = os.environ.get('TERM_PROGRAM', '')
    is_kitty = os.environ.get('KITTY_WINDOW_ID') or 'kitty' in term.lower()
    is_iterm = term_program == 'iTerm.app' or os.environ.get('ITERM_SESSION_ID')
    try:
        encoded = base64.standard_b64encode(image_bytes).decode('ascii')
        if is_kitty:
            sys.stdout.write(f"\033]1337;File=inline=1:{encoded}\007\n")
            render_msg = "Rendered via Kitty"
        elif is_iterm:
            name = os.path.basename(filepath)
            sys.stdout.write(f"\033]1337;File=name={name};size={len(image_bytes)};inline=1:{encoded}\007\n")
            render_msg = "Rendered via iTerm2"
    except Exception:
        pass

filename = os.path.basename(original_path)
context['output'] = f"""=== IMAGE ATTACHED ===
File: {filename}
Path: {original_path}{compressed_msg}
{render_msg}
======================

The image is now attached. The model will be able to see it."""
