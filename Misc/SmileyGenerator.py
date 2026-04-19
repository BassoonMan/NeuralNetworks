import numpy as np
from PIL import Image, ImageDraw
import random

class SmileyGenerator:
    def __init__(self, size=28, seed=None):
        self.size = size
        self.rng = random.Random(seed)

    def generate_single(self, smile=True, eyes=True):
        """Generate one randomized smiley/frowny face image.
        
        Returns:
            numpy array of shape (size, size), dtype uint8, values 0-255
            (0 = black background, 255 = white foreground)
        """
        s = self.size
        img = Image.new("L", (s, s), 0)
        draw = ImageDraw.Draw(img)

        # Scale factor relative to 28px baseline
        scale = s / 28.0
        line_w = max(1, int(round(1.0 * scale * self.rng.uniform(0.8, 1.4))))

        # Randomized face center (slight jitter)
        cx = s // 2 + self.rng.uniform(-1.5, 1.5) * scale
        cy = s // 2 + self.rng.uniform(-1.5, 1.5) * scale

        # --- Mouth (arc on an ellipse) ---
        mouth_w = self.rng.uniform(8, 16) * scale   # ellipse width
        mouth_h = self.rng.uniform(4, 10) * scale    # ellipse height (curvature)
        mouth_y_offset = self.rng.uniform(1.5, 5) * scale  # how far below center

        mouth_cy = cy + mouth_y_offset
        mouth_bbox = [
            cx - mouth_w / 2, mouth_cy - mouth_h / 2,
            cx + mouth_w / 2, mouth_cy + mouth_h / 2,
        ]

        if smile:
            # Draw bottom arc of ellipse (0° to 180° in PIL convention)
            draw.arc(mouth_bbox, start=0, end=180, fill=255, width=line_w)
        else:
            # Frown: top arc
            draw.arc(mouth_bbox, start=180, end=360, fill=255, width=line_w)

        # --- Eyes (optional) ---
        if eyes:
            eye_y = cy - self.rng.uniform(2, 5.5) * scale
            eye_spacing = self.rng.uniform(2.5, 5.5) * scale
            eye_r = self.rng.uniform(0.8, 2.0) * scale

            for ex in [cx - eye_spacing, cx + eye_spacing]:
                eye_bbox = [ex - eye_r, eye_y - eye_r, ex + eye_r, eye_y + eye_r]
                draw.ellipse(eye_bbox, fill=255)

        # Optional slight rotation for variety
        angle = self.rng.uniform(-8, 8)
        img = img.rotate(angle, resample=Image.BICUBIC, center=(cx, cy))

        return np.array(img, dtype=np.uint8)
    
    def generate_pair(self, eyes=True):
        """Generate a matched smile/frown pair with identical parameters."""
        s = self.size
        smileI = Image.new("L", (s, s), 0)
        drawS = ImageDraw.Draw(smileI)
        frownI = Image.new("L", (s, s), 0)
        drawF = ImageDraw.Draw(frownI)

        # Scale factor relative to 28px baseline
        scale = s / 28.0
        line_w = max(1, int(round(1.0 * scale * self.rng.uniform(0.8, 1.4))))

        # Randomized face center (slight jitter)
        cx = s // 2 + self.rng.uniform(-1.5, 1.5) * scale
        cy = s // 2 + self.rng.uniform(-1.5, 1.5) * scale

        # --- Mouth (arc on an ellipse) ---
        mouth_w = self.rng.uniform(8, 16) * scale   # ellipse width
        mouth_h = self.rng.uniform(4, 10) * scale    # ellipse height (curvature)
        mouth_y_offset = self.rng.uniform(1.5, 5) * scale  # how far below center

        mouth_cy = cy + mouth_y_offset
        mouth_bbox = [
            cx - mouth_w / 2, mouth_cy - mouth_h / 2,
            cx + mouth_w / 2, mouth_cy + mouth_h / 2,
        ]


        # Draw bottom arc of ellipse (0° to 180° in PIL convention)
        drawS.arc(mouth_bbox, start=0, end=180, fill=255, width=line_w)
        drawF.arc(mouth_bbox, start=180, end=360, fill=255, width=line_w)

        # --- Eyes (optional) ---
        if eyes:
            eye_y = cy - self.rng.uniform(2, 5.5) * scale
            eye_spacing = self.rng.uniform(2.5, 5.5) * scale
            eye_r = self.rng.uniform(0.8, 2.0) * scale

            for ex in [cx - eye_spacing, cx + eye_spacing]:
                eye_bbox = [ex - eye_r, eye_y - eye_r, ex + eye_r, eye_y + eye_r]
                drawS.ellipse(eye_bbox, fill=255)
                drawF.ellipse(eye_bbox, fill=255)

        # Optional slight rotation for variety
        angle = self.rng.uniform(-8, 8)
        smileI = smileI.rotate(angle, resample=Image.BICUBIC, center=(cx, cy))
        frownI = frownI.rotate(angle, resample=Image.BICUBIC, center=(cx, cy))

        return np.array(smileI, dtype=np.uint8), np.array(frownI, dtype=np.uint8)

    def generate_dataset_singles(self, n, smile_ratio=0.5, eyes_ratio=0.7):
        """Generate a labeled dataset.
        
        Returns:
            images: np.array of shape (n, size, size), dtype uint8
            labels: np.array of shape (n,), 0=frown, 1=smile
        """
        images = np.empty((n, self.size, self.size), dtype=np.uint8)
        labels = np.empty(n, dtype=np.int32)

        for i in range(n):
            is_smile = self.rng.random() < smile_ratio
            has_eyes = self.rng.random() < eyes_ratio
            images[i] = self.generate(smile=is_smile, eyes=has_eyes)
            labels[i] = int(is_smile)

        return images, labels
    
    def generate_dataset_pairs(self, n, eyes_ratio=0.7):
        """Generate a labeled dataset.
        
        Returns:
            images: np.array of shape (n, 2, size, size), dtype uint8
        """
        images = np.empty((n, 2, self.size, self.size), dtype=np.uint8)

        for i in range(n):
            has_eyes = self.rng.random() < eyes_ratio
            smile_img, frown_img = self.generate_pair(eyes=has_eyes)
            images[i] = [smile_img, frown_img]

        return images