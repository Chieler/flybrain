"""Render a documented Stage 2b asymmetric waypoint arrival as MP4.

Uses only the system Python standard library and ffmpeg. The controller is the
privileged waypoint solvability witness, not a connectome policy.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import stage2b
import street


OUT = ROOT / "docs" / "media" / "asymmetric_arrival.mp4"
PREVIEW = OUT.with_suffix(".png")
SCENARIO_LABEL = "stage2b-asymmetric-002"
SIZE = 640
FPS = 24
FRAMES = 384
MARGIN = 45
SCALE = (SIZE - 2 * MARGIN) / (2 * street.ARENA_BOUND)
FONT = {
    " ": ("00000",) * 7,
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
}


def px(x):
    return round(MARGIN + (x + street.ARENA_BOUND) * SCALE)


def py(y):
    return round(MARGIN + (street.ARENA_BOUND - y) * SCALE)


def rect(buf, x0, y0, x1, y1, color):
    x0, x1 = max(0, min(x0, x1)), min(SIZE, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(SIZE, max(y0, y1))
    if x1 <= x0 or y1 <= y0:
        return
    row = bytes(color) * (x1 - x0)
    for y in range(y0, y1):
        start = (y * SIZE + x0) * 3
        buf[start:start + len(row)] = row


def circle(buf, cx, cy, radius, color):
    for dy in range(-radius, radius + 1):
        span = math.isqrt(max(0, radius * radius - dy * dy))
        rect(buf, cx - span, cy + dy, cx + span + 1, cy + dy + 1, color)


def line(buf, x0, y0, x1, y1, radius, color):
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    for i in range(n + 1):
        x = round(x0 + (x1 - x0) * i / n)
        y = round(y0 + (y1 - y0) * i / n)
        circle(buf, x, y, radius, color)


def polygon(buf, points, color):
    """Fill a small convex polygon in screen coordinates."""
    top = max(0, math.floor(min(y for _, y in points)))
    bottom = min(SIZE, math.ceil(max(y for _, y in points)))
    for y in range(top, bottom):
        scan_y = y + 0.5
        crossings = []
        for a, b in zip(points, points[1:] + points[:1]):
            if (a[1] <= scan_y < b[1]) or (b[1] <= scan_y < a[1]):
                t = (scan_y - a[1]) / (b[1] - a[1])
                crossings.append(a[0] + t * (b[0] - a[0]))
        if len(crossings) >= 2:
            rect(buf, math.floor(min(crossings)), y,
                 math.ceil(max(crossings)) + 1, y + 1, color)


def car(buf, x, y, heading):
    """Top-down sedan, rotated with the recorded vehicle heading."""
    cx, cy = px(x), py(y)
    cosine, sine = math.cos(heading), math.sin(heading)

    def shape(points, color):
        transformed = [
            (cx + forward * cosine - side * sine,
             cy - forward * sine - side * cosine)
            for forward, side in points
        ]
        polygon(buf, transformed, color)

    # Four dark tires sit outside the yellow body.
    for front in (-10, 9):
        for side in (-1, 1):
            edge = side * 10
            shape([(front - 4, edge - 2), (front + 4, edge - 2),
                   (front + 4, edge + 2), (front - 4, edge + 2)],
                  (13, 19, 26))
    shape([(-18, -7), (-14, -9), (12, -9), (18, -6),
           (18, 6), (12, 9), (-14, 9), (-18, 7)], (13, 19, 26))
    shape([(-16, -6), (-13, -7), (11, -7), (16, -5),
           (16, 5), (11, 7), (-13, 7), (-16, 6)], (248, 182, 67))
    # Front windscreen, rear window, and the lighter roof make direction clear.
    shape([(-6, -6), (5, -6), (7, -5), (7, 5), (5, 6), (-6, 6)],
          (239, 207, 111))
    shape([(6, -5), (9, -5), (9, 5), (6, 5)], (63, 124, 157))
    shape([(-11, -5), (-8, -5), (-8, 5), (-11, 5)], (53, 94, 118))
    for side in (-5, 5):
        shape([(14, side - 2), (17, side - 2),
               (17, side + 1), (14, side + 1)], (255, 247, 187))


def label(buf, x, y, message, color, scale=2):
    for char in message:
        for row, bits in enumerate(FONT[char]):
            for col, bit in enumerate(bits):
                if bit == "1":
                    rect(buf, x + col * scale, y + row * scale,
                         x + (col + 1) * scale, y + (row + 1) * scale, color)
        x += 6 * scale


def main():
    data = json.loads((ROOT / "runs/stage2b/gate_split.json").read_text())
    row = next(s for s in data if s["label"] == SCENARIO_LABEL)
    start = row["start"]
    scenario = street.StreetScenario(
        row["layout"],
        street.StreetCarState(start["x"], start["y"], start["heading"], start["speed"]),
        row["target_x"], row["target_y"], row["target_radius"], row["timeout"], row["label"],
    )
    layout = street.initial_layouts()[scenario.layout]
    controller = stage2b.WaypointController(scenario, layout)
    result = street.run_street_episode(scenario, layout, controller, record=True)
    assert result.outcome == "arrival", result.outcome
    trajectory = result.trajectory

    base = bytearray(bytes((13, 18, 28)) * SIZE * SIZE)
    rect(base, MARGIN, MARGIN, SIZE - MARGIN, SIZE - MARGIN, (55, 65, 76))
    for building in layout.buildings:
        rect(base, px(building.min_x), py(building.max_y),
             px(building.max_x), py(building.min_y), (31, 39, 53))
    rect(base, 0, 0, SIZE, 30, (13, 18, 28))
    rect(base, 0, SIZE - 32, SIZE, SIZE, (13, 18, 28))
    label(base, 19, 8, "ASYMMETRIC ARRIVAL", (228, 238, 248))
    label(base, 19, SIZE - 25, "WAYPOINT BASELINE", (167, 187, 207))
    tx, ty = px(scenario.target_x), py(scenario.target_y)
    circle(base, tx, ty, round(scenario.target_radius * SCALE), (29, 113, 91))
    circle(base, tx, ty, 5, (94, 245, 167))
    sx, sy = px(scenario.start.x), py(scenario.start.y)
    circle(base, sx, sy, 5, (106, 181, 252))

    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{SIZE}x{SIZE}",
        "-framerate", str(FPS), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(OUT),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for frame in range(FRAMES):
            progress = min(frame / (FRAMES - 24), 1.0)
            index = min(round(progress * (len(trajectory) - 1)), len(trajectory) - 1)
            buf = base.copy()
            sampled = trajectory[:index + 1:8]
            if sampled[-1] != trajectory[index]:
                sampled.append(trajectory[index])
            for a, b in zip(sampled, sampled[1:]):
                line(buf, px(a[0]), py(a[1]), px(b[0]), py(b[1]), 2, (83, 198, 248))
            x, y, heading = trajectory[index][:3]
            car(buf, x, y, heading)
            if progress == 1.0:
                label(buf, SIZE - 118, SIZE - 25, "ARRIVED", (94, 245, 167))
            process.stdin.write(buf)
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    subprocess.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", "00:00:08", "-i", str(OUT), "-frames:v", "1", str(PREVIEW),
    ], check=True)
    print(f"{OUT} — {result.outcome}, {result.elapsed_time:.2f} simulated seconds")


if __name__ == "__main__":
    main()
