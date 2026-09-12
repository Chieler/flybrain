"""Phase 5 viewer: replay a Stage 1 episode in a pygame window.

The controller drives a recorded episode first (`run_episode(record=True)`), then
this module only animates the trajectory — rendering is decoupled from the
~13 ms/step neural compute, so a real-graph run plays back smoothly.

World frame: +x east, +y north, CCW angles. Screen conversion (y-flip) lives
HERE and nowhere else, per the simulation contract.

Default controller is the LABELED conventional compass baseline, so the viewer
runs with no connectome download. `--controller neural` needs `--data` and a
calibration `--checkpoint`.
"""

from __future__ import annotations

import argparse
import math

import pygame

from simulation import (
    ARENA_BOUND, CAR_RADIUS, TARGET_RADIUS, run_episode,
)
import evaluate as ev

WIN = 640          # square window, px
MARGIN = 24        # px border around the arena
BG = (18, 18, 22)
ARENA = (60, 60, 70)
TARGET = (70, 200, 120)
CAR = (240, 200, 80)
PATH = (90, 110, 160)
BUILDING = (48, 50, 58)
ROAD = (82, 82, 90)

_SCALE = (WIN - 2 * MARGIN) / (2 * ARENA_BOUND)


def _to_screen(wx: float, wy: float) -> tuple[int, int]:
    sx = MARGIN + (wx + ARENA_BOUND) * _SCALE
    sy = WIN - (MARGIN + (wy + ARENA_BOUND) * _SCALE)  # flip: north is up
    return int(sx), int(sy)


def _rect_to_screen(rect):
    left, bottom = _to_screen(rect.min_x, rect.min_y)
    right, top = _to_screen(rect.max_x, rect.max_y)
    return pygame.Rect(left, top, right - left, bottom - top)


def _car_triangle(x: float, y: float, heading: float) -> list[tuple[int, int]]:
    # Nose + two tail corners, sized by CAR_RADIUS in world units.
    r = CAR_RADIUS * 1.6
    pts = [(r, 0.0), (-r * 0.7, r * 0.6), (-r * 0.7, -r * 0.6)]
    c, s = math.cos(heading), math.sin(heading)
    return [_to_screen(x + px * c - py * s, y + px * s + py * c) for px, py in pts]


def _make_controller(kind: str, data: str | None, checkpoint: str | None):
    if kind == "baseline":
        return ev.conventional_baseline(), None
    # neural: load the real brain + the calibrated adapter from the checkpoint.
    from brain import Adapter, Brain, NeuralController
    cp = ev.load_checkpoint(checkpoint)
    brain = Brain.load(data or cp["data_dir"], cp["params"])
    ctrl = NeuralController(brain, Adapter(gain=cp["gain"], bias=cp["bias"]))
    return ctrl, ctrl.reset


def replay(scenario, controller, reset=None, fps: int = 60) -> None:
    if reset is not None:
        reset()
    result = run_episode(scenario, controller, record=True)
    traj = result.trajectory or []

    pygame.init()
    screen = pygame.display.set_mode((WIN, WIN))
    pygame.display.set_caption(f"flybrain — {result.outcome} @ {result.elapsed_time:.1f}s")
    clock = pygame.time.Clock()
    target = _to_screen(scenario.target_x, scenario.target_y)
    tgt_r = int(TARGET_RADIUS * _SCALE)

    i = 0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        screen.fill(BG)
        pygame.draw.rect(screen, ARENA, (MARGIN, MARGIN, WIN - 2 * MARGIN, WIN - 2 * MARGIN), 1)
        pygame.draw.circle(screen, TARGET, target, tgt_r, 2)
        if i > 1:
            pygame.draw.lines(screen, PATH, False,
                              [_to_screen(x, y) for x, y, _ in traj[:i]], 2)
        if traj:
            x, y, h = traj[min(i, len(traj) - 1)]
            pygame.draw.polygon(screen, CAR, _car_triangle(x, y, h))

        pygame.display.flip()
        clock.tick(fps)
        if i < len(traj) - 1:
            i += 1  # advance until the last frame, then hold until closed

    pygame.quit()


def _make_stage2_controller(kind: str, avoidance_gain: float, brake_distance: float,
                            data: str | None, stage1_checkpoint: str | None):
    import evaluate_stage2 as ev2
    from brain import Stage2Adapter, Stage2NeuralController
    if kind == "baseline":
        adapter = Stage2Adapter(0.0, 0.0, avoidance_gain, brake_distance)
        return ev2.direct_compass_controller(adapter), None
    cp = ev.load_checkpoint(stage1_checkpoint)
    from brain import Brain
    brain = Brain.load(data or cp["data_dir"], cp["params"])
    adapter = Stage2Adapter(cp["gain"], cp["bias"], avoidance_gain, brake_distance)
    ctrl = Stage2NeuralController(brain, adapter)
    return ctrl, ctrl.reset


def replay_stage2(args) -> None:
    import json
    from pathlib import Path

    import evaluate_stage2 as ev2
    from street import initial_layouts, run_street_episode

    layouts = initial_layouts()
    scenario = ev2.load_scenarios(args.scenarios)[args.index]
    layout = layouts[scenario.layout]

    avoidance_gain, brake_distance = 0.5, 4.0
    if Path(args.stage2_checkpoint).exists():
        cp = json.loads(Path(args.stage2_checkpoint).read_text())
        avoidance_gain, brake_distance = cp["avoidance_gain"], cp["brake_distance"]

    controller, reset = _make_stage2_controller(
        args.controller, avoidance_gain, brake_distance, args.data, args.checkpoint)
    if reset is not None:
        reset()
    result = run_street_episode(scenario, layout, controller, record=True)
    traj = result.trajectory or []

    pygame.init()
    screen = pygame.display.set_mode((WIN, WIN))
    pygame.display.set_caption(
        f"flybrain stage2 — {result.outcome} @ {result.elapsed_time:.1f}s")
    clock = pygame.time.Clock()
    target = _to_screen(scenario.target_x, scenario.target_y)
    from simulation import TARGET_RADIUS as _TR
    tgt_r = int(scenario.target_radius * _SCALE) or int(_TR * _SCALE)

    i, frames, running = 0, 0, True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        screen.fill(BG)
        pygame.draw.rect(screen, ROAD, (MARGIN, MARGIN, WIN - 2 * MARGIN, WIN - 2 * MARGIN))
        for building in layout.buildings:
            pygame.draw.rect(screen, BUILDING, _rect_to_screen(building))
        pygame.draw.circle(screen, TARGET, target, tgt_r, 2)
        if i > 1:
            pygame.draw.lines(screen, PATH, False,
                              [_to_screen(x, y) for x, y, _, _ in traj[:i]], 2)
        if traj:
            x, y, h, _ = traj[min(i, len(traj) - 1)]
            pygame.draw.polygon(screen, CAR, _car_triangle(x, y, h))

        pygame.display.flip()
        clock.tick(args.fps)
        if i < len(traj) - 1:
            i += 1
        frames += 1
        if args.frames is not None and frames >= args.frames:
            running = False

    pygame.quit()


def _pick_scenario(args):
    if args.scenarios:
        return ev.load_scenarios(args.scenarios)[args.index]
    return ev.generate_scenarios(args.index + 1, args.seed)[args.index]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Replay a Stage 1 episode (pygame).")
    p.add_argument("--controller", choices=("baseline", "neural"), default="baseline")
    p.add_argument("--scenarios", help="scenarios.json; omit to generate one by --seed.")
    p.add_argument("--index", type=int, default=0, help="Scenario index to replay.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--data", help="Prepared connectome dir (neural controller).")
    p.add_argument("--checkpoint", default="checkpoint.json")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--stage", choices=(1, 2), type=int, default=1)
    p.add_argument("--stage2-checkpoint", default="runs/stage2/checkpoint.json")
    p.add_argument("--frames", type=int, help="Stop after this many replay frames (smoke tests).")
    args = p.parse_args()

    if args.stage == 2:
        replay_stage2(args)
    else:
        if args.controller == "neural" and not (args.data or args.checkpoint):
            p.error("--controller neural needs --data or a --checkpoint carrying data_dir")
        controller, reset = _make_controller(args.controller, args.data, args.checkpoint)
        replay(_pick_scenario(args), controller, reset, args.fps)
