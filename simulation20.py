from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from collections import deque

import pygame

CFG = {
    "grid_w": 120,
    "grid_h": 80,
    "fullscreen": False,
    "panel_w": 360,
    "cell_gap": 0,
    "initial_plants": 1800,
    "initial_predators": 120,
    "plant_spread_prob": 0.060,
    "plant_spawn_prob": 0.0015,
    "pred_initial_energy": 22.0,
    "metabolic_cost": 0.20,
    "move_cost": 0.80,
    "eat_gain": 7.0,
    "sense_radius": 6,
    "repro_threshold": 30.0,
    "repro_cost": 10.0,
    "baby_energy": 15.0,
    "three_step_eat": True,
    "gens_per_sec": 12,
    "history_len": 300,
}

DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

COL_BG         = (8,  10,  14)
COL_GRID_BG    = (11, 14,  20)
COL_PANEL_BG   = (10, 12,  17)
COL_PLANT      = (52, 211, 153)
COL_PREDATOR   = (99, 179, 255)
COL_BORDER     = (30, 36,  50)
COL_TEXT       = (180, 190, 210)
COL_TEXT_DIM   = (80,  95, 120)
COL_TEXT_HEAD  = (220, 230, 245)
COL_ACCENT_G   = (52,  211, 153)
COL_ACCENT_B   = (99,  179, 255)
COL_RUN        = (52,  211, 153)
COL_PAUSE      = (255, 180,  60)
COL_SEPARATOR  = (25,  32,  46)


@dataclass
class Predator:
    x: int
    y: int
    energy: float
    eat_wait: int = 0
    eat_target: tuple[int, int] | None = None


def in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def neighbors4(x: int, y: int, w: int, h: int):
    for dx, dy in DIRS4:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield nx, ny


def bfs_find_nearest_plant(
    start: tuple[int, int],
    plants: set[tuple[int, int]],
    w: int,
    h: int,
    radius: int,
) -> tuple[int, int] | None:
    sx, sy = start
    q = deque([(sx, sy, 0)])
    seen = {(sx, sy)}
    while q:
        x, y, d = q.popleft()
        if d > 0 and (x, y) in plants:
            return (x, y)
        if d == radius:
            continue
        for nx, ny in neighbors4(x, y, w, h):
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny, d + 1))
    return None


def sign(v: int) -> int:
    return (v > 0) - (v < 0)


def draw_rounded_rect(surface, color, rect, radius=8, border=0, border_color=None):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border and border_color:
        pygame.draw.rect(surface, border_color, rect, width=border, border_radius=radius)


def draw_graph(surface: pygame.Surface, rect: pygame.Rect, hist_green, hist_blue, font_small):
    draw_rounded_rect(surface, (14, 18, 28), rect, radius=8, border=1, border_color=COL_BORDER)

    if len(hist_green) < 2:
        return

    pad = 14
    gx0 = rect.x + pad
    gy0 = rect.y + pad + 22
    gw  = rect.w - 2 * pad
    gh  = rect.h - 2 * pad - 22

    max_val = max(max(hist_green), max(hist_blue), 1)
    n = len(hist_green)

    def to_xy(i, val):
        x = gx0 + int(gw * (i / max(n - 1, 1)))
        y = gy0 + int(gh * (1.0 - val / max_val))
        return x, y

    pts_g = [to_xy(i, v) for i, v in enumerate(hist_green)]
    pts_b = [to_xy(i, v) for i, v in enumerate(hist_blue)]

    pygame.draw.lines(surface, COL_ACCENT_G, False, pts_g, 2)
    pygame.draw.lines(surface, COL_ACCENT_B, False, pts_b, 2)

    dot_x_g = rect.x + pad
    dot_x_b = rect.x + pad + 90
    dot_y   = rect.y + pad + 7

    pygame.draw.circle(surface, COL_ACCENT_G, (dot_x_g + 5, dot_y), 4)
    pygame.draw.circle(surface, COL_ACCENT_B, (dot_x_b + 5, dot_y), 4)

    lg = font_small.render("Plants", True, COL_TEXT_DIM)
    lb = font_small.render("Predators", True, COL_TEXT_DIM)
    surface.blit(lg, (dot_x_g + 13, dot_y - 6))
    surface.blit(lb, (dot_x_b + 13, dot_y - 6))


def draw_stat_row(surface, font, label, value, keys_hint, x, y, w):
    lbl = font.render(label, True, COL_TEXT_DIM)
    val = font.render(value, True, COL_TEXT_HEAD)
    hint = font.render(keys_hint, True, (55, 68, 90))
    surface.blit(lbl, (x, y))
    surface.blit(val, (x + w - hint.get_width() - val.get_width() - 8, y))
    surface.blit(hint, (x + w - hint.get_width(), y))


def draw_divider(surface, x, y, w):
    pygame.draw.line(surface, COL_SEPARATOR, (x, y), (x + w, y), 1)


def main():
    pygame.init()
    pygame.display.set_caption("Ecosystem Simulation")

    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h

    if CFG["fullscreen"]:
        screen = pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode(
            (min(1400, screen_w), min(900, screen_h)), pygame.RESIZABLE
        )

    clock = pygame.time.Clock()
    font       = pygame.font.SysFont("consolas", 14)
    font_small = pygame.font.SysFont("consolas", 13)
    font_big   = pygame.font.SysFont("consolas", 18, bold=True)
    font_title = pygame.font.SysFont("consolas", 22, bold=True)

    def compute_layout():
        sw, sh = screen.get_size()
        panel_w = min(CFG["panel_w"], max(260, sw // 4))
        sim_w   = sw - panel_w

        cell_size = min(sim_w // CFG["grid_w"], sh // CFG["grid_h"])
        cell_size = max(4, cell_size)

        grid_px_w = cell_size * CFG["grid_w"]
        grid_px_h = cell_size * CFG["grid_h"]

        ox = (sim_w - grid_px_w) // 2
        oy = (sh - grid_px_h) // 2

        grid_rect  = pygame.Rect(ox, oy, grid_px_w, grid_px_h)
        panel_rect = pygame.Rect(sim_w, 0, panel_w, sh)
        return cell_size, grid_rect, panel_rect

    cell_size, grid_rect, panel_rect = compute_layout()

    w, h = CFG["grid_w"], CFG["grid_h"]
    plants: set[tuple[int, int]] = set()
    predators: list[Predator] = []
    generation = 0

    hist_green: list[int] = []
    hist_blue:  list[int] = []

    paused = False
    accumulator_s = 0.0

    def reset_world():
        nonlocal plants, predators, generation, hist_green, hist_blue, paused, accumulator_s
        plants        = set()
        predators     = []
        generation    = 0
        hist_green    = []
        hist_blue     = []
        paused        = False
        accumulator_s = 0.0

        while len(plants) < CFG["initial_plants"]:
            plants.add((random.randrange(w), random.randrange(h)))

        occupied = set(plants)
        while len(predators) < CFG["initial_predators"]:
            x, y = random.randrange(w), random.randrange(h)
            if (x, y) in occupied:
                continue
            occupied.add((x, y))
            predators.append(Predator(x, y, CFG["pred_initial_energy"]))

    reset_world()

    def cell_at_mouse(mx, my):
        if not grid_rect.collidepoint(mx, my):
            return None
        gx = (mx - grid_rect.x) // cell_size
        gy = (my - grid_rect.y) // cell_size
        if 0 <= gx < w and 0 <= gy < h:
            return int(gx), int(gy)
        return None

    def world_step():
        nonlocal plants, predators, generation, hist_green, hist_blue

        generation += 1

        pred_pos = {(p.x, p.y) for p in predators}

        if plants:
            proposals = []
            for (px, py) in plants:
                if random.random() < CFG["plant_spread_prob"]:
                    neigh = list(neighbors4(px, py, w, h))
                    random.shuffle(neigh)
                    for nx, ny in neigh:
                        if (nx, ny) not in plants and (nx, ny) not in pred_pos:
                            proposals.append((nx, ny))
                            break
            random.shuffle(proposals)
            for nx, ny in proposals:
                if (nx, ny) not in plants and (nx, ny) not in pred_pos:
                    plants.add((nx, ny))

        if CFG["plant_spawn_prob"] > 0:
            trials = int(w * h * CFG["plant_spawn_prob"])
            for _ in range(trials):
                rx, ry = random.randrange(w), random.randrange(h)
                if (rx, ry) not in plants and (rx, ry) not in pred_pos:
                    if random.random() < 0.65:
                        plants.add((rx, ry))

        new_predators: list[Predator] = []
        pred_pos  = {(p.x, p.y) for p in predators}
        plants_set = plants

        for p in predators:
            p.energy -= CFG["metabolic_cost"]

            if CFG["three_step_eat"] and p.eat_wait > 0:
                p.eat_wait -= 1
                if p.eat_wait == 0 and p.eat_target is not None:
                    tx, ty = p.eat_target
                    if (tx, ty) in plants_set and abs(tx - p.x) + abs(ty - p.y) == 1:
                        if (tx, ty) not in pred_pos:
                            pred_pos.discard((p.x, p.y))
                            p.x, p.y = tx, ty
                            pred_pos.add((p.x, p.y))
                        if (tx, ty) in plants_set:
                            plants_set.remove((tx, ty))
                            p.energy += CFG["eat_gain"]
                    p.eat_target = None

        pred_pos = {(p.x, p.y) for p in predators}

        move_proposals: dict[tuple[int, int], list[int]] = {}

        for idx, p in enumerate(predators):
            if p.energy <= 0:
                continue
            if CFG["three_step_eat"] and p.eat_wait > 0:
                continue

            target = bfs_find_nearest_plant(
                (p.x, p.y), plants_set, w, h, CFG["sense_radius"]
            )

            def propose(nx, ny):
                if (nx, ny) in plants_set:
                    return False
                if (nx, ny) in pred_pos:
                    return False
                move_proposals.setdefault((nx, ny), []).append(idx)
                return True

            moved = False
            if target is not None:
                tx, ty = target
                dx = sign(tx - p.x)
                dy = sign(ty - p.y)
                candidates = []
                if dx != 0:
                    candidates.append((p.x + dx, p.y))
                if dy != 0:
                    candidates.append((p.x, p.y + dy))
                random.shuffle(candidates)
                for nx, ny in candidates:
                    if in_bounds(nx, ny, w, h) and propose(nx, ny):
                        moved = True
                        break

            if not moved:
                neigh = list(neighbors4(p.x, p.y, w, h))
                random.shuffle(neigh)
                for nx, ny in neigh:
                    if propose(nx, ny):
                        break

        winners: dict[int, tuple[int, int]] = {}
        for dest, idxs in move_proposals.items():
            if len(idxs) == 1:
                winners[idxs[0]] = dest
            else:
                winners[random.choice(idxs)] = dest

        pred_pos = {(p.x, p.y) for p in predators}
        for idx, dest in winners.items():
            p = predators[idx]
            if p.energy <= 0:
                continue
            nx, ny = dest
            if (nx, ny) in pred_pos:
                continue
            pred_pos.discard((p.x, p.y))
            p.x, p.y = nx, ny
            pred_pos.add((p.x, p.y))
            p.energy -= CFG["move_cost"]

        if CFG["three_step_eat"]:
            pred_pos = {(p.x, p.y) for p in predators}
            order = list(range(len(predators)))
            random.shuffle(order)
            for idx in order:
                p = predators[idx]
                if p.energy <= 0 or p.eat_wait > 0:
                    continue
                adj_plants = [
                    (nx, ny)
                    for nx, ny in neighbors4(p.x, p.y, w, h)
                    if (nx, ny) in plants_set
                ]
                if adj_plants:
                    p.eat_target = random.choice(adj_plants)
                    p.eat_wait   = 2

        pred_pos = {(p.x, p.y) for p in predators}
        for p in predators:
            if p.energy <= 0:
                continue
            if p.energy >= CFG["repro_threshold"]:
                neigh = list(neighbors4(p.x, p.y, w, h))
                random.shuffle(neigh)
                for nx, ny in neigh:
                    if (nx, ny) not in plants_set and (nx, ny) not in pred_pos:
                        p.energy -= CFG["repro_cost"]
                        baby = Predator(nx, ny, CFG["baby_energy"])
                        new_predators.append(baby)
                        pred_pos.add((nx, ny))
                        break

        predators = [p for p in predators if p.energy > 0]
        predators.extend(new_predators)

        hist_green.append(len(plants_set))
        hist_blue.append(len(predators))
        if len(hist_green) > CFG["history_len"]:
            hist_green = hist_green[-CFG["history_len"]:]
            hist_blue  = hist_blue[-CFG["history_len"]:]

        plants = plants_set

    def clamp_cfg(key, lo, hi):
        CFG[key] = max(lo, min(hi, CFG[key]))

    running = True
    while running:
        dt_s = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE and not CFG["fullscreen"]:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                cell_size, grid_rect, panel_rect = compute_layout()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused        = not paused
                    accumulator_s = 0.0
                elif event.key == pygame.K_n:
                    if paused:
                        world_step()
                elif event.key == pygame.K_x:
                    reset_world()
                elif event.key == pygame.K_f:
                    CFG["fullscreen"] = not CFG["fullscreen"]
                    if CFG["fullscreen"]:
                        screen = pygame.display.set_mode(
                            (info.current_w, info.current_h), pygame.FULLSCREEN
                        )
                    else:
                        screen = pygame.display.set_mode(
                            (min(1400, info.current_w), min(900, info.current_h)),
                            pygame.RESIZABLE,
                        )
                    cell_size, grid_rect, panel_rect = compute_layout()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                cell = cell_at_mouse(mx, my)
                if cell:
                    gx, gy = cell
                    mods  = pygame.key.get_mods()
                    shift = bool(mods & pygame.KMOD_SHIFT)
                    if shift:
                        plants.discard((gx, gy))
                        for i in range(len(predators) - 1, -1, -1):
                            if predators[i].x == gx and predators[i].y == gy:
                                predators.pop(i)
                    else:
                        if event.button == 1:
                            if not any(p.x == gx and p.y == gy for p in predators):
                                plants.add((gx, gy))
                        elif event.button == 3:
                            if (gx, gy) not in plants and not any(
                                p.x == gx and p.y == gy for p in predators
                            ):
                                predators.append(
                                    Predator(gx, gy, CFG["pred_initial_energy"])
                                )

        keys = pygame.key.get_pressed()

        if keys[pygame.K_g]: CFG["plant_spread_prob"] += 0.002
        if keys[pygame.K_h]: CFG["plant_spread_prob"] -= 0.002
        if keys[pygame.K_e]: CFG["plant_spawn_prob"]  += 0.0002
        if keys[pygame.K_d]: CFG["plant_spawn_prob"]  -= 0.0002
        if keys[pygame.K_v]: CFG["sense_radius"]      += 1
        if keys[pygame.K_b]: CFG["sense_radius"]      -= 1
        if keys[pygame.K_t]: CFG["eat_gain"]          += 0.2
        if keys[pygame.K_y]: CFG["eat_gain"]          -= 0.2
        if keys[pygame.K_u]: CFG["move_cost"]         += 0.05
        if keys[pygame.K_j]: CFG["move_cost"]         -= 0.05
        if keys[pygame.K_i]: CFG["repro_threshold"]   += 0.5
        if keys[pygame.K_k]: CFG["repro_threshold"]   -= 0.5
        if keys[pygame.K_o]: CFG["repro_cost"]        += 0.5
        if keys[pygame.K_l]: CFG["repro_cost"]        -= 0.5
        if keys[pygame.K_EQUALS] or keys[pygame.K_KP_PLUS]:  CFG["gens_per_sec"] += 1
        if keys[pygame.K_MINUS]  or keys[pygame.K_KP_MINUS]: CFG["gens_per_sec"] -= 1

        clamp_cfg("plant_spread_prob", 0.0,  0.30)
        clamp_cfg("plant_spawn_prob",  0.0,  0.02)
        clamp_cfg("eat_gain",          0.0,  50.0)
        clamp_cfg("move_cost",         0.0,  10.0)
        clamp_cfg("repro_threshold",   1.0, 200.0)
        clamp_cfg("repro_cost",        0.0, 200.0)
        CFG["sense_radius"] = max(1,  min(20,  int(CFG["sense_radius"])))
        CFG["gens_per_sec"] = max(1,  min(240, int(CFG["gens_per_sec"])))

        if not paused:
            accumulator_s += dt_s
            step_interval = 1.0 / CFG["gens_per_sec"]
            steps = 0
            while accumulator_s >= step_interval and steps < 200:
                world_step()
                accumulator_s -= step_interval
                steps += 1

        screen.fill(COL_BG)

        pygame.draw.rect(screen, COL_GRID_BG, grid_rect)
        pygame.draw.rect(screen, COL_BORDER,  grid_rect, 1)

        gap = CFG["cell_gap"]
        cs  = cell_size - 2 * gap

        plant_surf = pygame.Surface((cs, cs), pygame.SRCALPHA)
        plant_surf.fill(COL_PLANT)
        for (px, py) in plants:
            rx = grid_rect.x + px * cell_size + gap
            ry = grid_rect.y + py * cell_size + gap
            screen.blit(plant_surf, (rx, ry))

        pred_surf = pygame.Surface((cs, cs), pygame.SRCALPHA)
        pred_surf.fill(COL_PREDATOR)
        for p in predators:
            rx = grid_rect.x + p.x * cell_size + gap
            ry = grid_rect.y + p.y * cell_size + gap
            screen.blit(pred_surf, (rx, ry))

        pygame.draw.rect(screen, COL_PANEL_BG, panel_rect)
        pygame.draw.line(
            screen, COL_BORDER,
            (panel_rect.x, 0), (panel_rect.x, screen.get_height()), 1
        )

        px = panel_rect.x
        pw = panel_rect.w
        inner_w = pw - 28

        title = font_title.render("ECOSYSTEM", True, COL_TEXT_HEAD)
        screen.blit(title, (px + 14, 16))

        status_text  = "PAUSED" if paused else "RUNNING"
        status_color = COL_PAUSE if paused else COL_RUN
        dot_r = 5
        dot_x = px + 14 + dot_r
        dot_y = 48 + dot_r
        pygame.draw.circle(screen, status_color, (dot_x, dot_y), dot_r)
        status_surf = font_big.render(status_text, True, status_color)
        screen.blit(status_surf, (dot_x + dot_r + 8, 44))

        gen_surf = font_small.render(f"Generation  {generation}", True, COL_TEXT_DIM)
        screen.blit(gen_surf, (px + 14, 70))

        draw_divider(screen, px + 14, 90, inner_w)

        graph_rect = pygame.Rect(px + 14, 98, inner_w, min(200, panel_rect.h - 420))
        draw_graph(screen, graph_rect, hist_green, hist_blue, font_small)

        count_y = graph_rect.bottom + 10

        pygame.draw.circle(screen, COL_ACCENT_G, (px + 22, count_y + 8), 5)
        plants_count = font_big.render(f"{len(plants):,}", True, COL_ACCENT_G)
        plants_label = font_small.render("plants", True, COL_TEXT_DIM)
        screen.blit(plants_count, (px + 32, count_y))
        screen.blit(plants_label, (px + 32, count_y + 20))

        mid_x = px + pw // 2
        pygame.draw.circle(screen, COL_ACCENT_B, (mid_x + 8, count_y + 8), 5)
        pred_count = font_big.render(f"{len(predators):,}", True, COL_ACCENT_B)
        pred_label = font_small.render("predators", True, COL_TEXT_DIM)
        screen.blit(pred_count, (mid_x + 18, count_y))
        screen.blit(pred_label, (mid_x + 18, count_y + 20))

        params_y = count_y + 48
        draw_divider(screen, px + 14, params_y, inner_w)
        params_y += 8

        params_label = font_small.render("PARAMETERS", True, COL_TEXT_DIM)
        screen.blit(params_label, (px + 14, params_y))
        params_y += 18

        params = [
            ("spread prob",   f"{CFG['plant_spread_prob']:.3f}", "G/H"),
            ("spawn prob",    f"{CFG['plant_spawn_prob']:.4f}",  "E/D"),
            ("sense radius",  f"{CFG['sense_radius']}",          "V/B"),
            ("eat gain",      f"{CFG['eat_gain']:.1f}",          "T/Y"),
            ("move cost",     f"{CFG['move_cost']:.2f}",         "U/J"),
            ("repro thresh",  f"{CFG['repro_threshold']:.1f}",   "I/K"),
            ("repro cost",    f"{CFG['repro_cost']:.1f}",        "O/L"),
            ("gens / sec",    f"{CFG['gens_per_sec']}",          "+/-"),
        ]

        for label, value, keys_hint in params:
            draw_stat_row(screen, font_small, label, value, keys_hint, px + 14, params_y, inner_w)
            params_y += 17

        draw_divider(screen, px + 14, params_y + 4, inner_w)
        params_y += 14

        controls = [
            ("SPACE",     "pause / resume"),
            ("N",         "step (when paused)"),
            ("X",         "reset world"),
            ("F",         "fullscreen"),
            ("L-click",   "place plant"),
            ("R-click",   "place predator"),
            ("Shift+click", "erase cell"),
        ]

        for key, desc in controls:
            key_surf  = font_small.render(key, True, (70, 90, 120))
            desc_surf = font_small.render(desc, True, COL_TEXT_DIM)
            screen.blit(key_surf,  (px + 14, params_y))
            screen.blit(desc_surf, (px + 14 + 90, params_y))
            params_y += 16

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
