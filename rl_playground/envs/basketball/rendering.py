"""Pygame drawing only. Physics and rewards live in env.py."""

import math

import numpy as np
import pygame

from rl_playground.envs.basketball import config


class Renderer:
    def __init__(self, mode):
        pygame.font.init()
        self.mode = mode
        self.size = (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.canvas = pygame.Surface(self.size)
        self.window = None
        self.clock = pygame.time.Clock()
        self.fonts = {
            size: pygame.font.SysFont("Arial", size, bold=True)
            for size in (13, 15, 17, 20, 26, 36)
        }
        if mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(self.size)
            pygame.display.set_caption("Basketball Launcher | your shot, your physics")

    def text(self, message, pos, size=17, color=(231, 236, 245)):
        self.canvas.blit(self.fonts[size].render(str(message), True, color), pos)

    def render(self, env):
        s = env.settings
        c = self.canvas
        w, h = self.size
        ink, muted, orange = (27, 40, 57), (116, 128, 145), (242, 133, 55)
        floor = h - 155
        scale = min((w - 120) / s.world_width, (floor - 150) / s.world_height)
        left = (w - s.world_width * scale) / 2

        def point(x, y):
            return (round(left + x * scale), round(floor - y * scale))

        c.fill((16, 25, 39))
        self.text("COURTSIDE", (38, 24), 26)
        self.text("BASKETBALL LAUNCHER / 01", (39, 59), 13, (137, 158, 183))
        self.text(env.mode_label, (435, 37), 15, orange)
        self.text(f"{env.baskets:02d} / {env.attempts:02d}", (w - 178, 23), 36)
        self.text("BASKETS / SHOTS", (w - 175, 65), 13, (137, 158, 183))

        # A quiet indoor court keeps the ball and wind readable.
        pygame.draw.rect(
            c, (237, 233, 222), (24, 100, w - 48, floor - 72), border_radius=18
        )
        for metres in range(1, math.ceil(s.world_height)):
            y = point(0, metres)[1]
            pygame.draw.line(c, (223, 221, 213), (44, y), (w - 44, y))
            self.text(f"{metres}m", (48, y - 16), 13, (174, 174, 166))
        pygame.draw.rect(c, (216, 175, 125), (24, floor, w - 48, 29))
        pygame.draw.line(c, (171, 125, 82), (24, floor), (w - 24, floor), 3)
        for x in range(50, w - 30, 75):
            pygame.draw.line(c, (200, 156, 107), (x, floor + 3), (x + 12, floor + 27))

        # Hoop and net: the scoring opening is between the two orange rim edges.
        board_x = s.hoop_x + s.backboard_offset
        pygame.draw.line(
            c,
            (173, 181, 183),
            point(board_x + 0.23, 0),
            point(board_x + 0.23, s.hoop_y + 0.7),
            7,
        )
        pygame.draw.line(
            c,
            (173, 181, 183),
            point(board_x, s.hoop_y + 0.7),
            point(board_x + 0.23, s.hoop_y + 0.7),
            7,
        )
        pygame.draw.line(
            c,
            (65, 87, 99),
            point(board_x, s.hoop_y - s.backboard_below),
            point(board_x, s.hoop_y + s.backboard_above),
            6,
        )
        pygame.draw.line(
            c,
            (248, 251, 249),
            point(board_x - 0.025, s.hoop_y - s.backboard_below),
            point(board_x - 0.025, s.hoop_y + s.backboard_above),
            2,
        )
        rim_left, rim_right = s.hoop_x - s.hoop_width / 2, s.hoop_x + s.hoop_width / 2
        for i in range(6):
            x = rim_left + i * s.hoop_width / 5
            bottom = s.hoop_x + (x - s.hoop_x) * 0.55
            pygame.draw.line(
                c,
                (156, 164, 161),
                point(x, s.hoop_y - 0.03),
                point(bottom, s.hoop_y - 0.52),
                1,
            )
        for depth in (0.16, 0.33, 0.50):
            half = s.hoop_width / 2 * (1 - depth * 0.85)
            pygame.draw.line(
                c,
                (156, 164, 161),
                point(s.hoop_x - half, s.hoop_y - depth),
                point(s.hoop_x + half, s.hoop_y - depth),
                1,
            )
        pygame.draw.line(
            c, orange, point(rim_left, s.hoop_y), point(rim_right, s.hoop_y), 3
        )
        for x in (rim_left, rim_right):
            pygame.draw.circle(
                c,
                (213, 91, 35),
                point(x, s.hoop_y),
                max(3, round(s.rim_radius * scale)),
            )

        # Launcher, chosen direction, and the actual path (never a predicted solution).
        base = point(s.launcher_x, 0)
        origin = point(s.launcher_x, s.launcher_y)
        pygame.draw.line(c, ink, base, origin, 10)
        pygame.draw.line(
            c, ink, (base[0] - 30, base[1] - 2), (base[0] + 30, base[1] - 2), 8
        )
        speed, angle = env.decode_action(env.aim)
        radians = math.radians(angle)
        tip = point(
            s.launcher_x + 0.8 * math.cos(radians),
            s.launcher_y + 0.8 * math.sin(radians),
        )
        pygame.draw.line(c, (67, 93, 111), origin, tip, 15)
        pygame.draw.circle(c, ink, origin, 13)
        if env.result == "READY":
            end = point(
                s.launcher_x + 1.3 * math.cos(radians),
                s.launcher_y + 1.3 * math.sin(radians),
            )
            pygame.draw.line(c, orange, tip, end, 2)
        for index in range(0, len(env.trajectory), 12):
            x, y = env.trajectory[index]
            pygame.draw.circle(c, (192, 157, 123), point(x, y), 2)
        bx, by = point(*env.ball)
        radius = max(8, round(s.ball_radius * scale))
        pygame.draw.circle(c, (205, 189, 164), (bx + 3, by + 4), radius)
        pygame.draw.circle(c, orange, (bx, by), radius)
        pygame.draw.circle(c, (117, 62, 32), (bx, by), radius, 1)
        pygame.draw.line(
            c, (117, 62, 32), (bx - radius + 2, by), (bx + radius - 2, by), 1
        )
        pygame.draw.line(
            c, (117, 62, 32), (bx, by - radius + 2), (bx, by + radius - 2), 1
        )

        # Both a numeric acceleration and moving marks communicate wind.
        wx, wy = w - 320, 120
        pygame.draw.rect(
            c, (253, 250, 242), (wx - 14, wy - 6, 280, 91), border_radius=12
        )
        direction = (
            "RIGHT" if env.wind > 0.025 else "LEFT" if env.wind < -0.025 else "CALM"
        )
        self.text(f"WIND  {direction}", (wx, wy), 15, ink)
        self.text(
            f"{abs(env.wind):.2f} m/s2  |  fixed this shot", (wx, wy + 24), 13, muted
        )
        if abs(env.wind) > 0.025:
            sign = 1 if env.wind > 0 else -1
            length = 18 + 78 * abs(env.wind) / max(s.max_wind, 1e-8)
            start = (wx + 128 - sign * length / 2, wy + 65)
            end = (start[0] + sign * length, start[1])
            pygame.draw.line(c, (49, 143, 151), start, end, 3)
            pygame.draw.polygon(
                c,
                (49, 143, 151),
                [end, (end[0] - sign * 9, end[1] - 5), (end[0] - sign * 9, end[1] + 5)],
            )
            phase = (
                (pygame.time.get_ticks() / 700) % 1
                if self.mode == "human"
                else env.flight_time % 1
            )
            for i in range(3):
                px = wx + 15 + ((i / 3 + sign * phase) % 1) * 40
                pygame.draw.line(
                    c, (108, 182, 184), (px, wy + 61), (px + sign * 10, wy + 61), 2
                )

        # Controls stay visible while the user is aiming.
        self.text("POWER", (40, h - 105), 13, (137, 158, 183))
        self.text(f"{env.aim[0] * 100:.0f}%  /  {speed:.1f} m/s", (40, h - 83), 20)
        pygame.draw.rect(c, (46, 61, 80), (40, h - 49, 220, 5), border_radius=2)
        pygame.draw.rect(
            c, orange, (40, h - 49, round(220 * float(env.aim[0])), 5), border_radius=2
        )
        self.text("ANGLE", (310, h - 105), 13, (137, 158, 183))
        self.text(f"{angle:.1f} degrees", (310, h - 83), 20)
        status_color = (91, 208, 155) if env.result == "BASKET!" else orange
        self.text(env.result, (565, h - 96), 26, status_color)
        self.text("UP/DOWN angle   LEFT/RIGHT power", (w - 342, h - 105), 13)
        self.text("SPACE shoot / next   R new wind   ESC quit", (w - 342, h - 81), 13)
        self.text(
            "ONE SHOT. READ THE WIND. FIND YOUR ARC.", (40, h - 25), 13, (104, 123, 148)
        )

        if self.mode == "rgb_array":
            return np.transpose(pygame.surfarray.array3d(c), (1, 0, 2))
        # Keep close/Escape responsive even while step() animates a flight.
        for event in pygame.event.get([pygame.QUIT]):
            env.window_closed = True
        if pygame.key.get_pressed()[pygame.K_ESCAPE]:
            env.window_closed = True
        if not env.window_closed:
            self.window.blit(c, (0, 0))
            pygame.display.flip()
            self.clock.tick(config.FPS)
        return None

    def close(self):
        if self.window is not None:
            pygame.display.quit()
