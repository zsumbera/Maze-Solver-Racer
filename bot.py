import sys
import math
import time
import random
from heapq import heappush, heappop
from collections import deque
from typing import List, Set, Tuple, Dict, Optional

# --- KONSTANSOK ---
# A térképen előforduló cellatípusok numerikus kódjai
EMPTY = 0
WALL = -1
START = 1
UNKNOWN = 3
OIL = 91
SAND = 92
GOAL = 100

Coord = Tuple[int, int]


def read_ints(line: str) -> List[int]:
    """Segédfüggvény a standard bemenetről érkező számsorok listává alakításához."""
    if not line: return []
    return list(map(int, line.strip().split()))


class MapManager:
    """A globális térkép és a felfedezett információk (célok, látogatottság) kezelője."""

    def __init__(self, h, w):
        self.height = h
        self.width = w
        # A teljes pálya inicializálása ismeretlenként
        self.grid = [[UNKNOWN for _ in range(w)] for _ in range(h)]
        self.goals: Set[Coord] = set()
        self.visit_counts: Dict[Coord, int] = {}
        self.start_pos: Optional[Coord] = None

    def update(self, bot_r, bot_c, local_grid, radius):
        """Frissíti a belső térképet a bot aktuális látómezeje alapján."""
        if self.start_pos is None:
            self.start_pos = (bot_r, bot_c)

        # Látogatottság növelése az aktuális cellán
        self.visit_counts[(bot_r, bot_c)] = self.visit_counts.get((bot_r, bot_c), 0) + 1

        # Szomszédos cellák inicializálása a visit_counts-ban a mozgás finomításához
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r, c = bot_r + dr, bot_c + dc
                if 0 <= r < self.height and 0 <= c < self.width:
                    if (r, c) not in self.visit_counts:
                        self.visit_counts[(r, c)] = 0

        # A látott terület (local_grid) beírása a globális térképbe
        size = 2 * radius + 1
        for lr in range(size):
            for lc in range(size):
                gr = bot_r - radius + lr
                gc = bot_c - radius + lc

                if 0 <= gr < self.height and 0 <= gc < self.width:
                    cell_val = local_grid[lr][lc]
                    if self.grid[gr][gc] != cell_val:
                        self.grid[gr][gc] = cell_val
                        # Célpontok követése: ha GOAL-t látunk, hozzáadjuk, ha már nem az, töröljük
                        if cell_val == GOAL:
                            self.goals.add((gr, gc))
                        elif (gr, gc) in self.goals:
                            self.goals.discard((gr, gc))

    def is_walkable(self, r, c):
        """Ellenőrzi, hogy a megadott koordináta a pályán belül van-e és nem fal."""
        if not (0 <= r < self.height and 0 <= c < self.width):
            return False
        return self.grid[r][c] != WALL

    def get_dynamic_cushion(self, r, c, current_speed):
        """
        SEBESSÉGFÜGGŐ BIZTONSÁGI SÁV
        Ha gyorsan megyünk, jobban félünk a faltól.
        Ha lassan, akkor simulhatunk.
        """
        wall_penalty = 0.0
        neighbor_walls = 0

        # Falak számolása a szomszédban
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                if self.grid[nr][nc] == WALL:
                    neighbor_walls += 1

        if neighbor_walls > 0:
            # Alapból kis büntetés (hogy preferálja a közepét)
            wall_penalty = 2.0
            # Ha gyorsak vagyunk, növeljük a büntetést (ne csapódjunk be)
            if current_speed > 2.0:
                wall_penalty += 10.0

        return wall_penalty

    def get_visit_penalty(self, r, c):
        """Büntetés a már sokszor meglátogatott helyek elkerülésére."""
        return self.visit_counts.get((r, c), 0) * 50.0

    def get_target(self, bot_r, bot_c, vr, vc):
        """Meghatározza a bot aktuális úticélját (lehet konkrét cél vagy ismeretlen terület)."""
        if self.goals:
            # Ha ismerünk GOAL cellát, a legközelebbit választjuk
            return min(self.goals, key=lambda g: abs(g[0] - bot_r) + abs(g[1] - bot_c))

        # Irányvektor és sebesség számítása a felfedezéshez
        speed = math.sqrt(vr ** 2 + vc ** 2)
        has_momentum = speed > 0.5
        norm_vr, norm_vc = 0, 0
        if has_momentum:
            norm_vr, norm_vc = vr / speed, vc / speed

        # BFS (szélességi keresés) az ismeretlen (UNKNOWN) cellák felé
        q = deque([(bot_r, bot_c)])
        visited = {(bot_r, bot_c)}

        best_target = None
        max_dist_from_start = -1

        idx = 0
        while idx < len(q):
            r, c = q[idx];
            idx += 1
            idx += 1

            if idx > 3000: break  # Teljesítményvédelem: ne keressünk túl mélyen

            if self.grid[r][c] == UNKNOWN:
                dist_start = 0
                if self.start_pos:
                    dist_start = abs(r - self.start_pos[0]) + abs(c - self.start_pos[1])

                # Irány szűrés: preferáljuk azokat az ismeretlen cellákat, amik felé haladunk
                is_forward = True
                if has_momentum:
                    tr, tc = r - bot_r, c - bot_c
                    dot = tr * norm_vr + tc * norm_vc
                    if dot < -0.2: is_forward = False

                if is_forward:
                    if dist_start > max_dist_from_start:
                        max_dist_from_start = dist_start
                        best_target = (r, c)

            # Szomszédok hozzáadása a sorhoz
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if (nr, nc) not in visited:
                        if self.grid[nr][nc] != WALL:
                            visited.add((nr, nc))
                            q.append((nr, nc))

        if best_target:
            return best_target

        return self.get_target_fallback(bot_r, bot_c)

    def get_target_fallback(self, bot_r, bot_c):
        """Egyszerűbb BFS keresés, ha az irányított keresés nem talált célt."""
        q = deque([(bot_r, bot_c)])
        visited = {(bot_r, bot_c)}
        idx = 0
        while idx < len(q):
            r, c = q[idx];
            idx += 1
            if self.grid[r][c] == UNKNOWN: return (r, c)
            if idx > 1000: break
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and self.is_walkable(nr, nc):
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return (self.height // 2, self.width // 2)


class AStarPathfinder:
    """A* algoritmus az optimális útvonal megkereséséhez a célpontig."""

    def get_path(self, start: Coord, target: Coord, map_mgr: MapManager, player_positions: Set[Coord],
                 current_speed: float) -> List[Coord]:
        pq = [(0, start)]  # Prioritási sor: (költség, koordináta)
        came_from = {start: None}
        cost_so_far = {start: 0}

        while pq:
            _, current = heappop(pq)
            if current == target: break

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_node = (current[0] + dr, current[1] + dc)

                if not map_mgr.is_walkable(next_node[0], next_node[1]): continue
                if next_node in player_positions: continue  # Másik bot elkerülése

                # DINAMIKUS KÖLTSÉG: Fal közelsége és látogatottság alapján
                cushion = map_mgr.get_dynamic_cushion(next_node[0], next_node[1], current_speed)
                visit_penalty = map_mgr.get_visit_penalty(next_node[0], next_node[1])

                cell_cost = 1 + cushion + visit_penalty

                new_cost = cost_so_far[current] + cell_cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    # Heurisztika (Manhattan távolság) súlyozva
                    h = (abs(next_node[0] - target[0]) + abs(next_node[1] - target[1])) * 1.5
                    heappush(pq, (new_cost + h, next_node))
                    came_from[next_node] = current

        # Ha nem értük el a célt, a legközelebbi ismert ponthoz tervezünk
        if target not in came_from:
            if not cost_so_far: return []
            target = min(cost_so_far, key=lambda n: cost_so_far[n] + abs(n[0] - target[0]) + abs(n[1] - target[1]))

        # Útvonal visszafejtése
        path = []
        curr = target
        while curr is not None:
            path.append(curr)
            curr = came_from.get(curr)
        path.reverse()
        return path


class Bot:
    """A bot fő vezérlő osztálya, itt dől el a gyorsulás iránya."""

    def __init__(self, H, W, N, R):
        self.map_mgr = MapManager(H, W)
        self.pathfinder = AStarPathfinder()
        self.R = R
        self.pos_history = deque(maxlen=8)
        self.last_pos = None
        self.last_sent_accel = (0, 0)
        self.failed_moves_in_row = set()
        self.stuck_counter = 0

    def update_memory(self, my_r, my_c, local_grid):
        """Frissíti a bot belső állapotát és észleli, ha elakadt."""
        self.map_mgr.update(my_r, my_c, local_grid, self.R)

        if self.last_pos == (my_r, my_c):
            self.stuck_counter += 1
            if self.last_sent_accel != (0, 0):
                self.failed_moves_in_row.add(self.last_sent_accel)
        else:
            self.stuck_counter = 0
            self.failed_moves_in_row.clear()

        self.last_pos = (my_r, my_c)
        self.pos_history.append((my_r, my_c))

    def check_future_collision(self, r, c, vr, vc, ar, ac):
        """
        CRASH GUARD: Szimulálja a jövőbeli pozíciót.
        Ha fal, akkor FALSE-t ad vissza.
        """
        # Következő sebesség számítása a fizikának megfelelően
        next_vr = vr + ar
        next_vc = vc + ac

        # Következő pozíció
        next_r = r + next_vr
        next_c = c + next_vc

        # Raycasting: végigellenőrizzük a mozgásvonalat a falak ellen
        steps = max(abs(next_vr), abs(next_vc))
        if steps == 0: return self.map_mgr.is_walkable(int(next_r), int(next_c))

        for i in range(1, steps + 1):
            t = i / steps
            intermediate_r = int(r + next_vr * t)
            intermediate_c = int(c + next_vc * t)
            if not self.map_mgr.is_walkable(intermediate_r, intermediate_c):
                return False  # ÜTKÖZÉS!

        return True

    def choose_acceleration(self, r, c, vr, vc, player_positions):
        """A bot legfontosabb döntéshozatali logikája."""
        current_speed = math.sqrt(vr ** 2 + vc ** 2)

        # 1. HARD UNSTUCK: Ha elakadtunk, próbálunk egy érvényes szabad irányba lökni
        if self.stuck_counter > 3:
            valid_moves = []
            for ar, ac in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                if (ar, ac) in self.failed_moves_in_row: continue

                # CRASH GUARD használata a biztonságos meneküléshez
                if self.check_future_collision(r, c, vr, vc, ar, ac):
                    valid_moves.append((ar, ac))

            if valid_moves:
                move = random.choice(valid_moves)
                self.last_sent_accel = move
                return move
            return (0, 0)

        # 2. Célkeresés és útvonaltervezés
        target = self.map_mgr.get_target(r, c, vr, vc)
        path = self.pathfinder.get_path((r, c), target, self.map_mgr, player_positions, current_speed)

        if len(path) < 2:
            self.last_sent_accel = (0, 0)
            return (0, 0)

        # 3. Lookahead: A célpontot nem a közvetlen szomszédból, hanem távolabbról választjuk a sebesség függvényében
        lookahead_idx = int(current_speed * 1.5) + 2
        lookahead_idx = min(lookahead_idx, len(path) - 1)
        aim_point = path[lookahead_idx]

        # Kívánt sebességvektor számítása
        target_vr = aim_point[0] - r
        target_vc = aim_point[1] - c
        dist = math.sqrt(target_vr ** 2 + target_vc ** 2)

        if dist > 0:
            desired_speed = 5.0  # Maximális sebesség
            if len(path) < 5:
                desired_speed = 1.0  # Lassítás a cél közelében
            elif len(path) < 10:
                desired_speed = 2.0

            target_vr = (target_vr / dist) * desired_speed
            target_vc = (target_vc / dist) * desired_speed

        # Szükséges gyorsulás
        req_ar = target_vr - vr
        req_ac = target_vc - vc

        def clamp(val):
            return max(-1, min(1, int(round(val))))

        # 4. SMART AVOIDANCE + CRASH GUARD: A legjobb biztonságos gyorsulás kiválasztása
        best_alt = None
        min_penalty = float('inf')

        # Minden lehetséges gyorsulási kombináció (-1, 0, 1) tesztelése
        for ar in range(-1, 2):
            for ac in range(-1, 2):
                if (ar, ac) in self.failed_moves_in_row: continue

                nr = r + vr + ar
                nc = c + vc + ac

                # Ütközésvizsgálat
                if not self.check_future_collision(r, c, vr, vc, ar, ac):
                    continue

                # Költségfüggvény: távolság a célponttól + korábbi pozíciók büntetése
                dist_sq = (nr - aim_point[0]) ** 2 + (nc - aim_point[1]) ** 2
                penalty = dist_sq * 1.0

                # Ne menjünk oda, ahol mostanában voltunk (oszcilláció ellen)
                for i, pos in enumerate(reversed(self.pos_history)):
                    if pos == (int(nr), int(nc)):
                        penalty += (10 - i) * 50
                        if i == 0: penalty += 200

                # Állóhelyzet büntetése, ha menni kellene
                if current_speed < 0.1 and ar == 0 and ac == 0:
                    penalty += 5000.0

                if penalty < min_penalty:
                    min_penalty = penalty
                    best_alt = (ar, ac)

        # Tartalék terv, ha nincs ideális lépés
        if best_alt is None:
            for ar, ac in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if (ar, ac) not in self.failed_moves_in_row:
                    if self.check_future_collision(r, c, vr, vc, ar, ac):
                        best_alt = (ar, ac);
                        break

            if best_alt is None:
                best_alt = (0, 0)

        # 5. FORCE START: Indítás segítése, ha a bot nem akarna elindulni
        if current_speed < 0.1 and best_alt == (0, 0):
            if len(path) >= 2:
                next_pos = path[1]
                force_ar = clamp(next_pos[0] - r)
                force_ac = clamp(next_pos[1] - c)

                if self.check_future_collision(r, c, vr, vc, force_ar, force_ac):
                    best_alt = (force_ar, force_ac)

            if best_alt == (0, 0):
                valid_randoms = []
                for ar, ac in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    if self.check_future_collision(r, c, vr, vc, ar, ac):
                        valid_randoms.append((ar, ac))
                if valid_randoms:
                    best_alt = random.choice(valid_randoms)

        self.last_sent_accel = best_alt
        return best_alt


def main() -> None:
    """A bot belépési pontja, kezeli a kommunikációs protokollt."""
    header = sys.stdin.readline()
    if not header: return
    try:
        # Pálya paraméterek beolvasása (Magasság, Szélesség, Játékosok száma, Látótávolság)
        parts = list(map(int, header.strip().split()))
        H, W, N, R = parts[0], parts[1], parts[2], parts[3]
    except ValueError:
        return

    bot = Bot(H, W, N, R)

    while True:
        line = sys.stdin.readline()
        if not line: break
        stripped = line.strip()
        # A verseny vége üzenet figyelése
        if stripped == "~~~END~~~": break

        # Saját pozíció és sebesség beolvasása
        parts = read_ints(stripped)
        if len(parts) < 4: continue
        my_r, my_c, my_v_r, my_v_c = parts[0], parts[1], parts[2], parts[3]

        # Többi játékos pozíciójának beolvasása
        player_positions: Set[Coord] = set()
        for _ in range(N):
            parts = read_ints(sys.stdin.readline())
            if len(parts) == 2:
                player_positions.add((parts[0], parts[1]))
        player_positions.discard((my_r, my_c))

        # Aktuális látótér (local grid) beolvasása
        local_grid: List[List[int]] = []
        for _ in range(2 * R + 1):
            row_line = sys.stdin.readline()
            if not row_line: break
            row = read_ints(row_line)
            local_grid.append(row)

        if len(local_grid) != 2 * R + 1: break

        # Memória frissítése és a következő lépés kiszámítása
        bot.update_memory(my_r, my_c, local_grid)
        a_r, a_c = bot.choose_acceleration(my_r, my_c, my_v_r, my_v_c, player_positions)

        # Gyorsulási parancs elküldése
        print(f"{a_r} {a_c}")
        sys.stdout.flush()


if __name__ == "__main__":
    # Kezdeti READY jelzés a szervernek
    print("READY")
    sys.stdout.flush()
    main()