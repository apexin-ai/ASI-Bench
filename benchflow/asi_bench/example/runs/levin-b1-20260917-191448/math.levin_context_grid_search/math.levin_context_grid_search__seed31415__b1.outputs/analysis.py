"""Local product-of-contexts policy for Levin Tree Search.

Only training trajectories are replayed. Inference uses local observations, not
level geometry. All modular coefficients and mixture weights are fitted.
"""
import itertools
import math
from collections import Counter

ACTIONS = ('U', 'D', 'L', 'R')
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
INDEX = {a: i for i, a in enumerate(ACTIONS)}


def _symbol(value):
    if isinstance(value, dict):
        value = value.get('cell', value.get('symbol', value.get('marker', '#')))
    return str(value) if value is not None else '#'


def _observation(state, legal):
    history = state.get('history', state.get('history_tail', ())) or ()
    if isinstance(history, str):
        history = [a for a in history if a in INDEX]
    else:
        history = list(history)
    prev = state.get('previous_action', history[-1] if history else None)
    prev = INDEX.get(prev, prev if isinstance(prev, int) and 0 <= prev < 4 else 4)
    depth = int(state.get('depth', len(history)))
    run = state.get('run_length', state.get('run_length_bucket'))
    if run is None:
        run = 0
        for a in reversed(history):
            if INDEX.get(a, a) != prev:
                break
            run += 1
    try:
        run = int(run)
    except (ValueError, TypeError):
        run = 0
    cell = _symbol(state.get('cell', '?'))
    marker = ord(cell) - ord('a') if len(cell) == 1 else -1
    neighbors = state.get('neighbors', {}) or {}
    if isinstance(neighbors, dict):
        near = tuple(_symbol(neighbors.get(a, '#')) for a in ACTIONS)
    elif isinstance(neighbors, (tuple, list)) and len(neighbors) == 4:
        near = tuple(_symbol(v) for v in neighbors)
    else:
        near = ('#',) * 4
    tail = tuple(INDEX.get(a, a) for a in history[-2:])
    mask = sum(1 << INDEX[a] for a in legal if a in INDEX)
    return cell, marker, prev, depth, run, tail, mask, near


def _contexts(x):
    cell, marker, prev, depth, run, tail, mask, near = x
    run = min(run, 3)
    mod = depth % 4
    phase = depth // 15
    return (
        (cell,), (prev,), (tail, run), (mod, phase), (mask,),
        (cell, prev, run, mod), (cell, tail, mod),
        (cell, prev, run, mod, phase), (cell, mask),
        (prev, run, mod, phase),
    ) + tuple((a, near[a]) for a in range(4)) + tuple(
        (cell, a, near[a], prev) for a in range(4)
    )


def _replay(training_data):
    levels = training_data.get('levels', training_data.get('training_levels', [])) if isinstance(training_data, dict) else training_data
    rows = []
    for li, level in enumerate(levels):
        grid = level.get('grid')
        if not grid or 'start' not in level:
            continue
        r, c = level['start']
        history = []
        run = 0
        solution = level.get('solution', level.get('trajectory', []))
        for depth, action in enumerate(solution):
            if isinstance(action, dict):
                action = action.get('action')
            if action not in INDEX:
                break
            near = {}
            for a, (dr, dc) in zip(ACTIONS, DELTAS):
                rr, cc = r + dr, c + dc
                near[a] = grid[rr][cc] if 0 <= rr < len(grid) and 0 <= cc < len(grid[rr]) else '#'
            legal = [a for a in ACTIONS if near[a] != '#']
            state = dict(cell=grid[r][c], neighbors=near, history=history,
                         previous_action=history[-1] if history else None,
                         depth=depth, run_length=run)
            if action not in legal:
                break
            rows.append((_observation(state, legal), INDEX[action], li))
            dr, dc = DELTAS[INDEX[action]]
            r, c = r + dr, c + dc
            run = run + 1 if history and history[-1] == action else 1
            history.append(action)
    return rows


def _mod_value(x, model):
    cm, cp, cr, cd, cf, period, phase, cap, offset, initial = model
    _, marker, prev, depth, run, _, _, _ = x
    value = cm * marker + cp * prev + cr * min(run, cap)
    value += cd * (depth % period) + cf * (depth // phase)
    return (value + (initial if depth == 0 else offset)) % 4


def _fit_modular(rows):
    # Diversify the small screening set across levels and trajectory positions.
    data = [(x, y) for x, y, _ in rows if x[3] > 0 and x[1] >= 0]
    if len(data) < 24:
        return None
    sample = [data[(i * 104729) % len(data)] for i in range(min(48, len(data)))]
    candidates = []
    coefficients = list(itertools.product((1, 3), range(4), range(4), range(4), range(4)))
    for period in (4, 3, 5, 2, 6, 7, 8):
        for phase in range(4, 21):
            for cap in range(1, 7):
                features = [(x[1], x[2], min(x[4], cap), x[3] % period, x[3] // phase, y) for x, y in sample]
                for cm, cp, cr, cd, cf in coefficients:
                    m, p, r, d, f, y = features[0]
                    off = (y - cm*m - cp*p - cr*r - cd*d - cf*f) % 4
                    errors = 0
                    for m, p, r, d, f, y in features[1:]:
                        if (cm*m + cp*p + cr*r + cd*d + cf*f + off - y) % 4:
                            errors += 1
                            if errors > 2:
                                break
                    if errors <= 2:
                        candidates.append((errors, (cm, cp, cr, cd, cf, period, phase, cap, off, off)))
    if not candidates:
        return None
    candidates.sort(key=lambda v: v[0])
    best, best_hits = None, -1
    for _, model in candidates[:128]:
        hits = sum(_mod_value(x, model) == y for x, y in data)
        if hits > best_hits:
            best, best_hits = model, hits
    if best_hits < .9 * len(data):
        return None
    initial_offsets = Counter()
    for x, y, _ in rows:
        if x[3] == 0:
            initial_offsets[(y - _mod_value(x, best) + best[-1]) % 4] += 1
    if initial_offsets:
        best = best[:-1] + (initial_offsets.most_common(1)[0][0],)
    return best


def _tables(rows):
    tables = [{} for _ in range(18)]
    for x, y, _ in rows:
        for table, key in zip(tables, _contexts(x)):
            counts = table.setdefault(key, [0, 0, 0, 0])
            counts[y] += 1
    return tables


def _log_predictors(x, tables, model, epsilon):
    legal = [i for i in range(4) if x[6] & (1 << i)]
    result = []
    for table, key in zip(tables, _contexts(x)):
        counts = table.get(key, (0, 0, 0, 0))
        total = sum(counts[i] + .5 for i in legal)
        result.append([math.log((counts[i] + .5) / total) for i in legal])
    if model is not None:
        pred = _mod_value(x, model)
        if pred in legal and x[1] >= 0:
            scores = [1.0 - epsilon if i == pred else epsilon / max(1, len(legal)-1) for i in legal]
        else:
            scores = [1.0 / len(legal)] * len(legal)
        result.append([math.log(max(p, 1e-12)) for p in scores])
    return legal, result


class Policy:
    def __init__(self, training_data, config=None):
        self.config = config or {}
        rows = _replay(training_data)
        train = [r for r in rows if r[2] % 5 != 0]
        valid = [r for r in rows if r[2] % 5 == 0]
        if not train:
            train, valid = rows, rows
        model = _fit_modular(train)
        tables = _tables(train)
        # Entire levels are held out, rather than adjacent states on one path.
        mistakes = sum(_mod_value(x, model) != y for x, y, _ in valid) if model else len(valid)
        self.epsilon = max(.0001, min(.2, (mistakes + .5) / (len(valid) + 1)))
        n = len(tables) + (model is not None)
        self.weights = [1.0 / n] * n
        if model is not None:
            self.weights = [.1 / len(tables)] * len(tables) + [.9]
        validation = []
        for x, y, _ in valid:
            legal, logs = _log_predictors(x, tables, model, self.epsilon)
            if y in legal:
                validation.append((legal.index(y), logs))
        # Exponentiated-gradient optimization of geometric mixture log loss.
        for _ in range(100):
            gradient = [0.0] * n
            for target, logs in validation:
                scores = [sum(w * log[i] for w, log in zip(self.weights, logs)) for i in range(len(logs[0]))]
                maximum = max(scores)
                probs = [math.exp(s - maximum) for s in scores]
                total = sum(probs)
                probs = [p / total for p in probs]
                for j, log in enumerate(logs):
                    gradient[j] += sum(p * v for p, v in zip(probs, log)) - log[target]
            for j in range(n):
                self.weights[j] *= math.exp(-2.0 * gradient[j] / max(1, len(validation)))
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
        self.model = _fit_modular(rows)
        self.tables = _tables(rows)
        if (self.model is None) != (model is None):
            self.weights = [1.0 / (18 + (self.model is not None))] * (18 + (self.model is not None))
        self.training_states = len(rows)
        self.validation_states = len(valid)
        self.validation_modular_errors = mistakes if model else None

    def action_probs(self, level, state, legal_actions):
        legal_actions = [a for a in legal_actions if a in INDEX]
        if not legal_actions:
            return {}
        if len(legal_actions) == 1:
            return {legal_actions[0]: 1.0}
        x = _observation(state, legal_actions)
        legal, logs = _log_predictors(x, self.tables, self.model, self.epsilon)
        scores = [sum(w * log[i] for w, log in zip(self.weights, logs)) for i in range(len(legal))]
        maximum = max(scores)
        probs = [math.exp(s - maximum) for s in scores]
        total = sum(probs)
        return {ACTIONS[i]: p / total for i, p in zip(legal, probs)}


def make_policy(training_data: dict, config: dict | None = None):
    return Policy(training_data, config)
