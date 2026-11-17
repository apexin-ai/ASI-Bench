"""Private experimental-label scoring plus an independent public-input replay gate."""
from __future__ import annotations
import ast
import hashlib
import io
import json
import math
import os
import re
import secrets
import socket
import stat
import struct
import tempfile
import threading
import tokenize
import zipfile
import zlib
from pathlib import Path
from typing import Any
import numpy as np
from ai4sci_bench.core.scorer import Scorer, register_scorer
from ai4sci_bench.core.types import ScoreDetail
try:
    from ai4sci_bench.runner.runtime_root import resolve_runtime_root
except ImportError:  # Compatibility with the private source framework.
    def resolve_runtime_root(task_dir):
        return Path(task_dir).resolve().parents[2]
from ai4sci_bench.runner.submission_sandbox import ReadOnlyMount, SubmissionRunSpec, SandboxedSubmissionRunner

TASK_ID = 'biology.unseen_perturbation_rna_response'
SCORER_NAME = 'perturbation_prediction_score'
GATE_NAME = 'prediction_replay_consistency'
PREDICTION_SHAPE = (233, 3323)
MAX_OUTPUT_BYTES = 8 * 1024**2
MAX_SOURCE_BYTES = 64 * 1024
# Bound information that a source file can embed (e.g. a quantized answer copy).
MAX_COMPRESSED_SOURCE_BYTES = 12 * 1024
MAX_SOURCE_LITERAL_CHARS = 8192
MAX_SINGLE_LITERAL_CHARS = 1024
LITERAL_TOKEN_TYPES = frozenset(t for t in (tokenize.STRING, tokenize.NUMBER, tokenize.COMMENT,
                                            getattr(tokenize, 'FSTRING_MIDDLE', None)) if t is not None)
MAX_HEADER_BYTES = 4096
MAX_PUBLIC_FILE_BYTES = 64 * 1024**2
MAX_REFERENCE_BYTES = 32 * 1024**2
MAX_REFERENCE_UNCOMPRESSED_BYTES = 64 * 1024**2
PUBLIC_FILES = ('README.md', 'genes.json', 'manifest.json', 'metadata.json',
                'provenance.json', 'test_queries.json',
                'train_condition_stats.npz', 'validation_targets.npz')
REFERENCE_NPZ_SHA256 = '24da3819f2943905ce53b1995e8946e978673626ccc21a026e06a34ded36d123'
PUBLIC_INPUT_SHA256 = {'README.md': '517dc558584be77eb2549d791a9784028c061a6e2998abea72897eee1c7e7b77', 'genes.json': 'b2b436ab503e8eb80d291bfc527dbad7777ecb27a7cf6806cad96fed1708559a', 'manifest.json': 'f5cfa2ac4af1c21dccf1d031f775dadf8c4e68459c805bddf2b7f62eef183c88', 'metadata.json': '6541388b6ad8270b791c909c3c09d1a548108838b9f0f969fef7ded7861cc04b', 'provenance.json': '8ede8971696cd9b9b58ae5767de4b64cc3c830e3012dd814f20b0647230f576e', 'test_queries.json': 'c6142c4a549256e30994a28e1747b3ad1a76ac4c6a72ab7474138de2cb3d067a', 'train_condition_stats.npz': '213a9cfadf6122065a5f5f5beec43efd18c4ec70015e42e61d142a78fe5c2ae4', 'validation_targets.npz': 'd40483f6f1f22c35faa091d3150e163655827939fc53366565e948b72a872a72'}
EXPECTED_B = 0.03441638519456471
EXPECTED_T = 0.00714613515111071
ANCHOR_ATOL = 1e-12
ANCHOR_RTOL = 1e-12
REPLAY_ATOL = 1e-5
REPLAY_RTOL = 1e-6
REPLAY_TIMEOUT_SECONDS = 1200
REPLAY_SCRATCH_FILE_BYTES = 2 * 1024**3
RUNTIME_PACKAGES = ['numpy==1.24.3', 'scipy==1.11.4', 'pandas==1.5.3',
                    'scikit-learn==1.3.2']
REPLAY_RUNTIME = {'id': TASK_ID, '_task_dir': str(Path(__file__).parent.resolve()),
                  'runtime': {'python': '>=3.11,<3.12', 'dockerfile': 'runtime.txt',
                  'packages': RUNTIME_PACKAGES}, '_runtime_python': '>=3.11,<3.12',
                  '_runtime_packages': RUNTIME_PACKAGES}

def _regular_file(path: Path, limit: int) -> bytes:
    """Open a bounded regular file; ancestors are resolved once, the leaf is never followed."""
    path = Path(path).absolute()
    path = path.parent.resolve(strict=True) / path.name
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    fd = None
    try:
        for part in path.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                          dir_fd=directory)
            os.close(directory)
            directory = nxt
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=directory)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError(f"not a bounded regular file: {path.name}")
        with os.fdopen(fd, "rb") as stream:
            fd = None
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"file exceeds size limit: {path.name}")
        return data
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)

def _source_density_error(source: bytes) -> str | None:
    """Reject sources whose literals or incompressible content could carry an embedded answer."""
    compressed = len(zlib.compress(source, 9))
    if compressed > MAX_COMPRESSED_SOURCE_BYTES:
        return f'zlib-compressed source is {compressed} bytes; limit {MAX_COMPRESSED_SOURCE_BYTES}'
    total = 0
    for token in tokenize.tokenize(io.BytesIO(source).readline):
        if token.type in LITERAL_TOKEN_TYPES:
            if len(token.string) > MAX_SINGLE_LITERAL_CHARS:
                return (f'literal at line {token.start[0]} has {len(token.string)} characters; '
                        f'limit {MAX_SINGLE_LITERAL_CHARS}')
            total += len(token.string)
    if total > MAX_SOURCE_LITERAL_CHARS:
        return f'string/number/comment literals total {total} characters; limit {MAX_SOURCE_LITERAL_CHARS}'
    return None


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, OverflowError):
        return None

def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def _load_json(data):
    return json.loads(data, object_pairs_hook=_unique_object)

class _StartSignal:
    """Record the bootstrap handoff on the host, outside submitted-code control."""

    def __init__(self, directory):
        self.path = Path(directory) / 'start.sock'
        self.token = secrets.token_hex(24).encode('ascii')
        self.seen = threading.Event()
        self.stop = threading.Event()
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.thread = None
        self.error = None

    def __enter__(self):
        try:
            self.server.bind(str(self.path))
            # The host-only parent directory is private. Only this socket inode
            # is mounted; the listener accepts a fixed bounded signal.
            self.path.chmod(0o666)
            self.server.listen(1)
            self.server.settimeout(0.1)
            self.thread = threading.Thread(target=self._listen, daemon=True)
            self.thread.start()
        except Exception:
            self.server.close()
            raise
        return self

    def _listen(self):
        try:
            while not self.stop.is_set() and not self.seen.is_set():
                try:
                    connection, _ = self.server.accept()
                except socket.timeout:
                    continue
                except OSError as exc:
                    if not self.stop.is_set():
                        self.error = f'execution-signal listener failed: {exc}'
                    break
                with connection:
                    connection.settimeout(1.0)
                    payload = b''
                    try:
                        while len(payload) < len(self.token):
                            chunk = connection.recv(len(self.token) - len(payload))
                            if not chunk:
                                break
                            payload += chunk
                        if payload == self.token:
                            self.seen.set()
                            connection.sendall(b'ok')
                    except OSError as exc:
                        if self.seen.is_set():
                            self.error = f'execution-signal acknowledgement failed: {exc}'
                        continue
        except Exception as exc:
            self.error = f'execution-signal listener failed: {type(exc).__name__}: {exc}'
        finally:
            self.server.close()

    def __exit__(self, *args):
        self.stop.set()
        self.server.close()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                self.error = 'execution-signal listener did not terminate'

def _npy_array(raw: bytes, shape: tuple[int, ...], *, identifiers=False, integer_indices=False) -> np.ndarray:
    """Validate a bounded NPY header before interpreting its numeric payload."""
    stream = io.BytesIO(raw)
    version = np.lib.format.read_magic(stream)
    if version not in ((1, 0), (2, 0), (3, 0)):
        raise ValueError('unsupported NPY version')
    length_bytes = 2 if version == (1, 0) else 4
    prefix = stream.read(length_bytes)
    if len(prefix) != length_bytes:
        raise ValueError('truncated NPY header length')
    header_length = struct.unpack('<H' if length_bytes == 2 else '<I', prefix)[0]
    if header_length > MAX_HEADER_BYTES:
        raise ValueError('NPY header exceeds bounded size')
    header = stream.read(header_length)
    if len(header) != header_length:
        raise ValueError('truncated NPY header')
    try:
        info = ast.literal_eval(header.decode('utf8' if version == (3, 0) else 'latin1'))
    except (ValueError, SyntaxError, RecursionError, UnicodeError) as exc:
        raise ValueError('invalid literal NPY header') from exc
    if (not isinstance(info, dict) or set(info) != {'shape', 'fortran_order', 'descr'}
            or not isinstance(info['shape'], tuple)
            or any(type(n) is not int for n in info['shape'])
            or type(info['fortran_order']) is not bool or not isinstance(info['descr'], str)):
        raise ValueError('invalid NPY shape/order/dtype header fields')
    if info['shape'] != shape:
        raise ValueError(f'array shape must be {shape}')
    fortran, dtype = info['fortran_order'], np.dtype(info['descr'])
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise ValueError('object/structured/subarray dtypes are forbidden')
    if integer_indices:
        if dtype.kind not in ('i', 'u') or dtype.itemsize not in (1, 2, 4, 8):
            raise ValueError('self-feature columns must have an integer dtype')
    elif identifiers:
        if dtype.kind not in ('U', 'S') or not 0 < dtype.itemsize <= 1024:
            raise ValueError('reference identifiers must be bounded strings')
    elif dtype.kind != 'f' or dtype.itemsize not in (4, 8):
        raise ValueError('predictions and reference numeric arrays must be float32 or float64')
    offset = stream.tell()
    count = math.prod(shape)
    if len(raw) - offset != count * dtype.itemsize:
        raise ValueError('NPY payload size differs from declared shape/dtype')
    array = np.frombuffer(raw, dtype=dtype, count=count, offset=offset).reshape(
        shape, order='F' if fortran else 'C')
    if identifiers:
        return array.astype(str)
    if integer_indices:
        return array.astype(np.int64, copy=False)
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError('numeric array contains NaN or infinity')
    return array


def _prediction(path: Path) -> np.ndarray:
    return _npy_array(_regular_file(path, MAX_OUTPUT_BYTES), PREDICTION_SHAPE)


def _reference_metadata(ref_dir: Path, config: dict[str, Any]) -> dict:
    meta = _load_json(_regular_file(
        ref_dir / config.get('reference_file', 'reference.json'), 1024 * 1024))
    if (not isinstance(meta, dict) or meta.get('schema_version') != 1
            or meta.get('task_id') != TASK_ID or meta.get('shape') != list(PREDICTION_SHAPE)):
        raise ValueError('wrong immutable reference identity/schema/shape')
    hashes = meta.get('public_input_sha256')
    if (not isinstance(hashes, dict) or set(hashes) != set(PUBLIC_FILES)
            or hashes != PUBLIC_INPUT_SHA256):
        raise ValueError('immutable public file allowlist differs')
    for value in hashes.values():
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('invalid immutable public checksum')
    return meta


def _immutable_public(ref_dir: Path, meta: dict) -> dict[str, bytes]:
    data = {}
    for name in PUBLIC_FILES:
        raw = _regular_file(ref_dir / 'data' / name, MAX_PUBLIC_FILE_BYTES)
        if hashlib.sha256(raw).hexdigest() != meta['public_input_sha256'][name]:
            raise ValueError(f'immutable public checksum mismatch: {name}')
        data[name] = raw
    return data


def _weighted_loss(prediction, truth, weights) -> tuple[float, np.ndarray]:
    with np.errstate(over='raise', invalid='raise'):
        per_condition = np.sum(weights * np.square(prediction - truth), axis=1, dtype=np.float64)
        loss = float(np.mean(per_condition, dtype=np.float64))
    if not math.isfinite(loss):
        raise FloatingPointError('nonfinite prediction loss')
    return loss, per_condition


def _validate_self_features(arrays: dict) -> None:
    weights = arrays['metric_weights']
    # Each query is a canonical RNA target ENSG, and its own readout gets zero
    # scientific weight; no target column is removed from the input descriptors.
    if not all(re.fullmatch(r'ENSG[0-9]+', str(g)) for g in arrays['gene_ids']):
        raise ValueError('readout identifiers must be canonical RNA ENSG identifiers')
    self_columns = arrays['self_feature_columns']
    if (np.any(self_columns < 0) or np.any(self_columns >= PREDICTION_SHAPE[1])
            or not np.array_equal(arrays['gene_ids'][self_columns], arrays['condition_ids'])):
        raise ValueError('self-feature columns do not map query targets to RNA readout IDs')
    if np.any(weights[np.arange(PREDICTION_SHAPE[0]), self_columns] != 0.):
        raise ValueError('own targeted RNA must have exactly zero metric weight')


def _validated_reference(ref_dir: Path, config: dict[str, Any]) -> dict:
    meta = _reference_metadata(ref_dir, config)
    public = _immutable_public(ref_dir, meta)
    raw = _regular_file(ref_dir / 'reference.npz', MAX_REFERENCE_BYTES)
    if (meta.get('reference_npz_sha256') != REFERENCE_NPZ_SHA256
            or hashlib.sha256(raw).hexdigest() != REFERENCE_NPZ_SHA256):
        raise ValueError('immutable reference NPZ checksum mismatch')
    shapes = {'ground_truth': PREDICTION_SHAPE, 'metric_weights': PREDICTION_SHAPE,
              'technical_duplicate': PREDICTION_SHAPE,
              'training_perturbation_mean': (PREDICTION_SHAPE[1],),
              'training_control_mean': (PREDICTION_SHAPE[1],),
              'condition_ids': (PREDICTION_SHAPE[0],), 'gene_ids': (PREDICTION_SHAPE[1],),
              'self_feature_columns': (PREDICTION_SHAPE[0],)}
    arrays = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if (len(members) != len(shapes) or {m.filename for m in members} != {k+'.npy' for k in shapes}
                or sum(m.file_size for m in members) > MAX_REFERENCE_UNCOMPRESSED_BYTES
                or any(m.file_size > MAX_OUTPUT_BYTES for m in members)):
            raise ValueError('invalid immutable NPZ members or uncompressed size')
        for key, shape in shapes.items():
            arrays[key] = _npy_array(archive.read(key+'.npy'), shape,
                                     identifiers=key in ('condition_ids', 'gene_ids'),
                                     integer_indices=key == 'self_feature_columns')
    weights = arrays['metric_weights']
    if np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1., rtol=0, atol=1e-10):
        raise ValueError('private metric weights must be nonnegative and sum to1 per row')
    queries = _load_json(public['test_queries.json'])
    genes = _load_json(public['genes.json'])
    if (arrays['condition_ids'].tolist() != queries.get('condition_ids')
            or queries.get('shape') != list(PREDICTION_SHAPE)
            or arrays['gene_ids'].tolist() != genes.get('gene_ids')
            or len(set(arrays['condition_ids'])) != PREDICTION_SHAPE[0]
            or len(set(arrays['gene_ids'])) != PREDICTION_SHAPE[1]):
        raise ValueError('private/public condition or gene ordering differs')
    _validate_self_features(arrays)
    B, _ = _weighted_loss(arrays['training_perturbation_mean'][None, :], arrays['ground_truth'], weights)
    T, _ = _weighted_loss(arrays['technical_duplicate'], arrays['ground_truth'], weights)
    for key, computed, expected in [('B', B, EXPECTED_B), ('T', T, EXPECTED_T)]:
        declared = _finite_number(meta.get(key))
        if declared is None or not math.isclose(declared, expected, abs_tol=ANCHOR_ATOL, rel_tol=ANCHOR_RTOL):
            raise ValueError(f'wrong declared frozen {key} anchor')
        if not math.isclose(computed, expected, abs_tol=ANCHOR_ATOL, rel_tol=ANCHOR_RTOL):
            raise ValueError(f'computed immutable {key} anchor differs from freeze')
    if not B > T >= 0:
        raise ValueError('invalid immutable calibration anchors')
    arrays.update(B=B, T=T, reference_npz_sha256=hashlib.sha256(raw).hexdigest())
    return arrays


def _points(loss: float, B: float, T: float) -> float:
    if loss <= T:
        return 100.0
    if loss >= B:
        return 0.0
    return float(100.0 * (B-loss) / (B-T))


@register_scorer(SCORER_NAME)
class PerturbationPredictionScorer(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict[str, Any]) -> ScoreDetail:
        weight = float(config.get('weight', 100.))
        details = {}
        def finish(points, message):
            details['final_points'] = points
            return ScoreDetail(scorer_name=SCORER_NAME, score=points*weight/100.,
                               max_score=weight, passed=points > 0, details=details, message=message)
        try:
            ref = _validated_reference(Path(ref_dir), config)
        except Exception as exc:
            details.update(scorer_internal_error=True, failure_kind='missing_evaluator_input' if isinstance(exc, OSError) else 'evaluator_runtime_error', failure_category='invalid_immutable_reference',
                           error=f'{type(exc).__name__}: {exc}')
            return finish(0., 'evaluator reference validation failed; not a scientific zero')
        try:
            prediction = _prediction(Path(pred_dir) / 'results/predictions.npy')
            loss, per_condition = _weighted_loss(prediction, ref['ground_truth'], ref['metric_weights'])
            with np.errstate(over='raise', invalid='raise'):
                raw_mse = float(np.mean(np.square(prediction-ref['ground_truth']), dtype=np.float64))
        except (OSError, ValueError, TypeError, OverflowError, FloatingPointError, EOFError) as exc:
            details.update(valid=False, failure_category='invalid_prediction', error=str(exc))
            return finish(0., 'missing or invalid bounded finite prediction array')
        points = _points(loss, ref['B'], ref['T'])
        details.update(valid=True, loss=loss, raw_MSE=raw_mse,
                       per_condition_weighted_MSE=per_condition.tolist(), B=ref['B'], T=ref['T'],
                       frozen_B=EXPECTED_B, frozen_T=EXPECTED_T, shape=list(PREDICTION_SHAPE),
                       reference_npz_sha256=ref['reference_npz_sha256'])
        return finish(points, f'weighted prediction loss {loss:.12g}; scientific score {points:.6f}/100')


class PredictionSubmissionRunner(SandboxedSubmissionRunner):
    """Mature official isolation with the fixed4CPU/8GiB replay contract."""
    _TMPFS_SIZE = '8g'
    def __init__(self, repo_root, *, start_signal_path):
        super().__init__(repo_root)
        self.start_signal_path = Path(start_signal_path).resolve()

    def _build_docker_command(self, spec, **kwargs):
        command = super()._build_docker_command(spec, **kwargs)
        positions = [i+1 for i, token in enumerate(command[:-1])
                     if token == '--ulimit' and command[i+1].startswith('fsize=')]
        if len(positions) != 1:
            raise RuntimeError('unsupported SDK submission-runner fsize contract')
        command[positions[0]] = f'fsize={REPLAY_SCRATCH_FILE_BYTES}:{REPLAY_SCRATCH_FILE_BYTES}'
        if not hasattr(os, 'sched_getaffinity'):
            raise RuntimeError('cannot identify assigned CPUs for replay')
        allowed = sorted(os.sched_getaffinity(0))
        if len(allowed) < 4:
            raise RuntimeError('replay requires4assigned CPUs')
        extra = ['--cpuset-cpus', ','.join(map(str, allowed[:4])), '--memory-swap', '8g',
                 '-v', f'{self.start_signal_path}:/run/asibench-start.sock:rw',
                 '--ulimit', 'core=0:0', '--ulimit', 'cpu=4800:4800']
        for key, value in {'PYTHONHASHSEED':'0', 'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4',
                           'MKL_NUM_THREADS':'4', 'NUMEXPR_NUM_THREADS':'4',
                           'TMPDIR':'/tmp/work/scratch'}.items():
            extra.extend(['-e', f'{key}={value}'])
        image_position = len(command) - len(spec.command) - 1
        if command[image_position] != kwargs['image']:
            raise RuntimeError('unsupported SDK submission-runner command layout')
        command[image_position:image_position] = extra
        return command


_REPLAY_BOOTSTRAP = '''\
import os,resource,runpy,shutil,socket,sys
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(8589934592,8589934592))
work=Path('/tmp/work')
work.mkdir(mode=0o700)
(work/'scratch').mkdir(mode=0o700)
(work/'data').symlink_to('/input',target_is_directory=True)
(work/'results').symlink_to('/output/results',target_is_directory=True)
shutil.copyfile('/submission/analysis.py',work/'analysis.py')
os.chdir(work)
os.environ['TMPDIR']=str(work/'scratch')
sys.argv=['analysis.py']
with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as signal:
    signal.settimeout(10.0)
    signal.connect('/run/asibench-start.sock')
    signal.sendall(__START_TOKEN__)
    acknowledgement=b''
    while len(acknowledgement)<2:
        chunk=signal.recv(2-len(acknowledgement))
        if not chunk:
            raise RuntimeError('execution handoff channel closed without acknowledgement')
        acknowledgement+=chunk
    if acknowledgement!=b'ok':
        raise RuntimeError('invalid execution handoff acknowledgement')
runpy.run_path(str(work/'analysis.py'),run_name='__main__')
'''


def _make_submission_runner(start_signal_path):
    return PredictionSubmissionRunner(resolve_runtime_root(Path(__file__).parent),
                                      start_signal_path=start_signal_path)


def _replay(pred_dir: Path, public: dict[str, bytes]) -> tuple[dict, np.ndarray | None]:
    info = dict(executed=False, execution_started=False, passed=False, timed_out=False,
                image_identity=None, regenerated_artifact_valid=False,
                isolation='docker_official_submission_runner')
    try:
        source = _regular_file(pred_dir / 'analysis.py', MAX_SOURCE_BYTES)
        compile(source, 'analysis.py', 'exec')
        density_error = _source_density_error(source)
    except (OSError, ValueError, SyntaxError, OverflowError, RecursionError, tokenize.TokenError) as exc:
        info.update(error=f'invalid or missing analysis.py: {exc}', failure_category='invalid_source')
        return info, None
    if density_error is not None:
        info.update(error=f'analysis.py literal-density limit: {density_error}',
                    failure_category='source_literal_density')
        return info, None
    with tempfile.TemporaryDirectory(prefix='asibench_prediction_replay_') as temp:
        submitted = Path(temp)/'submission'; submitted.mkdir()
        (submitted/'analysis.py').write_bytes(source)
        public_dir = Path(temp)/'public'; public_dir.mkdir()
        for name, raw in public.items():
            (public_dir/name).write_bytes(raw)
        try:
            start_signal = _StartSignal(temp)
            bootstrap = _REPLAY_BOOTSTRAP.replace('__START_TOKEN__', repr(start_signal.token))
            spec = SubmissionRunSpec(task_metadata=REPLAY_RUNTIME, submission_dir=submitted,
                command=('python', '-I', '-B', '-c', bootstrap),
                readonly_inputs=(ReadOnlyMount(public_dir.resolve(), '/input'),),
                output_files=('results/predictions.npy',), timeout_seconds=float(REPLAY_TIMEOUT_SECONDS),
                termination_grace_seconds=1., cpu_limit=4., memory_limit='8g', pids_limit=4096,
                max_output_file_bytes=MAX_OUTPUT_BYTES, log_tail_bytes=2000)
            with start_signal:
                result = _make_submission_runner(start_signal.path).run(spec)
        except Exception as exc:
            info.update(scorer_internal_error=True, failure_category='replay_infrastructure_error',
                        error=f'cannot prepare isolated Docker replay: {type(exc).__name__}: {exc}')
            return info, None
        started = start_signal.seen.is_set()
        info.update(executed=started, execution_started=started, returncode=result.exit_code,
                    execution_signal='host_unix_socket_ack', elapsed_seconds=result.elapsed_seconds,
                    image_identity=result.image_identity, timed_out=result.timed_out)
        if result.infrastructure_error or start_signal.error:
            info.update(scorer_internal_error=True, failure_category='replay_infrastructure_error',
                        error=result.infrastructure_error or start_signal.error)
            return info, None
        if not started:
            info.update(scorer_internal_error=True, failure_category='replay_failed_before_source_started',
                        error=f'replay ended before source handoff: exit {result.exit_code}')
            return info, None
        if result.timed_out:
            info.update(failure_category='submission_execution_timeout', error='replay exceeded1200seconds')
            return info, None
        if not result.launched or not result.image_identity:
            info.update(scorer_internal_error=True, failure_category='missing_container_provenance',
                        error='official replay container identity unavailable')
            return info, None
        if result.exit_code != 0:
            info.update(failure_category='submission_execution_failure',
                        error=f'isolated replay exit {result.exit_code}: {result.stderr_tail}')
            return info, None
        payload = result.output_bytes('results/predictions.npy')
        try:
            if payload is None or len(payload) > MAX_OUTPUT_BYTES:
                raise ValueError('missing or oversized regenerated prediction artifact')
            array = _npy_array(payload, PREDICTION_SHAPE)
        except (ValueError, TypeError, OverflowError, EOFError) as exc:
            info.update(failure_category='invalid_regenerated_prediction', error=str(exc))
            return info, None
        info.update(regenerated_artifact_valid=True, regenerated_shape=list(array.shape))
        return info, array


def _compare_predictions(submitted, regenerated) -> dict:
    with np.errstate(over='ignore', invalid='ignore'):
        difference = np.abs(submitted-regenerated)
        maximum = float(difference.max())
        passed = bool(np.allclose(submitted, regenerated, atol=REPLAY_ATOL, rtol=REPLAY_RTOL))
    return dict(passed=passed, max_abs_error=maximum if math.isfinite(maximum) else None,
                nonfinite_difference=not math.isfinite(maximum), atol=REPLAY_ATOL, rtol=REPLAY_RTOL,
                tolerance_definition='abs(submitted-regenerated)<=atol+rtol*abs(regenerated)')


@register_scorer(GATE_NAME)
class PredictionReplayConsistency(Scorer):
    def score(self, pred_dir: Path, ref_dir: Path, config: dict[str, Any]) -> ScoreDetail:
        pred_dir, ref_dir = Path(pred_dir), Path(ref_dir)
        details = {'atol':REPLAY_ATOL, 'rtol':REPLAY_RTOL}
        def finish(passed, message):
            details['passed'] = bool(passed)
            return ScoreDetail(scorer_name=GATE_NAME, score=float(passed), max_score=1.,
                               passed=bool(passed), details=details, message=message)
        try:
            meta = _reference_metadata(ref_dir, config)
            public = _immutable_public(ref_dir, meta)
        except Exception as exc:
            details.update(scorer_internal_error=True, failure_kind='missing_evaluator_input' if isinstance(exc, OSError) else 'evaluator_runtime_error', failure_category='invalid_immutable_public_data',
                           error=f'{type(exc).__name__}: {exc}')
            return finish(False, 'evaluator immutable-public validation failed')
        # No scientific reference NPZ or hidden targets are read by this gate.
        input_ok = True
        try:
            for name in PUBLIC_FILES:
                raw = _regular_file(pred_dir/'data'/name, MAX_PUBLIC_FILE_BYTES)
                if hashlib.sha256(raw).hexdigest() != meta['public_input_sha256'][name]:
                    raise ValueError(f'workspace input differs: {name}')
        except (OSError, ValueError) as exc:
            input_ok = False; details['data_integrity_error'] = str(exc)
        try:
            submitted = _prediction(pred_dir/'results/predictions.npy')
            details['submitted_artifact_valid'] = True
        except (OSError, ValueError, TypeError, OverflowError, EOFError) as exc:
            submitted = None; details.update(submitted_artifact_valid=False, submitted_error=str(exc))
        # Every valid source gets clean replay, even if submitted output/input is invalid.
        info, regenerated = _replay(pred_dir, public)
        details['reproducibility'] = info
        info.update(atol=REPLAY_ATOL, rtol=REPLAY_RTOL, max_abs_error=None)
        if info.get('scorer_internal_error'):
            details['scorer_internal_error'] = True
            details['failure_kind'] = 'evaluator_unavailable'
            return finish(False, 'replay infrastructure error; not scientific difficulty evidence')
        if regenerated is None:
            return finish(False, 'source did not regenerate a valid prediction artifact')
        if submitted is None:
            info.update(failure_category='invalid_submitted_prediction', passed=False)
            return finish(False, 'submitted prediction artifact is invalid')
        comparison = _compare_predictions(submitted, regenerated)
        info.update(comparison)
        if not input_ok:
            info.update(passed=False, failure_category='workspace_public_input_modified')
            return finish(False, 'workspace public inputs differ; clean replay was attempted')
        if not comparison['passed']:
            info['failure_category'] = 'prediction_replay_mismatch'
            return finish(False, 'submitted and regenerated arrays differ beyond public tolerance')
        return finish(True, 'fresh public-input replay matches submitted predictions within tolerance')
