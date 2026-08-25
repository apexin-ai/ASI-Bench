"""Isolated second-stage execution of untrusted task submissions."""

from __future__ import annotations

import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping

from ai4sci_bench.runner.task_image import TaskImageBuilder


@dataclass(frozen=True)
class ReadOnlyMount:
    """One evaluator-owned host path exposed read-only in the container."""

    host_path: Path
    container_path: str


@dataclass(frozen=True)
class SubmissionRunSpec:
    """Complete, explicit resource and artifact contract for one execution."""

    task_metadata: Mapping[str, Any]
    submission_dir: Path
    command: tuple[str, ...]
    readonly_inputs: tuple[ReadOnlyMount, ...]
    output_files: tuple[str, ...]
    timeout_seconds: float = 300.0
    termination_grace_seconds: float = 10.0
    cpu_limit: float = 1.0
    memory_limit: str = "3g"
    pids_limit: int = 4096
    max_output_file_bytes: int = 64 * 1024 * 1024
    log_tail_bytes: int = 64 * 1024


@dataclass(frozen=True)
class SubmissionRunResult:
    """Immutable outcome and allowlisted output bytes from one execution."""

    launched: bool
    timed_out: bool
    exit_code: int | None
    elapsed_seconds: float
    stdout_tail: str
    stderr_tail: str
    output_artifacts: tuple[tuple[str, bytes], ...]
    image_identity: str | None
    infrastructure_error: str | None

    def output_bytes(self, name: str) -> bytes | None:
        """Return one allowlisted artifact, if the runner accepted it."""
        for artifact_name, data in self.output_artifacts:
            if artifact_name == name:
                return data
        return None


class _BoundedTail:
    """Keep only a fixed-size byte suffix while a stream is being drained."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._data = bytearray()

    def append(self, data: bytes) -> None:
        if self.limit <= 0 or not data:
            return
        if len(data) >= self.limit:
            self._data[:] = data[-self.limit :]
            return
        overflow = len(self._data) + len(data) - self.limit
        if overflow > 0:
            del self._data[:overflow]
        self._data.extend(data)

    def text(self) -> str:
        return bytes(self._data).decode("utf-8", errors="replace")


def _drain_pipe(pipe: BinaryIO, tail: _BoundedTail) -> None:
    try:
        while True:
            chunk = pipe.read(64 * 1024)
            if not chunk:
                return
            tail.append(chunk)
    finally:
        try:
            pipe.close()
        except OSError:
            pass


class SandboxedSubmissionRunner:
    """Run an untrusted program without agent credentials or network access."""

    _TMPFS_SIZE = "1g"
    _UNTRUSTED_UID = 65534
    _UNTRUSTED_GID = 65534

    def __init__(
        self,
        repo_root: Path,
        *,
        image_builder: TaskImageBuilder | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.image_builder = image_builder or TaskImageBuilder(self.repo_root)

    def run(self, spec: SubmissionRunSpec) -> SubmissionRunResult:
        """Execute one submission and return only explicitly allowed artifacts."""
        validated = self._validate_spec(spec)
        submission_dir, readonly_inputs, output_files = validated
        started = time.monotonic()

        try:
            image = self.image_builder.ensure_image(
                dict(spec.task_metadata),
                agent_type=None,
            )
            image_identity = self.image_builder.get_image_identity(image)
        except Exception as exc:
            return self._infrastructure_failure(started, str(exc))

        container_name = f"ai4sci-submission-{uuid.uuid4().hex[:12]}"
        launched = False
        timed_out = False
        exit_code: int | None = None
        infrastructure_error: str | None = None

        with tempfile.TemporaryDirectory(prefix="ai4sci_submission_") as temp_name:
            temp_dir = Path(temp_name)
            staged_submission_dir = temp_dir / "submission"
            output_dir = temp_dir / "output"
            try:
                self._stage_submission(submission_dir, staged_submission_dir)
                staged_readonly_inputs = self._stage_readonly_inputs(
                    readonly_inputs,
                    temp_dir / "inputs",
                )
            except Exception as exc:
                return self._infrastructure_failure(
                    started,
                    f"sandbox staging failed: {exc}",
                )
            self._prepare_output_dir(output_dir, output_files)
            stdout_buffer = _BoundedTail(spec.log_tail_bytes)
            stderr_buffer = _BoundedTail(spec.log_tail_bytes)
            command = self._build_docker_command(
                spec,
                image=image,
                container_name=container_name,
                submission_dir=staged_submission_dir,
                readonly_inputs=staged_readonly_inputs,
                output_dir=output_dir,
            )

            process: subprocess.Popen[bytes] | None = None
            drain_threads: list[threading.Thread] = []
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                launched = True
                for pipe, buffer in (
                    (process.stdout, stdout_buffer),
                    (process.stderr, stderr_buffer),
                ):
                    if pipe is None:
                        continue
                    thread = threading.Thread(
                        target=_drain_pipe,
                        args=(pipe, buffer),
                        daemon=True,
                    )
                    thread.start()
                    drain_threads.append(thread)
                try:
                    exit_code = process.wait(timeout=spec.timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._control_container(
                        [
                            "docker",
                            "stop",
                            "--time",
                            str(int(spec.termination_grace_seconds)),
                            container_name,
                        ],
                        timeout=spec.termination_grace_seconds + 5.0,
                    )
                    try:
                        exit_code = process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        self._control_container(
                            ["docker", "kill", container_name],
                            timeout=5.0,
                        )
                        try:
                            exit_code = process.wait(timeout=5.0)
                        except subprocess.TimeoutExpired:
                            infrastructure_error = (
                                "container process did not terminate after forced kill"
                            )
                except Exception as exc:
                    infrastructure_error = str(exc)
            except Exception as exc:
                infrastructure_error = str(exc)
            finally:
                self._control_container(
                    ["docker", "rm", "-f", container_name],
                    timeout=10.0,
                )
                for thread in drain_threads:
                    thread.join(timeout=5.0)

            artifacts = self._collect_outputs(
                output_dir,
                output_files,
                spec.max_output_file_bytes,
            )
            self._restore_output_permissions(output_dir)
            stdout_tail = stdout_buffer.text()
            stderr_tail = stderr_buffer.text()

        return SubmissionRunResult(
            launched=launched,
            timed_out=timed_out,
            exit_code=exit_code,
            elapsed_seconds=time.monotonic() - started,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            output_artifacts=artifacts,
            image_identity=image_identity,
            infrastructure_error=infrastructure_error,
        )

    def _validate_spec(
        self,
        spec: SubmissionRunSpec,
    ) -> tuple[Path, tuple[ReadOnlyMount, ...], tuple[str, ...]]:
        try:
            submission_dir = Path(spec.submission_dir).resolve(strict=True)
        except OSError as exc:
            raise ValueError(
                f"submission directory does not exist: {spec.submission_dir}"
            ) from exc
        if not submission_dir.is_dir():
            raise ValueError("submission directory must be a directory")
        if not spec.command or any(
            not isinstance(part, str) or not part or "\x00" in part
            for part in spec.command
        ):
            raise ValueError("submission command must contain nonempty strings")
        if spec.timeout_seconds <= 0 or spec.termination_grace_seconds < 0:
            raise ValueError("submission timeout values must be positive")
        if spec.cpu_limit <= 0:
            raise ValueError("submission CPU limit must be positive")
        if not spec.memory_limit or "\x00" in spec.memory_limit:
            raise ValueError("submission memory limit must be nonempty")
        if spec.pids_limit <= 0:
            raise ValueError("submission process limit must be positive")
        if spec.max_output_file_bytes <= 0 or spec.log_tail_bytes < 0:
            raise ValueError("submission artifact limits must be nonnegative")

        validated_mounts: list[ReadOnlyMount] = []
        targets = {"/submission", "/output"}
        for mount in spec.readonly_inputs:
            try:
                host_path = Path(mount.host_path).resolve(strict=True)
            except OSError as exc:
                raise ValueError(
                    f"read-only mount does not exist: {mount.host_path}"
                ) from exc
            container_path = self._validate_container_path(mount.container_path)
            if container_path in targets:
                raise ValueError(f"duplicate container path: {container_path}")
            targets.add(container_path)
            validated_mounts.append(ReadOnlyMount(host_path, container_path))

        validated_outputs: list[str] = []
        for name in spec.output_files:
            if not isinstance(name, str):
                raise ValueError("output file names must be strings")
            path = PurePosixPath(name)
            if (
                not name
                or path.is_absolute()
                or any(part in ("", ".", "..") for part in path.parts)
            ):
                raise ValueError(f"invalid output file path: {name!r}")
            validated_outputs.append(path.as_posix())
        if len(validated_outputs) != len(set(validated_outputs)):
            raise ValueError("output file allowlist contains duplicates")

        return (
            submission_dir,
            tuple(validated_mounts),
            tuple(validated_outputs),
        )

    @staticmethod
    def _validate_container_path(value: str) -> str:
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError("container path must be a string")
        path = PurePosixPath(value)
        if (
            not path.is_absolute()
            or str(path) == "/"
            or any(part in (".", "..") for part in path.parts)
        ):
            raise ValueError(f"invalid container path: {value!r}")
        return path.as_posix()

    def _build_docker_command(
        self,
        spec: SubmissionRunSpec,
        *,
        image: str,
        container_name: str,
        submission_dir: Path,
        readonly_inputs: tuple[ReadOnlyMount, ...],
        output_dir: Path,
    ) -> list[str]:
        command = [
            "docker",
            "run",
            "--name",
            container_name,
            "--rm",
            "--network",
            "none",
            "--log-driver",
            "local",
            "--log-opt",
            "max-size=1m",
            "--log-opt",
            "max-file=1",
            "--log-opt",
            "compress=false",
            "--cpus",
            str(float(spec.cpu_limit)),
            "--memory",
            spec.memory_limit,
            "--pids-limit",
            str(spec.pids_limit),
            "--ulimit",
            (f"fsize={spec.max_output_file_bytes}:{spec.max_output_file_bytes}"),
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            f"/tmp:rw,exec,nosuid,nodev,size={self._TMPFS_SIZE},mode=1777",
            "--tmpfs",
            f"/home/agent:rw,nosuid,nodev,size={self._TMPFS_SIZE},mode=1777",
            "--shm-size",
            "256m",
            "-e",
            "HOME=/home/agent",
            "-e",
            "LANG=C.UTF-8",
            "-e",
            "PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            "TMPDIR=/tmp",
            "-w",
            "/submission",
        ]
        command.extend(["--user", f"{self._UNTRUSTED_UID}:{self._UNTRUSTED_GID}"])
        command.extend(["-v", f"{submission_dir}:/submission:ro"])
        for mount in readonly_inputs:
            command.extend(["-v", f"{mount.host_path}:{mount.container_path}:ro"])
        command.extend(["-v", f"{output_dir}:/output:rw"])
        command.append(image)
        command.extend(spec.command)
        return command

    @staticmethod
    def _stage_submission(source_dir: Path, destination_dir: Path) -> None:
        """Create an immutable, sandbox-readable copy without mutating the source.

        Persisted workspaces are intentionally private (typically 0700/0600),
        while submission containers run as the untrusted 65534 uid.  A bind
        mount preserves host mode bits, so mounting the workspace directly can
        make a valid delivery unreadable.  Normalize only the evaluator-owned
        staging copy and preserve whether regular files were executable.
        """
        destination_dir.mkdir(mode=0o755)
        for source in sorted(source_dir.rglob("*")):
            relative = source.relative_to(source_dir)
            destination = destination_dir / relative
            source_stat = source.lstat()
            if stat.S_ISLNK(source_stat.st_mode):
                raise ValueError(f"submission contains a symlink: {relative}")
            if stat.S_ISDIR(source_stat.st_mode):
                destination.mkdir(mode=0o755)
                continue
            if not stat.S_ISREG(source_stat.st_mode):
                raise ValueError(f"submission contains a special file: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            shutil.copyfile(source, destination, follow_symlinks=False)
            destination.chmod(0o555 if source_stat.st_mode & 0o111 else 0o444)

        directories = [destination_dir]
        directories.extend(
            path
            for path in destination_dir.rglob("*")
            if path.is_dir() and not path.is_symlink()
        )
        for directory in sorted(
            directories,
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o555)

    @classmethod
    def _stage_readonly_inputs(
        cls,
        readonly_inputs: tuple[ReadOnlyMount, ...],
        staging_root: Path,
    ) -> tuple[ReadOnlyMount, ...]:
        """Copy evaluator inputs with modes readable by the untrusted uid."""
        staging_root.mkdir(mode=0o700)
        staged: list[ReadOnlyMount] = []
        for index, mount in enumerate(readonly_inputs):
            source = mount.host_path
            destination = staging_root / f"input_{index}"
            source_stat = source.lstat()
            if stat.S_ISDIR(source_stat.st_mode):
                cls._stage_submission(source, destination)
            elif stat.S_ISREG(source_stat.st_mode):
                shutil.copyfile(source, destination, follow_symlinks=False)
                destination.chmod(0o555 if source_stat.st_mode & 0o111 else 0o444)
            else:
                raise ValueError(
                    f"read-only input is not a regular file or directory: {source}"
                )
            staged.append(ReadOnlyMount(destination, mount.container_path))
        return tuple(staged)

    @staticmethod
    def _prepare_output_dir(
        output_dir: Path,
        output_files: tuple[str, ...],
    ) -> None:
        output_dir.mkdir(mode=0o755)
        for name in output_files:
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(mode=0o666)
            path.chmod(0o666)
        directories = [output_dir]
        directories.extend(
            path
            for path in output_dir.rglob("*")
            if path.is_dir() and not path.is_symlink()
        )
        for directory in directories:
            directory.chmod(0o555)

    @staticmethod
    def _restore_output_permissions(output_dir: Path) -> None:
        for path in output_dir.rglob("*"):
            if path.is_symlink():
                continue
            try:
                path.chmod(0o700 if path.is_dir() else 0o600)
            except OSError:
                continue
        try:
            output_dir.chmod(0o700)
        except OSError:
            pass

    @staticmethod
    def _control_container(command: list[str], *, timeout: float) -> bool:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=max(1.0, timeout),
                check=False,
            )
        except Exception:
            return False
        return result.returncode == 0

    @staticmethod
    def _collect_outputs(
        output_dir: Path,
        output_files: tuple[str, ...],
        max_bytes: int,
    ) -> tuple[tuple[str, bytes], ...]:
        artifacts: list[tuple[str, bytes]] = []
        output_root = output_dir.resolve()
        for name in output_files:
            path = output_dir / name
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(output_root):
                    continue
                size = resolved.stat().st_size
                if size > max_bytes:
                    continue
                artifacts.append((name, resolved.read_bytes()))
            except OSError:
                continue
        return tuple(artifacts)

    @staticmethod
    def _infrastructure_failure(
        started: float,
        message: str,
    ) -> SubmissionRunResult:
        return SubmissionRunResult(
            launched=False,
            timed_out=False,
            exit_code=None,
            elapsed_seconds=time.monotonic() - started,
            stdout_tail="",
            stderr_tail="",
            output_artifacts=(),
            image_identity=None,
            infrastructure_error=message,
        )
