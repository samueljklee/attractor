"""Execution environment abstraction for the Coding Agent Loop.

Provides a protocol for file and shell operations that can be backed by
different execution environments: local filesystem, Docker containers,
or Kubernetes pods. Tools call the environment instead of direct OS APIs.

Usage::

    from attractor_agent.environment import LocalEnvironment, DockerEnvironment

    # Default: runs on host
    env = LocalEnvironment()

    # Sandboxed: runs inside a Docker container
    env = DockerEnvironment(image="python:3.12-slim")
    await env.start()

    # Tools use the same interface either way
    content = await env.read_file("/workspace/main.py")
    await env.write_file("/workspace/out.py", "print('hello')")
    result = await env.exec_shell("python out.py")

Spec reference: coding-agent-loop-spec S4.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

# ------------------------------------------------------------------ #
# Shell result
# ------------------------------------------------------------------ #


@dataclass
class ShellResult:
    """Result of a shell command execution."""

    stdout: str
    stderr: str
    returncode: int

    @property
    def output(self) -> str:
        """Combined output formatted for LLM consumption."""
        parts: list[str] = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"STDERR:\n{self.stderr}")
        if self.returncode != 0:
            parts.append(f"Exit code: {self.returncode}")
        return "\n".join(parts) if parts else "(no output)"


# ------------------------------------------------------------------ #
# ExecutionEnvironment protocol
# ------------------------------------------------------------------ #


@runtime_checkable
class ExecutionEnvironment(Protocol):
    """Protocol for execution environments.

    Implementations provide file I/O and shell execution, abstracting
    whether operations run on the host, in Docker, or in K8s.
    """

    async def read_file(self, path: str) -> str:
        """Read a file's contents."""
        ...

    async def write_file(self, path: str, content: str) -> None:
        """Write content to a file, creating parent directories."""
        ...

    async def file_exists(self, path: str) -> bool:
        """Check if a file exists."""
        ...

    async def is_file(self, path: str) -> bool:
        """Check if path is a regular file (not directory)."""
        ...

    async def mkdir(self, path: str) -> None:
        """Create directory and parents."""
        ...

    async def exec_shell(
        self,
        command: str,
        timeout: int = 120,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ShellResult:
        """Execute a shell command."""
        ...

    async def glob(self, pattern: str, path: str = ".") -> list[str]:
        """Find files matching a glob pattern."""
        ...

    async def list_dir(self, path: str) -> list[str]:
        """List directory contents."""
        ...

    async def start(self) -> None:
        """Initialize the environment (e.g., start container)."""
        ...

    async def stop(self) -> None:
        """Clean up the environment (e.g., stop container)."""
        ...


# ------------------------------------------------------------------ #
# SIGTERM → SIGKILL escalation helper (Spec §9.4)
# ------------------------------------------------------------------ #


def _sigterm_sigkill(proc: subprocess.Popen[str]) -> None:
    """Send SIGTERM to process group, then SIGKILL after 2 s if still running.

    Spec §9.4: timed-out commands receive SIGTERM first for graceful
    shutdown, escalating to SIGKILL after 2 seconds.
    """
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return  # Process already gone
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except OSError:
            pass  # Exited between SIGTERM and SIGKILL
        proc.wait()


# ------------------------------------------------------------------ #
# LocalEnvironment -- direct host filesystem access
# ------------------------------------------------------------------ #


class LocalEnvironment:
    """Execution environment using the local filesystem.

    This is the default -- all operations run directly on the host.
    Behavior is identical to the pre-abstraction tool implementations.
    """

    async def read_file(self, path: str) -> str:
        file_path = Path(path).expanduser().resolve()
        return file_path.read_text(encoding="utf-8", errors="replace")

    async def write_file(self, path: str, content: str) -> None:
        file_path = Path(path).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    async def file_exists(self, path: str) -> bool:
        return Path(path).expanduser().resolve().exists()

    async def is_file(self, path: str) -> bool:
        return Path(path).expanduser().resolve().is_file()

    async def mkdir(self, path: str) -> None:
        Path(path).expanduser().resolve().mkdir(parents=True, exist_ok=True)

    async def exec_shell(
        self,
        command: str,
        timeout: int = 120,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ShellResult:
        cwd = working_dir or os.getcwd()
        shell_env = env or dict(os.environ)

        def _run() -> ShellResult:
            try:
                proc = subprocess.Popen(  # noqa: S603
                    ["bash", "-c", command],  # noqa: S607
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd,
                    env=shell_env,
                    start_new_session=True,
                )
            except OSError as e:
                return ShellResult(
                    stdout="",
                    stderr=f"Error: {e}",
                    returncode=-1,
                )

            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Spec §9.4: SIGTERM → wait 2 s → SIGKILL
                _sigterm_sigkill(proc)
                stdout, stderr = proc.communicate()
                return ShellResult(
                    stdout=stdout or "",
                    stderr=f"Command timed out after {timeout}s",
                    returncode=-1,
                )

            return ShellResult(
                stdout=stdout,
                stderr=stderr,
                returncode=proc.returncode,
            )

        return await asyncio.to_thread(_run)

    async def glob(self, pattern: str, path: str = ".") -> list[str]:
        search_path = Path(path).expanduser().resolve()
        skip_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            ".tox",
            ".mypy_cache",
            ".pytest_cache",
            "build",
            "dist",
        }
        results: list[str] = []
        for match in sorted(search_path.glob(pattern)):
            if skip_dirs & set(match.parts):
                continue
            rel = match.relative_to(search_path)
            suffix = "/" if match.is_dir() else ""
            results.append(f"{rel}{suffix}")
        return results

    async def list_dir(self, path: str) -> list[str]:
        p = Path(path).expanduser().resolve()
        return sorted(str(f.name) for f in p.iterdir())

    async def start(self) -> None:
        pass  # No-op for local

    async def stop(self) -> None:
        pass  # No-op for local


# ------------------------------------------------------------------ #
# DockerEnvironment -- containerized execution
# ------------------------------------------------------------------ #


class DockerEnvironment:
    """Execution environment using a Docker container.

    All file I/O and shell commands run inside an isolated container.
    Requires `docker` CLI to be installed on the host.

    Usage::

        env = DockerEnvironment(image="python:3.12-slim")
        await env.start()  # creates and starts container

        content = await env.read_file("/workspace/file.py")
        await env.write_file("/workspace/out.py", code)
        result = await env.exec_shell("python /workspace/out.py")

        await env.stop()  # removes container
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        workspace: str = "/workspace",
        name: str | None = None,
    ) -> None:
        self._image = image
        self._workspace = workspace
        self._name = name
        self._container_id: str | None = None

    @property
    def container_id(self) -> str | None:
        return self._container_id

    @property
    def is_running(self) -> bool:
        return self._container_id is not None

    async def __aenter__(self) -> DockerEnvironment:
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """Create and start the Docker container."""
        if self._container_id:
            return

        cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "-w",
            self._workspace,
        ]
        if self._name:
            cmd.extend(["--name", self._name])
        cmd.extend([self._image, "sleep", "infinity"])

        result = await self._run_host_command(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start Docker container: {result.stderr.strip()}")
        self._container_id = result.stdout.strip()[:12]

        # Create workspace directory
        await self._docker_exec(f"mkdir -p {self._workspace}")

    async def stop(self) -> None:
        """Stop and remove the container."""
        if not self._container_id:
            return
        await self._run_host_command(["docker", "stop", self._container_id])
        self._container_id = None

    async def read_file(self, path: str) -> str:
        self._check_running()
        result = await self._docker_exec(f"cat {self._quote(path)}")
        if result.returncode != 0:
            if "No such file" in result.stderr:
                raise FileNotFoundError(f"File not found: {path}")
            raise OSError(f"Failed to read {path}: {result.stderr}")
        return result.stdout

    async def write_file(self, path: str, content: str) -> None:
        self._check_running()
        # Create parent directory
        parent = str(Path(path).parent)
        await self._docker_exec(f"mkdir -p {self._quote(parent)}")
        # Write via stdin pipe
        result = await self._run_host_command(
            [
                "docker",
                "exec",
                "-i",
                self._container_id or "",
                "sh",
                "-c",
                f"cat > {self._quote(path)}",
            ],
            input_data=content,
        )
        if result.returncode != 0:
            raise OSError(f"Failed to write {path}: {result.stderr}")

    async def file_exists(self, path: str) -> bool:
        self._check_running()
        result = await self._docker_exec(f"test -e {self._quote(path)} && echo yes || echo no")
        return result.stdout.strip() == "yes"

    async def is_file(self, path: str) -> bool:
        self._check_running()
        result = await self._docker_exec(f"test -f {self._quote(path)} && echo yes || echo no")
        return result.stdout.strip() == "yes"

    async def mkdir(self, path: str) -> None:
        self._check_running()
        await self._docker_exec(f"mkdir -p {self._quote(path)}")

    async def exec_shell(
        self,
        command: str,
        timeout: int = 120,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ShellResult:
        self._check_running()
        cwd = working_dir or self._workspace

        cmd_parts = ["docker", "exec"]
        if env:
            for k, v in env.items():
                cmd_parts.extend(["-e", f"{k}={v}"])
        cmd_parts.extend(
            [
                "-w",
                cwd,
                self._container_id,  # type: ignore[arg-type]
                "bash",
                "-c",
                command,
            ]
        )

        return await self._run_host_command(cmd_parts, timeout=timeout)

    async def glob(self, pattern: str, path: str = ".") -> list[str]:
        self._check_running()
        # Use find + fnmatch pattern in the container
        search = path if path != "." else self._workspace
        result = await self._docker_exec(
            f"find {self._quote(search)} -name {self._quote(pattern)} "
            f"-type f 2>/dev/null | head -500"
        )
        if not result.stdout.strip():
            return []
        return sorted(result.stdout.strip().split("\n"))

    async def list_dir(self, path: str) -> list[str]:
        self._check_running()
        result = await self._docker_exec(f"ls -1 {self._quote(path)}")
        if result.returncode != 0:
            raise OSError(f"Failed to list {path}: {result.stderr}")
        return sorted(result.stdout.strip().split("\n")) if result.stdout.strip() else []

    def _check_running(self) -> None:
        if not self._container_id:
            raise RuntimeError("Docker container not running. Call start() first.")

    @staticmethod
    def _quote(s: str) -> str:
        """Shell-quote a string for use in docker exec commands."""
        import shlex

        return shlex.quote(s)

    async def _docker_exec(self, command: str) -> ShellResult:
        """Run a command inside the container."""
        return await self._run_host_command(
            [
                "docker",
                "exec",
                self._container_id,  # type: ignore[list-item]
                "bash",
                "-c",
                command,
            ]
        )

    @staticmethod
    async def _run_host_command(
        cmd: list[str],
        timeout: int = 120,
        input_data: str | None = None,
    ) -> ShellResult:
        """Run a command on the host (docker CLI)."""

        def _run() -> ShellResult:
            try:
                result = subprocess.run(  # noqa: S603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    input=input_data,
                )
            except subprocess.TimeoutExpired:
                return ShellResult(
                    stdout="",
                    stderr=f"Command timed out after {timeout}s",
                    returncode=-1,
                )
            except FileNotFoundError:
                return ShellResult(
                    stdout="",
                    stderr="docker CLI not found. Install Docker to use DockerEnvironment.",
                    returncode=-1,
                )
            return ShellResult(
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
            )

        return await asyncio.to_thread(_run)
