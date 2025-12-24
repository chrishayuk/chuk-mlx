"""
Telnet client for puzzle arcade server.

Provides async telnet connection to the puzzle arcade server,
handling JSON mode communication and puzzle episode streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# =============================================================================
# Telnet Protocol Constants
# =============================================================================

# Telnet command bytes
IAC = 255  # Interpret As Command
DONT = 254
DO = 253
WONT = 252
WILL = 251
SB = 250  # Subnegotiation Begin
SE = 240  # Subnegotiation End

# =============================================================================
# Enums
# =============================================================================


class PuzzleGame(str, Enum):
    """Available puzzle games."""

    SUDOKU = "sudoku"
    KENKEN = "kenken"
    KAKURO = "kakuro"
    BINARY = "binary"
    FUTOSHIKI = "futoshiki"
    NONOGRAM = "nonogram"
    LOGIC_GRID = "logic_grid"
    KILLER_SUDOKU = "killer_sudoku"
    LIGHTS_OUT = "lights_out"
    MASTERMIND = "mastermind"
    SLITHERLINK = "slitherlink"
    BRIDGES = "bridges"
    HITORI = "hitori"
    SHIKAKU = "shikaku"
    HIDATO = "hidato"
    TENTS = "tents"
    FILLOMINO = "fillomino"
    STAR_BATTLE = "star_battle"
    SOKOBAN = "sokoban"
    KNAPSACK = "knapsack"
    NURIKABE = "nurikabe"
    MINESWEEPER = "minesweeper"


class PuzzleDifficulty(str, Enum):
    """Puzzle difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# =============================================================================
# Response Models
# =============================================================================


class DifficultyProfile(BaseModel):
    """Difficulty characteristics of a puzzle."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    logic_depth: int = Field(default=1, description="Reasoning complexity 1-10")
    branching_factor: float = Field(default=1.0, description="Solution space size 1-10")
    state_observability: float = Field(default=1.0, description="Visible info 0-1")
    constraint_density: float = Field(default=0.5, description="Constraint saturation 0-1")


class PuzzleObservation(BaseModel):
    """Observation from puzzle server."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    type: str = Field(description="Message type (observation, result, complete)")
    game: str = Field(description="Game identifier")
    difficulty: str = Field(description="Difficulty level")
    seed: int = Field(default=0, description="Puzzle seed")
    moves: int = Field(default=0, description="Moves made")
    invalid_moves: int = Field(default=0, description="Invalid moves")
    hints_used: int = Field(default=0, description="Hints consumed")
    hints_remaining: int = Field(default=5, description="Hints left")
    optimal_steps: int | None = Field(default=None, description="Optimal solution length")
    is_complete: bool = Field(default=False, description="Whether puzzle is solved")
    grid: list[list[int]] | None = Field(default=None, description="Current grid state")
    grid_display: str | None = Field(default=None, description="ASCII grid display")
    difficulty_profile: DifficultyProfile | None = Field(default=None)


class PuzzleResult(BaseModel):
    """Result of an action."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    type: str = Field(description="Message type")
    success: bool = Field(description="Whether action succeeded")
    code: str | None = Field(default=None, description="Result code")
    message: str | None = Field(default=None, description="Result message")
    state: PuzzleObservation | None = Field(default=None, description="Updated state")


class PuzzleCompletion(BaseModel):
    """Puzzle completion message."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    type: str = Field(description="Message type (complete)")
    success: bool = Field(description="Whether solved successfully")
    game: str = Field(description="Game identifier")
    moves: int = Field(default=0, description="Total moves")
    invalid_moves: int = Field(default=0, description="Invalid moves")
    hints_used: int = Field(default=0, description="Hints used")
    optimal_steps: int | None = Field(default=None, description="Optimal steps")


# =============================================================================
# Client Configuration
# =============================================================================


class TelnetClientConfig(BaseModel):
    """Configuration for telnet client."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = Field(default="localhost", description="Server host")
    port: int = Field(default=8023, description="Server port")
    connect_timeout: float = Field(default=10.0, description="Connection timeout")
    read_timeout: float = Field(default=30.0, description="Read timeout")
    max_retries: int = Field(default=3, description="Max connection retries")
    retry_delay: float = Field(default=1.0, description="Delay between retries")


# =============================================================================
# Telnet Client
# =============================================================================


@dataclass
class TelnetGymClient:
    """
    Async telnet client for puzzle arcade server.

    Handles connection management, JSON mode, and puzzle commands.

    Example:
        async with TelnetGymClient(config) as client:
            # Start a puzzle
            obs = await client.start_puzzle(PuzzleGame.SUDOKU, PuzzleDifficulty.EASY)

            # Make moves
            result = await client.send_command("place 1 5 7")

            # Get hints
            hint = await client.send_command("hint")
    """

    config: TelnetClientConfig = field(default_factory=TelnetClientConfig)
    _reader: asyncio.StreamReader | None = field(default=None, init=False, repr=False)
    _writer: asyncio.StreamWriter | None = field(default=None, init=False, repr=False)
    _connected: bool = field(default=False, init=False)
    _json_mode: bool = field(default=False, init=False)
    _buffer: str = field(default="", init=False, repr=False)

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> None:
        """Connect to the puzzle server."""
        if self._connected:
            return

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                logger.info(
                    f"Connecting to {self.config.host}:{self.config.port} "
                    f"(attempt {attempt + 1}/{self.config.max_retries})"
                )

                self._reader, self._writer = await asyncio.wait_for(
                    asyncio.open_connection(self.config.host, self.config.port),
                    timeout=self.config.connect_timeout,
                )

                self._connected = True
                logger.info("Connected to puzzle server")

                # Read initial banner/menu (may take a moment)
                await self._drain_initial_output()

                return

            except (OSError, asyncio.TimeoutError) as e:
                last_error = e
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)

        raise ConnectionError(
            f"Failed to connect after {self.config.max_retries} attempts: {last_error}"
        )

    async def disconnect(self) -> None:
        """Disconnect from the server."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        self._reader = None
        self._writer = None
        self._connected = False
        self._json_mode = False
        logger.info("Disconnected from puzzle server")

    async def __aenter__(self) -> TelnetGymClient:
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Whether client is connected."""
        return self._connected

    # =========================================================================
    # Telnet Protocol Handling
    # =========================================================================

    def _strip_telnet_commands(self, data: bytes) -> bytes:
        """Strip telnet negotiation commands from raw data."""
        result = bytearray()
        i = 0
        while i < len(data):
            if data[i] == IAC:
                if i + 1 < len(data):
                    cmd = data[i + 1]
                    if cmd in (DO, DONT, WILL, WONT):
                        # 3-byte command: IAC + cmd + option
                        if i + 2 < len(data):
                            # Respond with refusal for DO/WILL
                            option = data[i + 2]
                            if cmd == DO and self._writer:
                                # Respond WONT
                                self._writer.write(bytes([IAC, WONT, option]))
                            elif cmd == WILL and self._writer:
                                # Respond DONT
                                self._writer.write(bytes([IAC, DONT, option]))
                            i += 3
                            continue
                    elif cmd == SB:
                        # Skip subnegotiation until SE
                        j = i + 2
                        while j < len(data) - 1:
                            if data[j] == IAC and data[j + 1] == SE:
                                i = j + 2
                                break
                            j += 1
                        else:
                            i = len(data)
                        continue
                    elif cmd == IAC:
                        # Escaped IAC (0xFF 0xFF = literal 0xFF)
                        result.append(IAC)
                        i += 2
                        continue
                    else:
                        # Other 2-byte commands
                        i += 2
                        continue
                i += 1
            else:
                result.append(data[i])
                i += 1
        return bytes(result)

    async def _read_raw_line(self) -> bytes:
        """Read raw bytes until newline."""
        if not self._reader:
            raise ConnectionError("Not connected")

        try:
            line = await asyncio.wait_for(
                self._reader.readline(),
                timeout=self.config.read_timeout,
            )
            return line
        except asyncio.TimeoutError as e:
            raise TimeoutError("Read timeout") from e

    async def _drain_initial_output(self) -> None:
        """Read and discard initial server output (banner, menu, etc.)."""
        if not self._reader:
            return

        # Wait a moment for the server to send initial data
        await asyncio.sleep(0.2)

        # Keep reading until we see the prompt and data stops
        total_drained = 0

        while True:
            try:
                raw = await asyncio.wait_for(
                    self._reader.read(4096),
                    timeout=1.0,  # Longer timeout to catch full menu
                )
                if not raw:
                    break
                # Strip telnet commands
                cleaned = self._strip_telnet_commands(raw)
                total_drained += len(cleaned)

                # Check if we've seen the prompt ("> " at end)
                try:
                    text = cleaned.decode("utf-8", errors="ignore")
                    if text.rstrip().endswith(">"):
                        logger.debug(f"Saw prompt, total drained: {total_drained} bytes")
                        break
                except Exception:
                    pass

            except asyncio.TimeoutError:
                break

        if total_drained:
            logger.debug(f"Drained {total_drained} bytes of initial output")

    async def _drain_text_output(self) -> None:
        """Read and discard text output until quiet."""
        await self._read_until_prompt()

    # =========================================================================
    # Low-Level Communication
    # =========================================================================

    async def _send(self, command: str) -> None:
        """Send a command to the server."""
        if not self._writer:
            raise ConnectionError("Not connected")

        logger.debug(f"Sending: {command}")
        self._writer.write(f"{command}\r\n".encode())
        await self._writer.drain()

    async def _read_line(self) -> str:
        """Read a single line from the server, stripping telnet commands."""
        raw = await self._read_raw_line()
        cleaned = self._strip_telnet_commands(raw)
        try:
            decoded = cleaned.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError:
            # Fall back to latin-1 if UTF-8 fails
            decoded = cleaned.decode("latin-1").rstrip("\r\n")
        logger.debug(f"Received: {decoded[:100]}...")
        return decoded

    async def _read_until_prompt(self) -> list[str]:
        """Read lines until we see a prompt or empty line."""
        lines = []
        while True:
            try:
                raw = await asyncio.wait_for(
                    self._reader.readline(),
                    timeout=2.0,  # Short timeout for prompt detection
                )
                cleaned = self._strip_telnet_commands(raw)
                try:
                    decoded = cleaned.decode("utf-8").rstrip("\r\n")
                except UnicodeDecodeError:
                    decoded = cleaned.decode("latin-1").rstrip("\r\n")
                if not decoded:
                    break
                lines.append(decoded)
                # Check for common prompts
                if decoded.endswith(">") or decoded.endswith(":"):
                    break
            except asyncio.TimeoutError:
                break
        return lines

    async def _read_json_response(self) -> dict:
        """Read a JSON response from the server."""
        if not self._reader:
            raise ConnectionError("Not connected")

        # Read lines until we get a complete JSON object
        buffer = ""
        brace_count = 0
        started = False

        while True:
            line = await self._read_line()

            # Skip empty lines before JSON
            if not line and not started:
                continue

            # Look for JSON start
            if "{" in line:
                started = True

            if started:
                buffer += line

                # Count braces
                brace_count += line.count("{") - line.count("}")

                # Complete JSON object
                if brace_count == 0 and buffer.strip():
                    try:
                        return json.loads(buffer)
                    except json.JSONDecodeError:
                        # Keep reading
                        continue

    async def _enable_json_mode(self) -> None:
        """Switch server to JSON output mode (must be inside a game)."""
        if self._json_mode:
            return

        await self._send("mode json")

        # Read JSON acknowledgment
        try:
            response = await self._read_json_response()
            if response.get("success"):
                logger.info("JSON mode enabled")
                self._json_mode = True
                return
        except Exception as e:
            logger.warning(f"Failed to parse mode json response: {e}")

        # Try reading text acknowledgment as fallback
        lines = await self._read_until_prompt()
        for line in lines:
            if "json" in line.lower() or "mode" in line.lower():
                logger.info("JSON mode enabled via text response")
                self._json_mode = True
                return

        # Assume it worked even without explicit confirmation
        self._json_mode = True
        logger.info("JSON mode assumed enabled")

    # =========================================================================
    # Puzzle Commands
    # =========================================================================

    async def start_puzzle(
        self,
        game: PuzzleGame | str,
        difficulty: PuzzleDifficulty | str = PuzzleDifficulty.EASY,
        seed: int | None = None,
    ) -> PuzzleObservation:
        """
        Start a new puzzle.

        Args:
            game: Puzzle game to play
            difficulty: Difficulty level
            seed: Optional seed for reproducibility

        Returns:
            Initial puzzle observation
        """
        if not self._connected:
            await self.connect()

        # Build command
        game_str = game.value if isinstance(game, PuzzleGame) else game
        diff_str = difficulty.value if isinstance(difficulty, PuzzleDifficulty) else difficulty

        # Build command with optional seed
        if seed is not None:
            command = f"{game_str} {diff_str} {seed}"
        else:
            command = f"{game_str} {diff_str}"

        # Start the game (response is in text mode initially)
        await self._send(command)

        # Drain the text-based game intro
        await self._drain_text_output()

        # Switch to JSON mode now that we're in a game
        self._json_mode = False
        await self._enable_json_mode()

        # Get current state in JSON format
        await self._send("show")
        response = await self._read_json_response()

        return PuzzleObservation(**response)

    async def send_action(self, action: str) -> PuzzleResult:
        """
        Send an action to the current puzzle.

        Args:
            action: Action command (e.g., "place 1 5 7")

        Returns:
            Result of the action
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        await self._send(action)
        response = await self._read_json_response()

        # Check response type
        response_type = response.get("type", "result")

        if response_type == "complete":
            # Puzzle completed
            completion = PuzzleCompletion(**response)
            return PuzzleResult(
                type="complete",
                success=completion.success,
                code="COMPLETE",
                message=f"Puzzle completed in {completion.moves} moves",
            )

        return PuzzleResult(**response)

    async def get_hint(self) -> PuzzleResult:
        """Get a hint for the current puzzle.

        Note: The hint command may return plain text instead of JSON
        depending on the server implementation. This method handles both.
        """
        if not self._connected:
            raise ConnectionError("Not connected")

        await self._send("hint")

        # Wait briefly for response
        await asyncio.sleep(0.2)

        # Read all available data
        buffer = b""
        while True:
            try:
                chunk = await asyncio.wait_for(
                    self._reader.read(4096),
                    timeout=1.0,
                )
                if not chunk:
                    break
                buffer += self._strip_telnet_commands(chunk)
                # Check if we have a complete response (prompt at end)
                try:
                    text = buffer.decode("utf-8", errors="ignore")
                    if text.rstrip().endswith(">"):
                        break
                except Exception:
                    pass
            except asyncio.TimeoutError:
                break

        # Decode and parse
        try:
            full_text = buffer.decode("utf-8", errors="ignore")
        except Exception:
            full_text = buffer.decode("latin-1", errors="ignore")

        # Clean up - remove echo and prompts
        lines = []
        for line in full_text.split("\n"):
            line = line.strip().rstrip("\r")
            if not line:
                continue
            # Skip command echo and prompt variations
            if line in ("hint", ">", "> hint"):
                continue
            if line.startswith(">"):
                line = line.lstrip("> ").strip()
                if not line:
                    continue
            lines.append(line)

        full_text = "\n".join(lines)

        # Try to parse as JSON first
        if full_text.strip().startswith("{"):
            try:
                response = json.loads(full_text)
                return PuzzleResult(**response)
            except (json.JSONDecodeError, Exception):
                pass

        # Parse as text hint
        hint_text = full_text.strip()
        if hint_text:
            # Extract the hint message
            if hint_text.startswith("Hint:"):
                hint_text = hint_text[5:].strip()
            return PuzzleResult(
                type="result",
                success=True,
                code="HINT",
                message=hint_text,
            )

        return PuzzleResult(
            type="result",
            success=False,
            code="NO_HINT",
            message="No hint available",
        )

    async def show_state(self) -> PuzzleObservation:
        """Get the current puzzle state."""
        await self._send("show")
        response = await self._read_json_response()
        return PuzzleObservation(**response)

    async def get_stats(self) -> dict:
        """Get current puzzle statistics."""
        await self._send("stats")
        response = await self._read_json_response()
        return response

    async def reset_puzzle(self) -> PuzzleObservation:
        """Reset the current puzzle to initial state."""
        await self._send("reset")
        response = await self._read_json_response()
        return PuzzleObservation(**response)

    async def quit_puzzle(self) -> None:
        """Quit the current puzzle and return to menu."""
        await self._send("quit")
        # Drain any output and reset JSON mode (menu is text-based)
        await self._drain_text_output()
        self._json_mode = False


# =============================================================================
# Episode Iterator
# =============================================================================


@dataclass
class PuzzleEpisode:
    """
    A complete puzzle episode for training.

    Contains the prompt (puzzle state), actions taken,
    and final outcome.
    """

    game: str
    difficulty: str
    seed: int
    initial_grid: list[list[int]] | None
    final_grid: list[list[int]] | None
    moves: list[str]
    success: bool
    total_moves: int
    invalid_moves: int
    hints_used: int
    optimal_steps: int | None
    difficulty_profile: DifficultyProfile | None

    def to_training_text(self, tokenizer) -> tuple[list[int], list[int]]:
        """
        Convert episode to training format.

        Returns:
            (input_ids, loss_mask) tuple
        """
        # Build prompt
        prompt_parts = [
            f"Game: {self.game}",
            f"Difficulty: {self.difficulty}",
            "Initial state:",
            self._format_grid(self.initial_grid) if self.initial_grid else "[unknown]",
            "\nSolve this puzzle step by step:",
        ]
        prompt = "\n".join(prompt_parts)

        # Build response
        response_parts = []
        for i, move in enumerate(self.moves, 1):
            response_parts.append(f"Step {i}: {move}")
        if self.success:
            response_parts.append("\nPuzzle solved successfully!")
        else:
            response_parts.append("\nPuzzle incomplete.")
        response = "\n".join(response_parts)

        # Tokenize
        prompt_ids = tokenizer.encode(prompt)
        response_ids = tokenizer.encode(response)

        # Combine with loss mask (0 for prompt, 1 for response)
        input_ids = prompt_ids + response_ids
        loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)

        return input_ids, loss_mask

    def _format_grid(self, grid: list[list[int]]) -> str:
        """Format grid for display."""
        if not grid:
            return ""
        lines = []
        for row in grid:
            lines.append(" ".join(str(c) if c > 0 else "." for c in row))
        return "\n".join(lines)
