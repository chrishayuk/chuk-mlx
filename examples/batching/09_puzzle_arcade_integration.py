#!/usr/bin/env python3
"""
Example 9: Puzzle Arcade Server Integration

Demonstrates real integration with the puzzle-arcade-server:
- Connecting via telnet to the puzzle server
- Streaming puzzle episodes
- Converting episodes to training samples
- Building training datasets from puzzles

Prerequisites:
    1. Start the puzzle arcade server:
       cd /path/to/puzzle-arcade-server
       uv run python -m chuk_puzzles_gym

    2. Run this example:
       python examples/batching/09_puzzle_arcade_integration.py

Server defaults to localhost:8023
"""

import asyncio
import sys

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 3)[0] + "/src")

from chuk_lazarus.data.batching.streaming import (
    ReplayBuffer,
    ReplayBufferConfig,
    SampleSource,
    StreamSample,
)
from chuk_lazarus.data.batching.streaming.telnet_client import (
    PuzzleDifficulty,
    PuzzleGame,
    TelnetClientConfig,
    TelnetGymClient,
)


class SimpleTokenizer:
    """Simple word-based tokenizer for demonstration."""

    def __init__(self):
        self._vocab = {}
        self._next_id = 1

    @property
    def eos_token_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        """Simple word-based tokenization."""
        tokens = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = self._next_id
                self._next_id += 1
            tokens.append(self._vocab[word])
        return tokens


async def demo_basic_connection():
    """Demonstrate basic connection to puzzle server."""
    print("\n" + "=" * 60)
    print("1. Basic Connection Test")
    print("=" * 60)

    config = TelnetClientConfig(
        host="localhost",
        port=8023,
        connect_timeout=5.0,
        max_retries=2,
    )

    try:
        async with TelnetGymClient(config) as client:
            print("   Connected to puzzle server!")
            print(f"   is_connected: {client.is_connected}")

            # Start a simple Sudoku puzzle
            print("\n   Starting Sudoku puzzle...")
            obs = await client.start_puzzle(PuzzleGame.SUDOKU, PuzzleDifficulty.EASY)

            print(f"   Game: {obs.game}")
            print(f"   Difficulty: {obs.difficulty}")
            print(f"   Seed: {obs.seed}")
            print(f"   Hints remaining: {obs.hints_remaining}")
            if obs.optimal_steps:
                print(f"   Optimal steps: {obs.optimal_steps}")

            # Show the grid
            if obs.grid_display:
                print("\n   Grid:")
                for line in obs.grid_display.split("\n")[:5]:
                    print(f"     {line}")
                print("     ...")

            # Get a hint
            print("\n   Getting hint...")
            hint_result = await client.get_hint()
            print(f"   Hint success: {hint_result.success}")
            if hint_result.message:
                print(f"   Hint: {hint_result.message}")

            # Get stats
            print("\n   Getting stats...")
            stats = await client.get_stats()
            print(f"   Moves: {stats.get('moves', 0)}")
            print(f"   Hints used: {stats.get('hints_used', 0)}")

            # Quit puzzle
            await client.quit_puzzle()
            print("\n   Puzzle quit successfully")

    except ConnectionError as e:
        print(f"   Connection failed: {e}")
        print("   Make sure the puzzle arcade server is running on localhost:8023")
        return False

    return True


async def demo_episode_collection():
    """Demonstrate collecting complete episodes."""
    print("\n" + "=" * 60)
    print("2. Episode Collection")
    print("=" * 60)

    config = TelnetClientConfig(host="localhost", port=8023)
    tokenizer = SimpleTokenizer()

    try:
        async with TelnetGymClient(config) as client:
            # Play through a puzzle using hints
            print("   Playing Sudoku using hints...")

            obs = await client.start_puzzle(
                PuzzleGame.SUDOKU,
                PuzzleDifficulty.EASY,
                seed=42,  # Reproducible
            )

            moves = []
            max_moves = 10  # Limit for demo

            while not obs.is_complete and len(moves) < max_moves:
                # Get hint (this is how we'd get optimal actions)
                hint = await client.get_hint()

                if hint.success and hint.message:
                    moves.append(hint.message)
                    print(f"     Move {len(moves)}: {hint.message[:50]}...")

                    # The hint already applied the move
                    # Get updated state
                    obs = await client.show_state()
                else:
                    print("     No more hints available")
                    break

            print(f"\n   Total moves: {len(moves)}")
            print(f"   Puzzle complete: {obs.is_complete}")

            # Convert to training samples
            print("\n   Converting to training samples...")

            samples = []
            for i, move in enumerate(moves):
                # Create prompt from puzzle state description
                prompt = f"Sudoku puzzle (move {i + 1}): What is the next optimal move?"
                response = move

                # Tokenize
                prompt_ids = tokenizer.encode(prompt)
                response_ids = tokenizer.encode(response)

                input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
                loss_mask = [0] * len(prompt_ids) + [1] * (len(response_ids) + 1)

                sample = StreamSample(
                    input_ids=tuple(input_ids),
                    loss_mask=tuple(loss_mask),
                    sample_id=f"sudoku_42_{i}",
                    dataset_id="puzzle_arcade",
                    source=SampleSource.GYM,
                    episode_id="sudoku_42",
                    step_index=i,
                    total_steps=len(moves),
                    difficulty=0.3,  # Easy
                    success=obs.is_complete,
                )
                samples.append(sample)

            print(f"   Created {len(samples)} training samples")
            for s in samples[:3]:
                print(f"     {s.sample_id}: {s.length} tokens")

    except ConnectionError as e:
        print(f"   Connection failed: {e}")
        return False

    return True


async def demo_buffer_filling():
    """Demonstrate filling a replay buffer with puzzle episodes."""
    print("\n" + "=" * 60)
    print("3. Replay Buffer Filling")
    print("=" * 60)

    config = TelnetClientConfig(host="localhost", port=8023)
    tokenizer = SimpleTokenizer()

    # Create replay buffer
    buffer = ReplayBuffer(
        ReplayBufferConfig(
            max_size=1000,
            track_difficulty=True,
            track_success=True,
        )
    )

    try:
        async with TelnetGymClient(config) as client:
            games = [
                (PuzzleGame.SUDOKU, PuzzleDifficulty.EASY),
                (PuzzleGame.SUDOKU, PuzzleDifficulty.MEDIUM),
            ]

            for game, difficulty in games:
                print(f"\n   Playing {game.value} ({difficulty.value})...")

                obs = await client.start_puzzle(game, difficulty)
                episode_id = f"{game.value}_{obs.seed}"

                step = 0
                max_steps = 5

                while not obs.is_complete and step < max_steps:
                    hint = await client.get_hint()
                    if not hint.success:
                        break

                    # Create sample
                    prompt = f"{game.value} puzzle step {step + 1}"
                    response = hint.message or "unknown"

                    prompt_ids = tokenizer.encode(prompt)
                    response_ids = tokenizer.encode(response)
                    input_ids = prompt_ids + response_ids

                    sample = StreamSample(
                        input_ids=tuple(input_ids),
                        loss_mask=tuple([0] * len(prompt_ids) + [1] * len(response_ids)),
                        sample_id=f"{episode_id}_{step}",
                        dataset_id="puzzle_arcade",
                        source=SampleSource.GYM,
                        episode_id=episode_id,
                        step_index=step,
                        total_steps=max_steps,
                        difficulty={"easy": 0.3, "medium": 0.5, "hard": 0.8}[difficulty.value],
                        reward=1.0 if obs.is_complete else 0.0,
                    )

                    buffer.add(sample)
                    step += 1
                    obs = await client.show_state()

                await client.quit_puzzle()

            print("\n   Buffer statistics:")
            print(f"     Size: {buffer.size}")
            print(f"     Mean difficulty: {buffer.mean_difficulty:.2f}")

            # Sample from buffer
            print("\n   Sampling from buffer...")
            sampled = buffer.sample(n=5)
            for s in sampled:
                print(f"     {s.sample_id}: difficulty={s.difficulty:.2f}")

    except ConnectionError as e:
        print(f"   Connection failed: {e}")
        return False

    return True


async def demo_multi_game():
    """Demonstrate playing multiple puzzle types."""
    print("\n" + "=" * 60)
    print("4. Multi-Game Demo")
    print("=" * 60)

    config = TelnetClientConfig(host="localhost", port=8023)

    # Games to try (may not all be available)
    games_to_try = [
        PuzzleGame.SUDOKU,
        PuzzleGame.LIGHTS_OUT,
        PuzzleGame.MASTERMIND,
    ]

    try:
        async with TelnetGymClient(config) as client:
            for game in games_to_try:
                try:
                    print(f"\n   Trying {game.value}...")
                    obs = await client.start_puzzle(game, PuzzleDifficulty.EASY)

                    print(f"     Game: {obs.game}")
                    print(f"     Difficulty: {obs.difficulty}")
                    print(f"     Optimal steps: {obs.optimal_steps}")

                    # Get one hint
                    hint = await client.get_hint()
                    if hint.success:
                        print(f"     First hint: {hint.message[:60] if hint.message else 'N/A'}...")

                    await client.quit_puzzle()

                except Exception as e:
                    print(f"     Failed: {e}")

    except ConnectionError as e:
        print(f"   Connection failed: {e}")
        return False

    return True


async def main():
    print("=" * 60)
    print("Puzzle Arcade Server Integration Demo")
    print("=" * 60)
    print("\nThis demo requires the puzzle arcade server running on localhost:8023")
    print("Start it with: cd puzzle-arcade-server && uv run python -m chuk_puzzles_gym")

    # Test basic connection first
    if not await demo_basic_connection():
        print("\n" + "=" * 60)
        print("Server not available - running in offline mode")
        print("=" * 60)
        print("\nTo test with the real server:")
        print("1. Clone puzzle-arcade-server")
        print("2. Start with: uv run python -m chuk_puzzles_gym")
        print("3. Run this example again")
        return

    # Run other demos
    await demo_episode_collection()
    await demo_buffer_filling()
    await demo_multi_game()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
