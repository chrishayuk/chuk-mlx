"""
Chess Tokenizer Example

Demonstrates a domain-specific tokenizer for chess notation (PGN/algebraic notation).
This tokenizer handles chess moves, pieces, squares, and game annotations.

Uses Pydantic models and enums for type-safe configuration.
"""

from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizer

from chuk_lazarus.data.tokenizers.batch_processing import (
    PaddingSide,
    create_batch,
    get_sequence_lengths,
)
from chuk_lazarus.data.tokenizers.token_stats import (
    calculate_compression_ratio,
)
from chuk_lazarus.data.tokenizers.validation import (
    check_roundtrip,
    create_validation_report,
)
from chuk_lazarus.data.tokenizers.vocab_manager import (
    get_vocabulary_stats,
)


class ChessPiece(str, Enum):
    """Chess pieces in standard notation."""

    KING = "K"
    QUEEN = "Q"
    ROOK = "R"
    BISHOP = "B"
    KNIGHT = "N"
    PAWN = ""  # Pawns have no letter in algebraic notation


class ChessResult(str, Enum):
    """Game result annotations."""

    WHITE_WINS = "1-0"
    BLACK_WINS = "0-1"
    DRAW = "1/2-1/2"
    ONGOING = "*"


class ChessAnnotation(str, Enum):
    """Move quality annotations."""

    BRILLIANT = "!!"
    GOOD = "!"
    INTERESTING = "!?"
    DUBIOUS = "?!"
    MISTAKE = "?"
    BLUNDER = "??"
    CHECK = "+"
    CHECKMATE = "#"
    CAPTURE = "x"
    CASTLES_KINGSIDE = "O-O"
    CASTLES_QUEENSIDE = "O-O-O"


class ChessTokenizerConfig(BaseModel):
    """Configuration for the chess tokenizer."""

    include_annotations: bool = Field(default=True, description="Include move quality annotations")
    include_move_numbers: bool = Field(default=True, description="Include move numbers")
    include_results: bool = Field(default=True, description="Include game result tokens")
    max_sequence_length: int = Field(default=512, ge=1, description="Maximum sequence length")


class ChessGame(BaseModel):
    """A chess game in PGN format."""

    moves: list[str] = Field(description="List of moves in algebraic notation")
    result: ChessResult = Field(default=ChessResult.ONGOING, description="Game result")
    white_player: str = Field(default="Unknown", description="White player name")
    black_player: str = Field(default="Unknown", description="Black player name")

    def to_pgn(self) -> str:
        """Convert to PGN move text."""
        pgn_parts = []
        for i, move in enumerate(self.moves):
            if i % 2 == 0:
                move_num = (i // 2) + 1
                pgn_parts.append(f"{move_num}.")
            pgn_parts.append(move)
        pgn_parts.append(self.result.value)
        return " ".join(pgn_parts)


class ChessTokenizer(PreTrainedTokenizer):
    """
    A tokenizer specialized for chess notation.

    Handles standard algebraic notation including:
    - Piece movements (Nf3, Bxe5, Qh4)
    - Castling (O-O, O-O-O)
    - Captures, checks, checkmates
    - Move numbers and results
    """

    # Chess-specific vocabulary
    SQUARES: ClassVar[list[str]] = [f"{file}{rank}" for file in "abcdefgh" for rank in "12345678"]

    PIECES: ClassVar[list[str]] = ["K", "Q", "R", "B", "N"]

    SPECIAL_MOVES: ClassVar[list[str]] = [
        "O-O",
        "O-O-O",
        "x",
        "+",
        "#",
        "=",
        "!",
        "?",
        "!!",
        "??",
        "!?",
        "?!",
    ]

    RESULTS: ClassVar[list[str]] = ["1-0", "0-1", "1/2-1/2", "*"]

    def __init__(self, config: ChessTokenizerConfig | None = None, **kwargs):
        """Initialize the chess tokenizer."""
        self.config = config or ChessTokenizerConfig()
        self._build_chess_vocab()

        # Initialize base class with special tokens
        super().__init__(
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<bos>",
            eos_token="<eos>",
            sep_token="<sep>",
            **kwargs,
        )

    def _build_chess_vocab(self) -> None:
        """Build the chess-specific vocabulary."""
        # Start with special tokens
        base_vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<sep>": 4,
        }

        current_id = len(base_vocab)

        # Add squares (a1-h8)
        for square in self.SQUARES:
            base_vocab[square] = current_id
            current_id += 1

        # Add pieces
        for piece in self.PIECES:
            base_vocab[piece] = current_id
            current_id += 1

        # Add special moves and annotations
        for move in self.SPECIAL_MOVES:
            if move not in base_vocab:
                base_vocab[move] = current_id
                current_id += 1

        # Add results
        if self.config.include_results:
            for result in self.RESULTS:
                if result not in base_vocab:
                    base_vocab[result] = current_id
                    current_id += 1

        # Add move numbers (1-200)
        if self.config.include_move_numbers:
            for num in range(1, 201):
                token = f"{num}."
                base_vocab[token] = current_id
                current_id += 1

        # Add common move patterns
        common_moves = self._generate_common_moves()
        for move in common_moves:
            if move not in base_vocab:
                base_vocab[move] = current_id
                current_id += 1

        self.vocab = base_vocab
        self.id_to_token = {v: k for k, v in base_vocab.items()}

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        """Return the vocabulary."""
        return self.vocab.copy()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into tokens."""
        return text.split()

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        return self.vocab.get(token, self.vocab.get("<unk>", 1))

    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token."""
        return self.id_to_token.get(index, "<unk>")

    def _generate_common_moves(self) -> list[str]:
        """Generate common chess move patterns."""
        moves = []

        # Pawn moves (e4, d5, etc.)
        for file in "abcdefgh":
            for rank in "12345678":
                moves.append(f"{file}{rank}")

        # Piece moves to squares
        for piece in self.PIECES:
            for file in "abcdefgh":
                for rank in "12345678":
                    moves.append(f"{piece}{file}{rank}")

        # Captures
        for piece in self.PIECES:
            for file in "abcdefgh":
                for rank in "12345678":
                    moves.append(f"{piece}x{file}{rank}")

        # Pawn captures
        for from_file in "abcdefgh":
            for to_file in "abcdefgh":
                if abs(ord(from_file) - ord(to_file)) == 1:
                    for rank in "12345678":
                        moves.append(f"{from_file}x{to_file}{rank}")

        return moves

    def tokenize_game(self, game: ChessGame) -> list[int]:
        """
        Tokenize a complete chess game.

        Args:
            game: ChessGame instance

        Returns:
            List of token IDs
        """
        pgn = game.to_pgn()
        tokens = self.encode(pgn, add_special_tokens=True)
        return tokens

    def tokenize_move(self, move: str) -> list[int]:
        """
        Tokenize a single chess move.

        Args:
            move: Move in algebraic notation (e.g., "Nf3", "e4", "O-O")

        Returns:
            List of token IDs
        """
        # Try exact match first
        if move in self.vocab:
            return [self.vocab[move]]

        # Decompose complex moves
        tokens = []

        # Handle castling
        if move in ("O-O", "O-O-O"):
            return [self.vocab.get(move, self.unk_token_id)]

        # Extract piece (if present)
        idx = 0
        if move and move[0] in self.PIECES:
            piece = move[0]
            if piece in self.vocab:
                tokens.append(self.vocab[piece])
            idx = 1

        # Handle capture
        if "x" in move:
            tokens.append(self.vocab.get("x", self.unk_token_id))
            move = move.replace("x", "")
            idx = 0 if not tokens or move[0] not in self.PIECES else 1

        # Extract destination square
        remaining = move[idx:]
        for i in range(len(remaining)):
            if remaining[i : i + 2] in self.SQUARES:
                tokens.append(self.vocab[remaining[i : i + 2]])
                remaining = remaining[i + 2 :]
                break

        # Handle check/checkmate
        if "+" in remaining:
            tokens.append(self.vocab.get("+", self.unk_token_id))
        if "#" in remaining:
            tokens.append(self.vocab.get("#", self.unk_token_id))

        # Handle annotations
        for ann in ["!!", "??", "!?", "?!", "!", "?"]:
            if ann in remaining:
                if ann in self.vocab:
                    tokens.append(self.vocab[ann])
                break

        return tokens if tokens else [self.unk_token_id]

    def decode_to_moves(self, token_ids: list[int]) -> list[str]:
        """
        Decode token IDs back to chess moves.

        Args:
            token_ids: List of token IDs

        Returns:
            List of move strings
        """
        moves = []
        current_move = []

        for tid in token_ids:
            token = self.id_to_token.get(tid, "<unk>")

            # Skip special tokens
            if token in ("<pad>", "<bos>", "<eos>", "<sep>", "<unk>"):
                continue

            # Move numbers indicate new move
            if token.endswith("."):
                if current_move:
                    moves.append("".join(current_move))
                    current_move = []
                continue

            # Results end the game
            if token in self.RESULTS:
                if current_move:
                    moves.append("".join(current_move))
                break

            # Squares or pieces start/continue moves
            if token in self.SQUARES or token in self.PIECES:
                if token in self.SQUARES and current_move:
                    current_move.append(token)
                    moves.append("".join(current_move))
                    current_move = []
                else:
                    current_move.append(token)
            else:
                current_move.append(token)

        if current_move:
            moves.append("".join(current_move))

        return moves


def demo_chess_tokenizer():
    """Demonstrate the chess tokenizer."""
    print("=" * 60)
    print("Chess Tokenizer Demo")
    print("=" * 60)

    # Create tokenizer
    config = ChessTokenizerConfig(
        include_annotations=True,
        include_move_numbers=True,
        include_results=True,
    )
    tokenizer = ChessTokenizer(config)

    # Show vocabulary stats
    vocab = tokenizer.get_vocab()
    stats = get_vocabulary_stats(vocab)
    print("\nVocabulary Statistics:")
    print(f"  Size: {stats.size}")
    print(f"  ID range: {stats.min_id} - {stats.max_id}")
    print(f"  Avg token length: {stats.avg_token_length:.2f}")

    # Create a sample game
    game = ChessGame(
        moves=["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O"],
        result=ChessResult.ONGOING,
        white_player="Kasparov",
        black_player="Karpov",
    )

    print(f"\nSample Game: {game.white_player} vs {game.black_player}")
    print(f"PGN: {game.to_pgn()}")

    # Tokenize the game
    tokens = tokenizer.tokenize_game(game)
    print(f"\nTokenized ({len(tokens)} tokens):")
    print(f"  IDs: {tokens[:20]}..." if len(tokens) > 20 else f"  IDs: {tokens}")

    # Decode back
    decoded = tokenizer.decode(tokens)
    print(f"  Decoded: {decoded}")

    # Test individual moves
    print("\nIndividual Move Tokenization:")
    test_moves = ["e4", "Nf3", "Bxe5", "O-O", "Qh4+", "Nxf7#"]
    for move in test_moves:
        move_tokens = tokenizer.tokenize_move(move)
        move_decoded = [tokenizer.id_to_token.get(t, "?") for t in move_tokens]
        print(f"  {move:8} -> {move_tokens} -> {move_decoded}")

    # Test compression ratio
    pgn_text = game.to_pgn()
    compression = calculate_compression_ratio(pgn_text, tokenizer)
    print("\nCompression Stats:")
    print(f"  Characters: {compression.char_count}")
    print(f"  Tokens: {compression.token_count}")
    print(f"  Chars/token: {compression.chars_per_token:.2f}")

    # Test roundtrip
    print("\nRoundtrip Test:")
    result = check_roundtrip(pgn_text, tokenizer)
    print(f"  Original:  {result.original}")
    print(f"  Decoded:   {result.decoded}")
    print(f"  Lossless:  {result.is_lossless}")

    # Batch processing
    print("\nBatch Processing:")
    games = [
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 *",
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 1-0",
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 0-1",
    ]
    batch = create_batch(
        games,
        tokenizer,
        padding=True,
        padding_side=PaddingSide.RIGHT,
    )
    seq_stats = get_sequence_lengths(batch.input_ids)
    print(f"  Batch size: {seq_stats.count}")
    print(f"  Min length: {seq_stats.min_length}")
    print(f"  Max length: {seq_stats.max_length}")
    print(f"  Mean length: {seq_stats.mean_length:.1f}")

    # Validation report
    print("\nValidation Report:")
    report = create_validation_report(tokenizer, "ChessTokenizer")
    print(f"  Valid: {report.is_valid}")
    print(f"  Errors: {report.error_count}")
    print(f"  Warnings: {report.warning_count}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_chess_tokenizer()
