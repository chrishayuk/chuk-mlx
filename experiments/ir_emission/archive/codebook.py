"""
Learned IR Codebook

Maps discrete codes to WASM IR fragments. The model learns to emit
sequences of codebook indices, which are then compiled to WASM bytes.

Design choices:
- VQ-VAE style: learned embeddings that map to IR fragments
- ~64 "atoms" that compose into programs
- Structured: opcodes, operand slots, control flow
"""

from dataclasses import dataclass
from enum import IntEnum

import mlx.core as mx
import mlx.nn as nn


class IROpcode(IntEnum):
    """IR opcodes - indices into the codebook."""

    # Special tokens
    PAD = 0
    START = 1
    END = 2

    # Operand slots (filled from extracted numbers)
    SLOT_0 = 3  # First extracted number
    SLOT_1 = 4  # Second extracted number
    SLOT_2 = 5  # Third extracted number
    SLOT_3 = 6  # Fourth extracted number

    # Integer constants
    CONST_0 = 7
    CONST_1 = 8
    CONST_2 = 9
    CONST_10 = 10

    # Binary arithmetic (i32)
    I32_ADD = 16
    I32_SUB = 17
    I32_MUL = 18
    I32_DIV_S = 19  # Signed division
    I32_REM_S = 20  # Signed remainder (modulo)

    # Comparison (i32)
    I32_EQ = 24
    I32_NE = 25
    I32_LT_S = 26
    I32_GT_S = 27
    I32_LE_S = 28
    I32_GE_S = 29

    # Unary
    I32_NEG = 32  # Negate (0 - x)
    I32_ABS = 33  # Absolute value

    # Control flow
    LOOP_BEGIN = 40
    LOOP_END = 41
    BR = 42  # Branch
    BR_IF = 43  # Branch if true
    IF_BEGIN = 44
    ELSE = 45
    IF_END = 46

    # Local variables
    LOCAL_GET_0 = 48
    LOCAL_SET_0 = 49
    LOCAL_GET_1 = 50
    LOCAL_SET_1 = 51
    LOCAL_TEE_0 = 52  # Set and keep on stack

    # Stack manipulation
    DROP = 56
    DUP = 57  # Duplicate top of stack (not native WASM, we expand)


# WASM bytecode mappings
OPCODE_TO_WASM = {
    IROpcode.I32_ADD: bytes([0x6A]),  # i32.add
    IROpcode.I32_SUB: bytes([0x6B]),  # i32.sub
    IROpcode.I32_MUL: bytes([0x6C]),  # i32.mul
    IROpcode.I32_DIV_S: bytes([0x6D]),  # i32.div_s
    IROpcode.I32_REM_S: bytes([0x6F]),  # i32.rem_s
    IROpcode.I32_EQ: bytes([0x46]),  # i32.eq
    IROpcode.I32_NE: bytes([0x47]),  # i32.ne
    IROpcode.I32_LT_S: bytes([0x48]),  # i32.lt_s
    IROpcode.I32_GT_S: bytes([0x4A]),  # i32.gt_s
    IROpcode.I32_LE_S: bytes([0x4C]),  # i32.le_s
    IROpcode.I32_GE_S: bytes([0x4E]),  # i32.ge_s
    IROpcode.LOCAL_GET_0: bytes([0x20, 0x00]),  # local.get 0
    IROpcode.LOCAL_SET_0: bytes([0x21, 0x00]),  # local.set 0
    IROpcode.LOCAL_GET_1: bytes([0x20, 0x01]),  # local.get 1
    IROpcode.LOCAL_SET_1: bytes([0x21, 0x01]),  # local.set 1
    IROpcode.LOCAL_TEE_0: bytes([0x22, 0x00]),  # local.tee 0
    IROpcode.DROP: bytes([0x1A]),  # drop
    IROpcode.LOOP_BEGIN: bytes([0x03, 0x40]),  # loop (void)
    IROpcode.LOOP_END: bytes([0x0B]),  # end
    IROpcode.IF_BEGIN: bytes([0x04, 0x40]),  # if (void)
    IROpcode.ELSE: bytes([0x05]),  # else
    IROpcode.IF_END: bytes([0x0B]),  # end
    IROpcode.BR: bytes([0x0C, 0x00]),  # br 0
    IROpcode.BR_IF: bytes([0x0D, 0x00]),  # br_if 0
}


def encode_i32_const(value: int) -> bytes:
    """Encode i32.const with LEB128 signed integer."""
    # i32.const opcode
    result = bytearray([0x41])

    # LEB128 encode the value
    value = value & 0xFFFFFFFF  # Treat as unsigned for bit ops
    if value >= 0x80000000:
        value -= 0x100000000  # Convert to signed

    more = True
    while more:
        byte = value & 0x7F
        value >>= 7
        # Sign extend
        if value == 0 and (byte & 0x40) == 0:
            more = False
        elif value == -1 and (byte & 0x40) != 0:
            more = False
        else:
            byte |= 0x80
        result.append(byte)

    return bytes(result)


@dataclass
class CodebookConfig:
    """Configuration for the IR codebook."""

    codebook_size: int = 64  # Number of discrete codes
    hidden_dim: int = 2048  # Model hidden dimension
    embedding_dim: int = 128  # Codebook embedding dimension
    max_ir_length: int = 16  # Maximum IR sequence length
    commitment_cost: float = 0.25  # VQ commitment loss weight


class IRCodebook(nn.Module):
    """
    Learned codebook mapping hidden states to IR fragments.

    Uses vector quantization: h13 → nearest codebook entry → IR opcode
    """

    def __init__(self, config: CodebookConfig):
        super().__init__()
        self.config = config

        # Project hidden state to embedding space
        self.input_proj = nn.Linear(config.hidden_dim, config.embedding_dim)

        # Codebook embeddings (learnable)
        self.embeddings = mx.random.normal(shape=(config.codebook_size, config.embedding_dim)) * 0.1

        # Output projection for autoregressive decoding
        self.output_proj = nn.Linear(config.embedding_dim, config.codebook_size)

    def quantize(self, z: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """
        Vector quantization: find nearest codebook entry.

        Args:
            z: Input embeddings (batch, embedding_dim)

        Returns:
            quantized: Quantized embeddings (batch, embedding_dim)
            indices: Codebook indices (batch,)
            commitment_loss: VQ commitment loss
        """
        # Compute distances to all codebook entries
        # z: (batch, embed_dim), embeddings: (codebook_size, embed_dim)
        distances = (
            mx.sum(z**2, axis=-1, keepdims=True)
            - 2 * z @ self.embeddings.T
            + mx.sum(self.embeddings**2, axis=-1)
        )

        # Find nearest
        indices = mx.argmin(distances, axis=-1)

        # Get quantized embeddings
        quantized = self.embeddings[indices]

        # Commitment loss: encourage encoder to commit to codebook
        commitment_loss = mx.mean((z - mx.stop_gradient(quantized)) ** 2)

        # Straight-through estimator: copy gradients from quantized to z
        quantized = z + mx.stop_gradient(quantized - z)

        return quantized, indices, commitment_loss

    def encode(self, hidden_state: mx.array) -> tuple[mx.array, mx.array]:
        """
        Encode hidden state to codebook indices.

        Args:
            hidden_state: L13 hidden state (batch, hidden_dim)

        Returns:
            indices: Codebook indices (batch,)
            commitment_loss: VQ loss for training
        """
        z = self.input_proj(hidden_state)
        _, indices, commitment_loss = self.quantize(z)
        return indices, commitment_loss

    def decode_to_logits(self, hidden_state: mx.array) -> mx.array:
        """
        Get logits over codebook for next-token prediction.

        Args:
            hidden_state: Hidden state (batch, hidden_dim)

        Returns:
            logits: (batch, codebook_size)
        """
        z = self.input_proj(hidden_state)
        return self.output_proj(z)

    def indices_to_wasm(
        self,
        indices: list[int],
        operands: list[int],
    ) -> bytes:
        """
        Convert codebook indices to WASM bytecode.

        Args:
            indices: Sequence of IROpcode values
            operands: Extracted numbers from the input

        Returns:
            WASM bytecode for the function body
        """
        wasm_bytes = bytearray()

        for idx in indices:
            opcode = IROpcode(idx)

            if opcode in (IROpcode.PAD, IROpcode.START, IROpcode.END):
                continue

            elif opcode == IROpcode.SLOT_0:
                if len(operands) > 0:
                    wasm_bytes.extend(encode_i32_const(operands[0]))
            elif opcode == IROpcode.SLOT_1:
                if len(operands) > 1:
                    wasm_bytes.extend(encode_i32_const(operands[1]))
            elif opcode == IROpcode.SLOT_2:
                if len(operands) > 2:
                    wasm_bytes.extend(encode_i32_const(operands[2]))
            elif opcode == IROpcode.SLOT_3:
                if len(operands) > 3:
                    wasm_bytes.extend(encode_i32_const(operands[3]))

            elif opcode == IROpcode.CONST_0:
                wasm_bytes.extend(encode_i32_const(0))
            elif opcode == IROpcode.CONST_1:
                wasm_bytes.extend(encode_i32_const(1))
            elif opcode == IROpcode.CONST_2:
                wasm_bytes.extend(encode_i32_const(2))
            elif opcode == IROpcode.CONST_10:
                wasm_bytes.extend(encode_i32_const(10))

            elif opcode in OPCODE_TO_WASM:
                wasm_bytes.extend(OPCODE_TO_WASM[opcode])

            elif opcode == IROpcode.I32_NEG:
                # Negate: 0 - x
                wasm_bytes.extend(encode_i32_const(0))
                wasm_bytes.extend(bytes([0x6B]))  # i32.sub (swap args)

            elif opcode == IROpcode.DUP:
                # Duplicate: local.tee 0, local.get 0 (requires local)
                wasm_bytes.extend(bytes([0x22, 0x00, 0x20, 0x00]))

        return bytes(wasm_bytes)


class IRSequenceDecoder(nn.Module):
    """
    Autoregressive decoder: h13 → sequence of IR opcodes.

    Given the L13 hidden state, generates a sequence of codebook
    indices representing the IR program.
    """

    def __init__(self, config: CodebookConfig):
        super().__init__()
        self.config = config
        self.codebook = IRCodebook(config)

        # Position embeddings for autoregressive decoding
        self.pos_embed = nn.Embedding(config.max_ir_length, config.embedding_dim)

        # Small transformer for sequence modeling
        self.layers = [
            nn.TransformerEncoderLayer(
                dims=config.embedding_dim,
                num_heads=4,
                mlp_dims=config.embedding_dim * 2,
            )
            for _ in range(2)
        ]

    def __call__(
        self,
        hidden_state: mx.array,
        target_ir: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Generate IR sequence from hidden state.

        Args:
            hidden_state: L13 hidden state (batch, hidden_dim)
            target_ir: Optional target sequence for teacher forcing (batch, seq_len)

        Returns:
            logits: (batch, max_ir_length, codebook_size)
            commitment_loss: VQ loss
        """
        batch_size = hidden_state.shape[0]

        # Initial embedding from hidden state
        z = self.codebook.input_proj(hidden_state)  # (batch, embed_dim)

        # Autoregressive decoding
        if target_ir is not None:
            # Teacher forcing: use target sequence
            seq_len = target_ir.shape[1]
            target_embeds = self.codebook.embeddings[target_ir]  # (batch, seq, embed)

            # Add position embeddings
            positions = mx.arange(seq_len)
            pos_embeds = self.pos_embed(positions)  # (seq, embed)

            # Prepend the hidden state embedding
            h = mx.concatenate([z[:, None, :], target_embeds[:, :-1, :]], axis=1)
            h = h + pos_embeds

            # Causal attention mask
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

            # Transform
            for layer in self.layers:
                h = layer(h, mask=mask)

            # Project to logits
            logits = self.codebook.output_proj(h)  # (batch, seq, codebook)

            # Commitment loss from quantizing the hidden state
            _, _, commitment_loss = self.codebook.quantize(z)

        else:
            # Greedy decoding
            logits_list = []
            h = z[:, None, :]  # (batch, 1, embed)

            for t in range(self.config.max_ir_length):
                pos_embed = self.pos_embed(mx.array([t]))
                h_t = h[:, -1:, :] + pos_embed

                for layer in self.layers:
                    h_t = layer(h_t)

                step_logits = self.codebook.output_proj(h_t[:, 0, :])
                logits_list.append(step_logits)

                # Get next token embedding
                next_idx = mx.argmax(step_logits, axis=-1)
                next_embed = self.codebook.embeddings[next_idx]
                h = mx.concatenate([h, next_embed[:, None, :]], axis=1)

                # Stop if END token
                if mx.all(next_idx == IROpcode.END):
                    break

            logits = mx.stack(logits_list, axis=1)
            _, _, commitment_loss = self.codebook.quantize(z)

        return logits, commitment_loss

    def generate(
        self,
        hidden_state: mx.array,
        temperature: float = 1.0,
        max_length: int | None = None,
    ) -> list[int]:
        """
        Generate IR sequence greedily or with sampling.

        Args:
            hidden_state: L13 hidden state (1, hidden_dim)
            temperature: Sampling temperature (0 = greedy)
            max_length: Maximum sequence length

        Returns:
            List of codebook indices
        """
        max_len = max_length or self.config.max_ir_length

        z = self.codebook.input_proj(hidden_state)
        h = z[:, None, :]

        indices = [IROpcode.START]

        for t in range(max_len):
            pos_embed = self.pos_embed(mx.array([t]))
            h_t = h[:, -1:, :] + pos_embed

            # No mask needed for single-step decoding
            for layer in self.layers:
                h_t = layer(h_t, mask=None)

            logits = self.codebook.output_proj(h_t[:, 0, :])

            if temperature == 0:
                next_idx = int(mx.argmax(logits, axis=-1).item())
            else:
                probs = mx.softmax(logits / temperature, axis=-1)
                next_idx = int(mx.random.categorical(mx.log(probs)).item())

            indices.append(next_idx)

            if next_idx == IROpcode.END:
                break

            next_embed = self.codebook.embeddings[mx.array([next_idx])]
            h = mx.concatenate([h, next_embed[:, None, :]], axis=1)

        return indices
