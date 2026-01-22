"""
Standardized CoT Training for Virtual Expert Routing.

This experiment trains small models (1B params) to generate standardized
chain-of-thought that reliably triggers virtual expert routing.

Format: <expert>: <spec> -> <result>

Examples:
    multiply: 256 * 4 -> 1024
    word_problem: eggs:16 | -3 -4 *2 -> 18
    schedule: Alice:2hr,Bob:1hr | no_overlap -> Alice:9-11,Bob:11-12
    time: Asia/Tokyo -> 14:30 JST
    chat: -> [response]

Usage:
    python -m experiments.cot_standardization.train --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
