#!/usr/bin/env python3
"""Fix tokenizer test patches."""

test_file = "tests/cli/commands/test_tokenizer.py"

# Read the file
with open(test_file) as f:
    content = f.read()

# Replace all instances
content = content.replace(
    'patch("chuk_lazarus.cli.commands.tokenizer.load_tokenizer"',
    'patch("chuk_lazarus.utils.tokenizer_loader.load_tokenizer"'
)

# Write it back
with open(test_file, "w") as f:
    f.write(content)

print("Fixed all load_tokenizer patches")
