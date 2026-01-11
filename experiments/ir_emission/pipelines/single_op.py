"""
Single Operation Pipeline.

Tests the neural compiler on single arithmetic operations:
- Addition, subtraction, multiplication, division
- Simple commands: "Add 11 and 94"
- Varied phrasing: "The sum of 25 and 17 is"
- Word problems: "Janet has 50 apples. She gives away 15."

Expected accuracy: 100%
"""

from .base import BasePipeline


class SingleOpPipeline(BasePipeline):
    """Pipeline for single arithmetic operations."""

    name = "single_op"

    def get_test_cases(self) -> list[tuple[str, int]]:
        """Return test cases for single operations."""
        return [
            # Simple commands
            ("Add 11 and 94", 105),
            ("Subtract 49 from 69", 20),
            ("Multiply 7 by 8", 56),
            ("Divide 48 by 6", 8),

            # Varied phrasing
            ("The sum of 25 and 17 is", 42),
            ("The difference of 100 and 37 is", 63),
            ("What is 12 times 9?", 108),
            ("What is 144 divided by 12?", 12),

            # Word problems
            ("Janet has 50 apples. She gives away 15. How many remain?", 35),
            ("Each box holds 8 items. How many in 7 boxes?", 56),
            ("A tank has 200 gallons. 75 leak out. How much is left?", 125),
            ("Tickets cost 15 dollars each. Cost for 4 tickets?", 60),
        ]
