"""Tests for special_registry module."""

from chuk_lazarus.data.tokenizers.runtime.special_registry import (
    CollisionReport,
    ReservedRange,
    SpecialTokenEntry,
    SpecialTokenRegistry,
    TokenCategory,
    check_collisions,
    create_standard_registry,
    get_reserved_ranges,
    register_special_token,
)


class TestTokenCategoryEnum:
    """Tests for TokenCategory enum."""

    def test_all_categories(self):
        assert TokenCategory.PADDING == "padding"
        assert TokenCategory.UNKNOWN == "unknown"
        assert TokenCategory.TOOL_CALL == "tool_call"
        assert TokenCategory.TOOL_RESULT == "tool_result"
        assert TokenCategory.MEMORY_LOAD == "memory_load"
        assert TokenCategory.PAGE_IN == "page_in"
        assert TokenCategory.SOLVER_OP == "solver_op"
        assert TokenCategory.THINK_START == "think_start"
        assert TokenCategory.THINK_END == "think_end"


class TestReservedRangeModel:
    """Tests for ReservedRange model."""

    def test_valid_range(self):
        range_obj = ReservedRange(
            category=TokenCategory.TOOL_CALL,
            start=50000,
            end=51000,
            description="Tool call tokens",
        )
        assert range_obj.start == 50000
        assert range_obj.end == 51000


class TestSpecialTokenEntryModel:
    """Tests for SpecialTokenEntry model."""

    def test_valid_entry(self):
        entry = SpecialTokenEntry(
            token_str="<TOOL_CALL>",
            token_id=50001,
            category=TokenCategory.TOOL_CALL,
            description="Start tool invocation",
        )
        assert entry.token_str == "<TOOL_CALL>"
        assert entry.token_id == 50001

    def test_entry_without_description(self):
        entry = SpecialTokenEntry(
            token_str="<PAD>",
            token_id=0,
            category=TokenCategory.PADDING,
        )
        assert entry.description == ""


class TestCollisionReportModel:
    """Tests for CollisionReport model."""

    def test_no_collisions(self):
        report = CollisionReport(
            has_collisions=False,
            collisions=[],
            reserved_violations=[],
        )
        assert not report.has_collisions
        assert len(report.collisions) == 0

    def test_with_collisions(self):
        report = CollisionReport(
            has_collisions=True,
            collisions=[("<A>", "<B>", 100)],
            reserved_violations=[],
        )
        assert report.has_collisions
        assert len(report.collisions) == 1


class TestSpecialTokenRegistry:
    """Tests for SpecialTokenRegistry class."""

    def test_create_empty_registry(self):
        registry = SpecialTokenRegistry()
        assert len(registry.tokens) == 0

    def test_register_token(self):
        registry = SpecialTokenRegistry()
        registry.register(
            token_str="<TOOL_CALL>",
            token_id=50001,
            category=TokenCategory.TOOL_CALL,
            description="Tool call start",
        )
        assert len(registry.tokens) == 1
        assert "<TOOL_CALL>" in registry.tokens

    def test_register_multiple_tokens(self):
        registry = SpecialTokenRegistry()
        registry.register("<PAD>", 0, TokenCategory.PADDING)
        registry.register("<UNK>", 1, TokenCategory.UNKNOWN)
        registry.register("<TOOL>", 50000, TokenCategory.TOOL_CALL)
        assert len(registry.tokens) == 3

    def test_register_auto_id(self):
        registry = SpecialTokenRegistry(next_dynamic_id=100)
        entry = registry.register("<AUTO>", category=TokenCategory.CUSTOM)
        assert entry.token_id == 100

    def test_register_duplicate_token_raises(self):
        registry = SpecialTokenRegistry()
        registry.register("<PAD>", 0, TokenCategory.PADDING)
        try:
            registry.register("<PAD>", 1, TokenCategory.PADDING)
            raise AssertionError("Should raise ValueError")
        except ValueError as e:
            assert "already registered" in str(e)

    def test_register_duplicate_id_raises(self):
        registry = SpecialTokenRegistry()
        registry.register("<PAD>", 0, TokenCategory.PADDING)
        try:
            registry.register("<UNK>", 0, TokenCategory.UNKNOWN)
            raise AssertionError("Should raise ValueError")
        except ValueError as e:
            assert "already used" in str(e)

    def test_get_by_category(self):
        registry = SpecialTokenRegistry()
        registry.register("<TOOL1>", 50001, TokenCategory.TOOL_CALL)
        registry.register("<TOOL2>", 50002, TokenCategory.TOOL_CALL)
        registry.register("<PAD>", 0, TokenCategory.PADDING)

        tool_tokens = registry.get_by_category(TokenCategory.TOOL_CALL)
        assert len(tool_tokens) == 2

    def test_get_by_id(self):
        registry = SpecialTokenRegistry()
        registry.register("<PAD>", 0, TokenCategory.PADDING)
        registry.register("<UNK>", 1, TokenCategory.UNKNOWN)

        entry = registry.get_by_id(0)
        assert entry is not None
        assert entry.token_str == "<PAD>"

    def test_get_by_id_not_found(self):
        registry = SpecialTokenRegistry()
        registry.register("<PAD>", 0, TokenCategory.PADDING)

        entry = registry.get_by_id(999)
        assert entry is None

    def test_check_collisions_none(self):
        registry = SpecialTokenRegistry()
        registry.register("<PAD>", 0, TokenCategory.PADDING)
        registry.register("<UNK>", 1, TokenCategory.UNKNOWN)

        report = registry.check_collisions()
        assert not report.has_collisions

    def test_reserved_ranges(self):
        registry = SpecialTokenRegistry()
        registry.reserve_range(50000, 51000, TokenCategory.TOOL_CALL, "Tool tokens")

        assert len(registry.reserved_ranges) == 1
        assert registry.reserved_ranges[0].start == 50000

    def test_reserved_violation_detection(self):
        registry = SpecialTokenRegistry()
        registry.reserve_range(50000, 51000, TokenCategory.TOOL_CALL)
        # Register wrong category in reserved range
        registry.register("<WRONG>", 50005, TokenCategory.MEMORY_LOAD)

        report = registry.check_collisions()
        assert report.has_collisions
        assert len(report.reserved_violations) == 1


class TestRegisterSpecialToken:
    """Tests for register_special_token function."""

    def test_basic_registration(self):
        registry = SpecialTokenRegistry()
        entry = register_special_token(
            registry,
            token_str="<TOOL>",
            token_id=50000,
            category=TokenCategory.TOOL_CALL,
        )
        assert entry is not None
        assert len(registry.tokens) == 1

    def test_registration_with_description(self):
        registry = SpecialTokenRegistry()
        entry = register_special_token(
            registry,
            token_str="<TOOL>",
            category=TokenCategory.TOOL_CALL,
            token_id=50000,
            description="Tool call token",
        )
        assert entry.description == "Tool call token"


class TestCheckCollisions:
    """Tests for check_collisions function."""

    def test_no_collisions(self):
        registry = SpecialTokenRegistry()
        registry.register("<A>", 0, TokenCategory.PADDING)
        registry.register("<B>", 1, TokenCategory.UNKNOWN)

        report = check_collisions(registry)
        assert not report.has_collisions


class TestGetReservedRanges:
    """Tests for get_reserved_ranges function."""

    def test_empty_ranges(self):
        registry = SpecialTokenRegistry()
        ranges = get_reserved_ranges(registry)
        assert len(ranges) == 0

    def test_with_ranges(self):
        registry = SpecialTokenRegistry()
        registry.reserve_range(50000, 51000, TokenCategory.TOOL_CALL)
        registry.reserve_range(51000, 52000, TokenCategory.MEMORY_LOAD)

        ranges = get_reserved_ranges(registry)
        assert len(ranges) == 2

    def test_range_categories(self):
        registry = SpecialTokenRegistry()
        registry.reserve_range(50000, 51000, TokenCategory.TOOL_CALL)

        ranges = get_reserved_ranges(registry)
        assert ranges[0].category == TokenCategory.TOOL_CALL


class TestCreateStandardRegistry:
    """Tests for create_standard_registry function."""

    def test_creates_registry(self):
        registry = create_standard_registry()
        assert isinstance(registry, SpecialTokenRegistry)

    def test_has_standard_tokens(self):
        registry = create_standard_registry()
        assert "<pad>" in registry.tokens
        assert "<unk>" in registry.tokens
        assert "<s>" in registry.tokens
        assert "</s>" in registry.tokens

    def test_has_reserved_ranges(self):
        registry = create_standard_registry()
        assert len(registry.reserved_ranges) >= 3

    def test_custom_vocab_size(self):
        registry = create_standard_registry(vocab_size=50000)
        # Reserved ranges should start at vocab_size
        assert registry.reserved_ranges[0].start == 50000
