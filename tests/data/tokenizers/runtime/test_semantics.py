"""Tests for semantics module."""

from chuk_lazarus.data.tokenizers.runtime.semantics import (
    SemanticDomain,
    SemanticMapping,
    TokenSemantics,
    create_standard_semantics,
    get_semantic_group,
    map_token_to_semantic,
)


class TestSemanticDomainEnum:
    """Tests for SemanticDomain enum."""

    def test_all_domains(self):
        assert SemanticDomain.MEMORY == "memory"
        assert SemanticDomain.TOOL == "tool"
        assert SemanticDomain.SOLVER == "solver"
        assert SemanticDomain.CONTROL == "control"
        assert SemanticDomain.DATA == "data"
        assert SemanticDomain.CUSTOM == "custom"

    def test_domain_values(self):
        assert SemanticDomain.MEMORY.value == "memory"
        assert SemanticDomain.TOOL.value == "tool"
        assert SemanticDomain.SOLVER.value == "solver"


class TestSemanticMappingModel:
    """Tests for SemanticMapping model."""

    def test_valid_mapping(self):
        mapping = SemanticMapping(
            token_str="<LOAD_PAGE>",
            token_id=100,
            domain=SemanticDomain.MEMORY,
            operation="load",
            full_path="memory.op.load",
        )
        assert mapping.token_str == "<LOAD_PAGE>"
        assert mapping.token_id == 100
        assert mapping.domain == SemanticDomain.MEMORY

    def test_mapping_with_arguments(self):
        mapping = SemanticMapping(
            token_str="<TOOL_CALL>",
            token_id=200,
            domain=SemanticDomain.TOOL,
            operation="call",
            full_path="tool.op.call",
            arguments=["tool_name", "args"],
            returns="result",
        )
        assert len(mapping.arguments) == 2
        assert mapping.returns == "result"

    def test_mapping_with_description(self):
        mapping = SemanticMapping(
            token_str="<ADD>",
            token_id=300,
            domain=SemanticDomain.SOLVER,
            operation="add",
            full_path="solver.op.add",
            description="Add two numbers",
        )
        assert mapping.description == "Add two numbers"

    def test_mapping_defaults(self):
        mapping = SemanticMapping(
            token_str="<TOKEN>",
            token_id=0,
            domain=SemanticDomain.CUSTOM,
            operation="op",
            full_path="custom.op.op",
        )
        assert mapping.arguments == []
        assert mapping.returns == ""
        assert mapping.description == ""


class TestTokenSemantics:
    """Tests for TokenSemantics registry class."""

    def test_create_empty_registry(self):
        semantics = TokenSemantics()
        assert len(semantics.mappings) == 0
        assert len(semantics.by_path) == 0
        assert len(semantics.domains) == 0

    def test_register_token(self):
        semantics = TokenSemantics()
        mapping = semantics.register(
            token_str="<LOAD_PAGE>",
            token_id=100,
            domain=SemanticDomain.MEMORY,
            operation="load",
        )
        assert isinstance(mapping, SemanticMapping)
        assert mapping.token_str == "<LOAD_PAGE>"
        assert mapping.token_id == 100
        assert len(semantics.mappings) == 1

    def test_register_creates_full_path(self):
        semantics = TokenSemantics()
        mapping = semantics.register(
            token_str="<STORE>",
            token_id=101,
            domain=SemanticDomain.MEMORY,
            operation="store",
        )
        assert mapping.full_path == "memory.op.store"

    def test_register_with_arguments(self):
        semantics = TokenSemantics()
        mapping = semantics.register(
            token_str="<ADD>",
            token_id=300,
            domain=SemanticDomain.SOLVER,
            operation="add",
            arguments=["a", "b"],
            returns="sum",
        )
        assert mapping.arguments == ["a", "b"]
        assert mapping.returns == "sum"

    def test_register_with_description(self):
        semantics = TokenSemantics()
        mapping = semantics.register(
            token_str="<MUL>",
            token_id=301,
            domain=SemanticDomain.SOLVER,
            operation="mul",
            description="Multiply two numbers",
        )
        assert mapping.description == "Multiply two numbers"

    def test_register_multiple_tokens(self):
        semantics = TokenSemantics()
        semantics.register("<A>", 1, SemanticDomain.MEMORY, "a")
        semantics.register("<B>", 2, SemanticDomain.TOOL, "b")
        semantics.register("<C>", 3, SemanticDomain.SOLVER, "c")
        assert len(semantics.mappings) == 3

    def test_register_updates_domains(self):
        semantics = TokenSemantics()
        semantics.register("<A>", 1, SemanticDomain.MEMORY, "a")
        semantics.register("<B>", 2, SemanticDomain.MEMORY, "b")
        assert SemanticDomain.MEMORY in semantics.domains
        assert len(semantics.domains[SemanticDomain.MEMORY]) == 2

    def test_register_updates_by_path(self):
        semantics = TokenSemantics()
        semantics.register("<LOAD>", 100, SemanticDomain.MEMORY, "load")
        assert "memory.op.load" in semantics.by_path
        assert semantics.by_path["memory.op.load"] == 100

    def test_get_by_id_found(self):
        semantics = TokenSemantics()
        semantics.register("<TOKEN>", 100, SemanticDomain.TOOL, "call")
        mapping = semantics.get_by_id(100)
        assert mapping is not None
        assert mapping.token_str == "<TOKEN>"

    def test_get_by_id_not_found(self):
        semantics = TokenSemantics()
        semantics.register("<TOKEN>", 100, SemanticDomain.TOOL, "call")
        mapping = semantics.get_by_id(999)
        assert mapping is None

    def test_get_by_path_found(self):
        semantics = TokenSemantics()
        semantics.register("<LOAD>", 100, SemanticDomain.MEMORY, "load")
        mapping = semantics.get_by_path("memory.op.load")
        assert mapping is not None
        assert mapping.token_id == 100

    def test_get_by_path_not_found(self):
        semantics = TokenSemantics()
        semantics.register("<LOAD>", 100, SemanticDomain.MEMORY, "load")
        mapping = semantics.get_by_path("memory.op.store")
        assert mapping is None

    def test_get_domain_tokens(self):
        semantics = TokenSemantics()
        semantics.register("<A>", 1, SemanticDomain.MEMORY, "a")
        semantics.register("<B>", 2, SemanticDomain.MEMORY, "b")
        semantics.register("<C>", 3, SemanticDomain.TOOL, "c")

        memory_tokens = semantics.get_domain_tokens(SemanticDomain.MEMORY)
        assert len(memory_tokens) == 2

        tool_tokens = semantics.get_domain_tokens(SemanticDomain.TOOL)
        assert len(tool_tokens) == 1

    def test_get_domain_tokens_empty(self):
        semantics = TokenSemantics()
        semantics.register("<A>", 1, SemanticDomain.MEMORY, "a")
        solver_tokens = semantics.get_domain_tokens(SemanticDomain.SOLVER)
        assert len(solver_tokens) == 0

    def test_get_domain_tokens_returns_mappings(self):
        semantics = TokenSemantics()
        semantics.register("<TOKEN>", 1, SemanticDomain.TOOL, "call")
        tokens = semantics.get_domain_tokens(SemanticDomain.TOOL)
        assert all(isinstance(t, SemanticMapping) for t in tokens)


class TestMapTokenToSemantic:
    """Tests for map_token_to_semantic function."""

    def test_basic_mapping(self):
        semantics = TokenSemantics()
        mapping = map_token_to_semantic(
            semantics,
            token_str="<LOAD>",
            token_id=100,
            domain=SemanticDomain.MEMORY,
            operation="load",
        )
        assert mapping.token_str == "<LOAD>"
        assert mapping.domain == SemanticDomain.MEMORY

    def test_mapping_with_kwargs(self):
        semantics = TokenSemantics()
        mapping = map_token_to_semantic(
            semantics,
            token_str="<ADD>",
            token_id=300,
            domain=SemanticDomain.SOLVER,
            operation="add",
            arguments=["a", "b"],
            returns="sum",
            description="Add two numbers",
        )
        assert mapping.arguments == ["a", "b"]
        assert mapping.returns == "sum"
        assert mapping.description == "Add two numbers"

    def test_mapping_registers_in_semantics(self):
        semantics = TokenSemantics()
        map_token_to_semantic(
            semantics,
            token_str="<TOKEN>",
            token_id=100,
            domain=SemanticDomain.TOOL,
            operation="call",
        )
        assert len(semantics.mappings) == 1
        assert 100 in semantics.mappings

    def test_multiple_mappings(self):
        semantics = TokenSemantics()
        map_token_to_semantic(semantics, "<A>", 1, SemanticDomain.MEMORY, "a")
        map_token_to_semantic(semantics, "<B>", 2, SemanticDomain.TOOL, "b")
        assert len(semantics.mappings) == 2


class TestGetSemanticGroup:
    """Tests for get_semantic_group function."""

    def test_get_group(self):
        semantics = TokenSemantics()
        semantics.register("<A>", 1, SemanticDomain.MEMORY, "a")
        semantics.register("<B>", 2, SemanticDomain.MEMORY, "b")
        semantics.register("<C>", 3, SemanticDomain.TOOL, "c")

        group = get_semantic_group(semantics, SemanticDomain.MEMORY)
        assert len(group) == 2

    def test_get_group_empty(self):
        semantics = TokenSemantics()
        semantics.register("<A>", 1, SemanticDomain.MEMORY, "a")
        group = get_semantic_group(semantics, SemanticDomain.SOLVER)
        assert len(group) == 0

    def test_get_group_returns_mappings(self):
        semantics = TokenSemantics()
        semantics.register("<TOKEN>", 1, SemanticDomain.TOOL, "call")
        group = get_semantic_group(semantics, SemanticDomain.TOOL)
        assert all(isinstance(m, SemanticMapping) for m in group)


class TestCreateStandardSemantics:
    """Tests for create_standard_semantics function."""

    def test_creates_registry(self):
        semantics = create_standard_semantics()
        assert isinstance(semantics, TokenSemantics)

    def test_has_memory_tokens(self):
        semantics = create_standard_semantics()
        memory_tokens = semantics.get_domain_tokens(SemanticDomain.MEMORY)
        assert len(memory_tokens) >= 4  # LOAD_PAGE, STORE_PAGE, PAGE_IN, PAGE_OUT

    def test_has_tool_tokens(self):
        semantics = create_standard_semantics()
        tool_tokens = semantics.get_domain_tokens(SemanticDomain.TOOL)
        assert len(tool_tokens) >= 2  # TOOL_CALL, TOOL_RESULT

    def test_has_solver_tokens(self):
        semantics = create_standard_semantics()
        solver_tokens = semantics.get_domain_tokens(SemanticDomain.SOLVER)
        assert len(solver_tokens) >= 4  # ADD, MUL, ARGMIN, ARGMAX

    def test_has_control_tokens(self):
        semantics = create_standard_semantics()
        control_tokens = semantics.get_domain_tokens(SemanticDomain.CONTROL)
        assert len(control_tokens) >= 2  # THINK, /THINK

    def test_load_page_token(self):
        semantics = create_standard_semantics()
        mapping = semantics.get_by_id(100)
        assert mapping is not None
        assert mapping.token_str == "<LOAD_PAGE>"
        assert mapping.domain == SemanticDomain.MEMORY

    def test_tool_call_token(self):
        semantics = create_standard_semantics()
        mapping = semantics.get_by_id(200)
        assert mapping is not None
        assert mapping.token_str == "<TOOL_CALL>"
        assert mapping.domain == SemanticDomain.TOOL

    def test_add_token(self):
        semantics = create_standard_semantics()
        mapping = semantics.get_by_id(300)
        assert mapping is not None
        assert mapping.token_str == "<ADD>"
        assert mapping.domain == SemanticDomain.SOLVER

    def test_think_token(self):
        semantics = create_standard_semantics()
        mapping = semantics.get_by_id(400)
        assert mapping is not None
        assert mapping.token_str == "<THINK>"
        assert mapping.domain == SemanticDomain.CONTROL

    def test_paths_registered(self):
        semantics = create_standard_semantics()
        assert "memory.op.load" in semantics.by_path
        assert "tool.op.call" in semantics.by_path
        assert "solver.op.add" in semantics.by_path
        assert "control.op.think" in semantics.by_path

    def test_get_by_path_works(self):
        semantics = create_standard_semantics()
        mapping = semantics.get_by_path("memory.op.load")
        assert mapping is not None
        assert mapping.token_str == "<LOAD_PAGE>"

    def test_arguments_registered(self):
        semantics = create_standard_semantics()
        load_mapping = semantics.get_by_id(100)
        assert load_mapping is not None
        assert "page_id" in load_mapping.arguments

        add_mapping = semantics.get_by_id(300)
        assert add_mapping is not None
        assert "a" in add_mapping.arguments
        assert "b" in add_mapping.arguments

    def test_returns_registered(self):
        semantics = create_standard_semantics()
        load_mapping = semantics.get_by_id(100)
        assert load_mapping is not None
        assert load_mapping.returns == "page_content"

        add_mapping = semantics.get_by_id(300)
        assert add_mapping is not None
        assert add_mapping.returns == "sum"

    def test_descriptions_registered(self):
        semantics = create_standard_semantics()
        mapping = semantics.get_by_id(100)
        assert mapping is not None
        assert mapping.description == "Load a memory page"
