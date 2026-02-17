"""Test academic metrics (Node F1, Edge F1, edit distance)."""
from agentgate import AgentTrace, AgentStep, StepKind
from agentgate.metrics import node_f1, edge_f1, tool_edit_distance


def _trace(tools: list[str]) -> AgentTrace:
    return AgentTrace(
        input="test",
        steps=[AgentStep(kind=StepKind.TOOL_CALL, name=t) for t in tools],
    )


# --- Node F1 ---

def test_node_f1_exact_match():
    assert node_f1(_trace(["search", "book"]), ["search", "book"]) == 1.0


def test_node_f1_partial():
    # actual has search+book, expected has search+book+pay
    # TP=2, precision=2/2=1.0, recall=2/3=0.67, F1=0.8
    score = node_f1(_trace(["search", "book"]), ["search", "book", "pay"])
    assert 0.79 < score < 0.81


def test_node_f1_wrong_tools():
    assert node_f1(_trace(["delete"]), ["search", "book"]) == 0.0


def test_node_f1_empty():
    assert node_f1(_trace([]), []) == 1.0


def test_node_f1_extra_tools():
    # actual has search+book+spam, expected has search+book
    # TP=2, precision=2/3=0.67, recall=2/2=1.0, F1=0.8
    score = node_f1(_trace(["search", "book", "spam"]), ["search", "book"])
    assert 0.79 < score < 0.81


# --- Edge F1 ---

def test_edge_f1_exact():
    assert edge_f1(_trace(["a", "b", "c"]), ["a", "b", "c"]) == 1.0


def test_edge_f1_reversed():
    # actual: a→b, expected: b→a — no overlap
    assert edge_f1(_trace(["a", "b"]), ["b", "a"]) == 0.0


def test_edge_f1_partial():
    # actual: a→b→c, edges: (a,b),(b,c)
    # expected: a→b→d, edges: (a,b),(b,d)
    # TP=1 ((a,b)), precision=1/2, recall=1/2, F1=0.5
    assert edge_f1(_trace(["a", "b", "c"]), ["a", "b", "d"]) == 0.5


def test_edge_f1_single_tool():
    # No edges with single tools
    assert edge_f1(_trace(["a"]), ["a"]) == 1.0


# --- Edit Distance ---

def test_edit_distance_exact():
    assert tool_edit_distance(_trace(["a", "b", "c"]), ["a", "b", "c"]) == 0.0


def test_edit_distance_completely_different():
    assert tool_edit_distance(_trace(["x"]), ["a", "b", "c"]) == 1.0


def test_edit_distance_one_swap():
    # ["a", "c", "b"] vs ["a", "b", "c"] — 2 substitutions / 3 = 0.67
    dist = tool_edit_distance(_trace(["a", "c", "b"]), ["a", "b", "c"])
    assert 0.6 < dist < 0.7


def test_edit_distance_empty():
    assert tool_edit_distance(_trace([]), []) == 0.0
