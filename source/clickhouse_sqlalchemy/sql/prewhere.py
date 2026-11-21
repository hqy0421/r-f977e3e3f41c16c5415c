"""
Automatic WHERE -> PREWHERE rewriting for MergeTree-family tables.

The rewrite moves only those top-level ``WHERE`` conjuncts that are
*simple conditions* on real sorting key (``ORDER BY``) columns compared
against bound values.  Such conditions are version-invariant for
``ReplacingMergeTree`` (every version of a row shares the same sorting
key values), so evaluating them in ``PREWHERE`` -- before ``FINAL``
merging -- cannot change the result, while letting ClickHouse skip
reading the wide non-key columns for filtered-out rows.

Conditions on mutable columns (e.g. a ``state`` column) must never be
moved: with ``FINAL`` that would filter row versions *before* the merge
and could resurrect rows that a newer version has already replaced.
"""

from sqlalchemy.sql import operators
from sqlalchemy.sql.elements import (
    BinaryExpression,
    BooleanClauseList,
    ColumnClause,
    Grouping,
    TextClause,
)
from sqlalchemy.sql.functions import FunctionElement
from sqlalchemy.sql.schema import Column, Table
from sqlalchemy.sql.selectable import SelectBase

from ..engines import MergeTree

# Comparison operators that are safe to evaluate in PREWHERE when their
# column side is a real sorting key column.
_SIMPLE_OPERATORS = frozenset({
    operators.eq,
    operators.ne,
    operators.lt,
    operators.le,
    operators.gt,
    operators.ge,
    operators.in_op,
    operators.not_in_op,
    operators.between_op,
})

_AND_OR_OPERATORS = frozenset({operators.and_, operators.or_})


def _walk(clause):
    """Yield the clause itself and every nested clause element."""
    seen = set()
    stack = [clause]
    while stack:
        elem = stack.pop()
        if id(elem) in seen:
            continue
        seen.add(id(elem))
        yield elem
        stack.extend(elem.get_children())


def _is_bound_value(expr):
    """True if *expr* contains only bound values/literals.

    Anything that references a column, a SQL function, a subquery or raw
    text is not a plain bound value and disqualifies the condition.
    """
    for elem in _walk(expr):
        if isinstance(
                elem, (ColumnClause, FunctionElement, SelectBase, TextClause)
        ):
            return False
    return True


def _is_key_column(expr, table, key_column_names):
    return (
        isinstance(expr, Column)
        and expr.table is table
        and expr.name in key_column_names
    )


def _is_simple_key_condition(clause, table, key_column_names):
    """A comparison of a real sorting key column to a bound value."""
    if not isinstance(clause, BinaryExpression):
        return False
    if clause.operator not in _SIMPLE_OPERATORS:
        return False
    if _is_key_column(clause.left, table, key_column_names):
        return _is_bound_value(clause.right)
    if _is_key_column(clause.right, table, key_column_names):
        return _is_bound_value(clause.left)
    return False


def _unwrap(clause):
    """Peel off transparent grouping parentheses."""
    while isinstance(clause, Grouping):
        clause = clause.element
    return clause


def _flatten(clause, operators_):
    """Yield leaf clauses of nested boolean clauses with given operators."""
    clause = _unwrap(clause)
    if (isinstance(clause, BooleanClauseList)
            and clause.operator in operators_):
        for subclause in clause.clauses:
            for leaf in _flatten(subclause, operators_):
                yield leaf
    else:
        yield clause


def _is_movable_or(clause, table, key_column_names):
    """An OR is movable only if every leaf inside it is a simple key
    condition; an OR that mixes in mutable columns stays in WHERE."""
    if not (isinstance(clause, BooleanClauseList)
            and clause.operator is operators.or_):
        return False
    return all(
        _is_simple_key_condition(leaf, table, key_column_names)
        for leaf in _flatten(clause, _AND_OR_OPERATORS)
    )


def _single_table_source(select_stmt):
    """Return the single real table the statement reads from, or None.

    Anything that is not exactly one plain table (JOINs, subqueries,
    aliases, multiple sources, ...) makes the source unclear and the
    statement is left untouched.
    """
    froms = select_stmt.get_final_froms()
    if len(froms) != 1:
        return None
    from_clause = froms[0]
    if not isinstance(from_clause, Table):
        return None
    return from_clause


def _sorting_key_column_names(table):
    """Names of the table's real sorting key (ORDER BY) columns.

    A column that is merely *referenced* by a sorting key expression
    (e.g. ``ORDER BY toDate(ts)``) is not itself a sorting key column and
    is not included.
    """
    engine = getattr(table, 'engine', None)
    if not isinstance(engine, MergeTree):
        return None
    order_by = engine.order_by
    if order_by is None:
        return None
    names = set()
    for expr in order_by.get_expressions_or_columns():
        if isinstance(expr, Column) and expr.table is table:
            names.add(expr.name)
    return names


def extract_prewhere_criteria(select_stmt):
    """Split a select's WHERE criteria into (kept, prewhere) tuples.

    Returns ``None`` when the rewrite does not apply (no single clear
    MergeTree source, no usable sorting key columns, or nothing
    movable), in which case the statement should be used as is.
    """
    where_criteria = getattr(select_stmt, '_where_criteria', None)
    if not where_criteria:
        return None

    table = _single_table_source(select_stmt)
    if table is None:
        return None

    key_column_names = _sorting_key_column_names(table)
    if not key_column_names:
        return None

    kept = []
    moved = []
    for criterion in where_criteria:
        for conjunct in _flatten(criterion, {operators.and_}):
            if (_is_simple_key_condition(
                    conjunct, table, key_column_names)
                    or _is_movable_or(conjunct, table, key_column_names)):
                moved.append(conjunct)
            else:
                kept.append(conjunct)

    if not moved:
        return None
    return tuple(kept), tuple(moved)
