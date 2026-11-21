from sqlalchemy.sql import coercions, roles
from sqlalchemy.sql.base import _generative
from sqlalchemy.sql.selectable import (
    Select as StandardSelect,
)
from sqlalchemy.sql.visitors import InternalTraversal

from ..ext.clauses import (
    ArrayJoin,
    LeftArrayJoin,
    LimitByClause,
    sample_clause,
)
from .prewhere import extract_prewhere_criteria


__all__ = ('Select', 'select')


class Select(StandardSelect):
    inherit_cache = True

    _with_cube = False
    _with_rollup = False
    _with_totals = False
    _final_clause = None
    _sample_clause = None
    _limit_by_clause = None
    _array_join = None
    _prewhere_criteria = ()

    # ClickHouse-specific state must take part in the statement cache key,
    # otherwise compiled SQL of e.g. a FINAL or PREWHERE statement could be
    # served from cache for the default statement and vice versa.
    _ch_traverse_internals = [
        ('_with_cube', InternalTraversal.dp_boolean),
        ('_with_rollup', InternalTraversal.dp_boolean),
        ('_with_totals', InternalTraversal.dp_boolean),
        ('_final_clause', InternalTraversal.dp_boolean),
        ('_sample_clause', InternalTraversal.dp_clauseelement),
        ('_limit_by_clause', InternalTraversal.dp_clauseelement),
        ('_array_join', InternalTraversal.dp_clauseelement),
        ('_prewhere_criteria', InternalTraversal.dp_clauseelement_tuple),
    ]
    _traverse_internals = (
        StandardSelect._traverse_internals + _ch_traverse_internals
    )
    _cache_key_traversal = (
        StandardSelect._cache_key_traversal + _ch_traverse_internals
    )

    @_generative
    def with_cube(self):
        self._with_cube = True
        return self

    @_generative
    def with_rollup(self):
        self._with_rollup = True
        return self

    @_generative
    def with_totals(self):
        self._with_totals = True
        return self

    @_generative
    def final(self):
        self._final_clause = True
        return self

    @_generative
    def sample(self, sample):
        self._sample_clause = sample_clause(sample)
        return self

    @_generative
    def limit_by(self, by_clauses, limit, offset=None):
        self._limit_by_clause = LimitByClause(by_clauses, limit, offset)
        return self

    @_generative
    def prewhere(self, *criteria):
        """Return a new select with the given PREWHERE criteria added.

        The criteria are rendered as a PREWHERE clause and are joined to
        the existing PREWHERE clause via AND, if any.
        """
        for criterion in criteria:
            criterion = coercions.expect(
                roles.WhereHavingRole, criterion, apply_propagate_attrs=self
            )
            self._prewhere_criteria += (criterion,)
        return self

    def prewhere_rewrite(self):
        """Return a new select with eligible WHERE conditions moved to
        PREWHERE.

        Only simple conditions on real sorting key (ORDER BY) columns of
        a single MergeTree-family table are moved; everything else --
        conditions on mutable columns, ORs mixing such columns, JOINs or
        unclear sources -- stays in WHERE.  When the rewrite does not
        apply the statement is returned unchanged.

        The original select object and its bound parameters are left
        untouched; the returned statement shares the same parameter
        objects and differs from the original in its compilation cache
        key.
        """
        extracted = extract_prewhere_criteria(self)
        if extracted is None:
            return self

        kept, moved = extracted
        new_select = self._generate()
        new_select._where_criteria = kept
        new_select._prewhere_criteria = self._prewhere_criteria + moved
        return new_select

    def _add_array_join(self, columns, left):
        join_type = ArrayJoin if not left else LeftArrayJoin
        self._array_join = join_type(*columns)

    @_generative
    def array_join(self, *columns, **kwargs):
        left = kwargs.get("left", False)
        self._add_array_join(columns, left=left)
        return self

    @_generative
    def left_array_join(self, *columns):
        self._add_array_join(columns, left=True)
        return self

    def join(self, right, onclause=None, isouter=False, full=False, type=None,
             strictness=None, distribution=None):
        flags = tuple({
            'full': full,
            'type': type,
            'strictness': strictness,
            'distribution': distribution
        }.items())
        return super().join(right, onclause=onclause, isouter=isouter,
                            full=flags)


select = Select
