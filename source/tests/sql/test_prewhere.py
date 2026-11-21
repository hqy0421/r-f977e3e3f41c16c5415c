from unittest import TestCase

from sqlalchemy import Column, and_, func, or_, text
from sqlalchemy import select as sa_select

from clickhouse_sqlalchemy import engines, select, Table, types
from tests.session import mocked_engine
from tests.testcase import CompilationTestCase, NativeSessionTestCase


class PreWhereCompilationTestCase(CompilationTestCase):
    def _make_table(self, *columns, **kwargs):
        engine = kwargs.pop('engine', None)
        if engine is None:
            engine = engines.ReplacingMergeTree(
                order_by=('id', 'ts'), version='val'
            )
        return Table(
            kwargs.pop('name', 't1'),
            self.metadata(),
            Column('id', types.UInt32),
            Column('ts', types.DateTime),
            Column('state', types.String),
            Column('val', types.Int32),
            *(columns + (engine,))
        )

    def test_simple_condition_moves(self):
        table = self._make_table()

        query = select(table) \
            .where(and_(table.c.id == 1, table.c.state == 'open')) \
            .final() \
            .prewhere_rewrite()

        self.assertEqual(
            self.compile(query),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 FINAL '
            'PREWHERE t1.id = %(id_1)s '
            'WHERE t1.state = %(state_1)s'
        )

    def test_multiple_key_conditions_move(self):
        table = self._make_table()

        query = select(table) \
            .where(table.c.id == 1) \
            .where(table.c.ts == '2020-01-01') \
            .where(table.c.state == 'open') \
            .prewhere_rewrite()

        self.assertEqual(
            self.compile(query),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 '
            'PREWHERE t1.id = %(id_1)s AND t1.ts = %(ts_1)s '
            'WHERE t1.state = %(state_1)s'
        )

    def test_or_of_key_columns_moves_wholesale(self):
        table = self._make_table()

        query = select(table) \
            .where(and_(
                or_(table.c.id == 1, table.c.id == 2),
                table.c.state == 'open'
            )) \
            .prewhere_rewrite()

        self.assertEqual(
            self.compile(query),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 '
            'PREWHERE t1.id = %(id_1)s OR t1.id = %(id_2)s '
            'WHERE t1.state = %(state_1)s'
        )

    def test_or_mixed_with_mutable_column_stays(self):
        table = self._make_table()

        query = select(table) \
            .where(and_(
                table.c.id == 1,
                or_(table.c.ts == '2020-01-01', table.c.state == 'open')
            )) \
            .prewhere_rewrite()

        self.assertEqual(
            self.compile(query),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 '
            'PREWHERE t1.id = %(id_1)s '
            'WHERE t1.ts = %(ts_1)s OR t1.state = %(state_1)s'
        )

    def test_or_nested_and_of_key_columns_moves(self):
        table = self._make_table()

        query = select(table) \
            .where(or_(
                and_(table.c.id == 1, table.c.ts == '2020-01-01'),
                table.c.id == 2
            )) \
            .prewhere_rewrite()

        self.assertEqual(
            self.compile(query),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 '
            'PREWHERE t1.id = %(id_1)s AND t1.ts = %(ts_1)s '
            'OR t1.id = %(id_2)s'
        )

    def test_expression_key_column_is_not_a_key_column(self):
        # ORDER BY toDate(ts): ts is only referenced by the sorting key
        # expression, it is not itself a sorting key column.
        table = self._make_table(
            engine=engines.ReplacingMergeTree(
                order_by=text('toDate(ts)'), version='val'
            )
        )

        query = select(table).where(table.c.ts == '2020-01-01') \
            .prewhere_rewrite()

        self.assertEqual(
            self.compile(query),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 '
            'WHERE t1.ts = %(ts_1)s'
        )

    def test_mixed_plain_and_expression_key(self):
        table = self._make_table(
            engine=engines.MergeTree(order_by=('id', text('toDate(ts)')))
        )

        query = select(table) \
            .where(and_(table.c.id == 1, table.c.ts == '2020-01-01')) \
            .prewhere_rewrite()

        self.assertEqual(
            self.compile(query),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 '
            'PREWHERE t1.id = %(id_1)s '
            'WHERE t1.ts = %(ts_1)s'
        )

    def test_join_is_left_untouched(self):
        table = self._make_table()
        other = self._make_table(name='t2', engine=engines.Memory())

        stmt = select(table) \
            .join(other, table.c.id == other.c.id) \
            .where(table.c.id == 1)
        query = stmt.prewhere_rewrite()

        self.assertIs(query, stmt)
        self.assertNotIn('PREWHERE', self.compile(query))

    def test_subquery_source_is_left_untouched(self):
        table = self._make_table()
        subquery = sa_select(table).subquery()

        stmt = select(subquery).where(subquery.c.id == 1)
        query = stmt.prewhere_rewrite()

        self.assertIs(query, stmt)
        self.assertNotIn('PREWHERE', self.compile(query))

    def test_non_mergetree_engine_is_left_untouched(self):
        table = self._make_table(engine=engines.Memory())

        stmt = select(table).where(table.c.id == 1)
        query = stmt.prewhere_rewrite()

        self.assertIs(query, stmt)
        self.assertNotIn('PREWHERE', self.compile(query))

    def test_column_vs_column_condition_stays(self):
        table = self._make_table()

        query = select(table) \
            .where(and_(table.c.id == 1, table.c.val == table.c.id)) \
            .prewhere_rewrite()

        self.assertEqual(
            self.compile(query),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 '
            'PREWHERE t1.id = %(id_1)s '
            'WHERE t1.val = t1.id'
        )

    def test_function_condition_stays(self):
        table = self._make_table()

        query = select(table) \
            .where(and_(
                table.c.id == 1,
                func.toDate(table.c.ts) == '2020-01-01'
            )) \
            .prewhere_rewrite()

        self.assertEqual(
            self.compile(query),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 '
            'PREWHERE t1.id = %(id_1)s '
            'WHERE toDate(t1.ts) = %(toDate_1)s'
        )

    def test_in_and_between_move(self):
        table = self._make_table()

        query = select(table) \
            .where(and_(
                table.c.id.in_([1, 2, 3]),
                table.c.ts.between('2020-01-01', '2020-02-01'),
                table.c.state == 'open'
            )) \
            .prewhere_rewrite()

        self.assertEqual(
            self.compile(query, render_postcompile=True),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 '
            'PREWHERE t1.id IN (%(id_1_1)s, %(id_1_2)s, %(id_1_3)s) '
            'AND t1.ts BETWEEN %(ts_1)s AND %(ts_2)s '
            'WHERE t1.state = %(state_1)s'
        )

    def test_nested_and_is_flattened(self):
        table = self._make_table()

        query = select(table) \
            .where(and_(
                table.c.id == 1,
                and_(table.c.ts == '2020-01-01', table.c.state == 'open')
            )) \
            .prewhere_rewrite()

        self.assertEqual(
            self.compile(query),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 '
            'PREWHERE t1.id = %(id_1)s AND t1.ts = %(ts_1)s '
            'WHERE t1.state = %(state_1)s'
        )

    def test_all_conditions_moved_leaves_no_where(self):
        table = self._make_table()

        query = select(table) \
            .where(and_(table.c.id == 1, table.c.ts == '2020-01-01')) \
            .prewhere_rewrite()

        self.assertEqual(
            self.compile(query),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 '
            'PREWHERE t1.id = %(id_1)s AND t1.ts = %(ts_1)s'
        )

    def test_existing_prewhere_is_combined(self):
        table = self._make_table()

        query = select(table) \
            .prewhere(table.c.val > 10) \
            .where(and_(table.c.id == 1, table.c.state == 'open')) \
            .prewhere_rewrite()

        self.assertEqual(
            self.compile(query),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 '
            'PREWHERE t1.val > %(val_1)s AND t1.id = %(id_1)s '
            'WHERE t1.state = %(state_1)s'
        )

    def test_no_where_returns_same_object(self):
        table = self._make_table()

        stmt = select(table).final()
        self.assertIs(stmt.prewhere_rewrite(), stmt)

    def test_nothing_movable_returns_same_object(self):
        table = self._make_table()

        stmt = select(table).where(table.c.state == 'open')
        self.assertIs(stmt.prewhere_rewrite(), stmt)

    def test_manual_prewhere(self):
        table = self._make_table()

        query = select(table) \
            .prewhere(table.c.id == 1) \
            .prewhere(table.c.ts == '2020-01-01') \
            .where(table.c.state == 'open') \
            .final()

        self.assertEqual(
            self.compile(query),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 FINAL '
            'PREWHERE t1.id = %(id_1)s AND t1.ts = %(ts_1)s '
            'WHERE t1.state = %(state_1)s'
        )

    def test_original_statement_and_params_are_preserved(self):
        table = self._make_table()

        stmt = select(table) \
            .where(and_(table.c.id == 1, table.c.state == 'open')) \
            .final()
        query = stmt.prewhere_rewrite()

        self.assertIsNot(query, stmt)
        # the original statement is unchanged
        self.assertEqual(
            self.compile(stmt),
            'SELECT t1.id, t1.ts, t1.state, t1.val '
            'FROM t1 FINAL '
            'WHERE t1.id = %(id_1)s AND t1.state = %(state_1)s'
        )
        # and the rewritten one carries the same bound parameters
        self.assertEqual(
            self._compile(stmt).params,
            self._compile(query).params,
        )

    def test_orm_prewhere(self):
        table = self._make_table()

        query = self.session.query(table.c.id) \
            .filter(table.c.state == 'open') \
            .prewhere(table.c.id == 1) \
            .final()

        self.assertEqual(
            self.compile(query),
            'SELECT t1.id AS t1_id '
            'FROM t1 FINAL '
            'PREWHERE t1.id = %(id_1)s '
            'WHERE t1.state = %(state_1)s'
        )


class PreWhereCacheTestCase(TestCase):
    """Default and rewritten queries must not share compiled cache entries."""

    def _make_table(self, metadata):
        return Table(
            't1', metadata,
            Column('id', types.UInt32),
            Column('state', types.String),
            engines.ReplacingMergeTree(order_by='id')
        )

    def test_core_cache_is_not_mixed(self):
        from sqlalchemy import MetaData
        table = self._make_table(MetaData())

        def make_queries():
            base = select(table.c.id).where(
                and_(table.c.id == 1, table.c.state == 'open')
            )
            return [
                select(table.c.id),
                select(table.c.id).final(),
                base,
                base.prewhere_rewrite(),
                select(table.c.id).prewhere(table.c.id == 1),
                select(table.c.id),
                base,
                base.prewhere_rewrite(),
            ]

        with mocked_engine() as engine:
            for query in make_queries():
                engine.session.execute(query)

        self.assertEqual(
            engine.history,
            [
                'SELECT t1.id FROM t1',
                'SELECT t1.id FROM t1 FINAL',
                'SELECT t1.id FROM t1 '
                'WHERE t1.id = %(id_1)s AND t1.state = %(state_1)s',
                'SELECT t1.id FROM t1 '
                'PREWHERE t1.id = %(id_1)s WHERE t1.state = %(state_1)s',
                'SELECT t1.id FROM t1 PREWHERE t1.id = %(id_1)s',
                'SELECT t1.id FROM t1',
                'SELECT t1.id FROM t1 '
                'WHERE t1.id = %(id_1)s AND t1.state = %(state_1)s',
                'SELECT t1.id FROM t1 '
                'PREWHERE t1.id = %(id_1)s WHERE t1.state = %(state_1)s',
            ]
        )

    def test_orm_cache_is_not_mixed(self):
        from sqlalchemy import MetaData
        table = self._make_table(MetaData())

        with mocked_engine() as engine:
            session = engine.session

            def run(query):
                # the mocked cursor returns no rows; only the emitted SQL
                # is of interest here
                try:
                    query.all()
                except Exception:
                    pass

            run(session.query(table.c.id).final())
            run(session.query(table.c.id))
            run(session.query(table.c.id).filter(table.c.id == 5).final())
            run(session.query(table.c.id).filter(table.c.id == 5))
            run(session.query(table.c.id).prewhere(table.c.id == 1))
            run(session.query(table.c.id))

        self.assertEqual(
            engine.history,
            [
                'SELECT t1.id AS t1_id FROM t1 FINAL',
                'SELECT t1.id AS t1_id FROM t1',
                'SELECT t1.id AS t1_id FROM t1 FINAL WHERE t1.id = %(id_1)s',
                'SELECT t1.id AS t1_id FROM t1 WHERE t1.id = %(id_1)s',
                'SELECT t1.id AS t1_id FROM t1 PREWHERE t1.id = %(id_1)s',
                'SELECT t1.id AS t1_id FROM t1',
            ]
        )


class PreWhereFinalTestCase(NativeSessionTestCase):
    """
    ReplacingMergeTree + FINAL: moving sorting key conditions to PREWHERE
    must not change results; mutable column conditions must stay in WHERE
    so that replaced row versions are not resurrected.
    """

    def _make_table(self):
        return Table(
            'test_prewhere_final',
            self.metadata(),
            Column('id', types.UInt32),
            Column('ver', types.UInt32),
            Column('state', types.String),
            Column('payload', types.String),
            engines.ReplacingMergeTree(order_by='id', version='ver')
        )

    def test_latest_version_state_change(self):
        table = self._make_table()

        with self.create_table(table):
            self.session.execute(table.insert(), [
                {'id': 1, 'ver': 1, 'state': 'open', 'payload': 'a' * 100},
            ])
            # a newer version of the same row changes the state
            self.session.execute(table.insert(), [
                {'id': 1, 'ver': 2, 'state': 'closed', 'payload': 'b' * 100},
            ])

            stmt = select(table.c.id, table.c.state) \
                .where(and_(table.c.id == 1, table.c.state == 'open')) \
                .final()
            query = stmt.prewhere_rewrite()

            # only the sorting key condition is moved
            self.assertEqual(
                self.compile(query),
                'SELECT test_prewhere_final.id, test_prewhere_final.state '
                'FROM test_prewhere_final FINAL '
                'PREWHERE test_prewhere_final.id = %(id_1)s '
                'WHERE test_prewhere_final.state = %(state_1)s'
            )

            # the old row version must not come back
            self.assertEqual(self.session.execute(stmt).fetchall(), [])
            self.assertEqual(self.session.execute(query).fetchall(), [])

            # the latest version is found by both forms
            stmt = select(table.c.id, table.c.state) \
                .where(and_(table.c.id == 1, table.c.state == 'closed')) \
                .final()
            query = stmt.prewhere_rewrite()
            self.assertEqual(
                self.session.execute(stmt).fetchall(), [(1, 'closed')]
            )
            self.assertEqual(
                self.session.execute(query).fetchall(), [(1, 'closed')]
            )

            # OR of key columns moves, mutable state stays in WHERE
            stmt = select(table.c.id, table.c.state) \
                .where(and_(
                    or_(table.c.id == 1, table.c.id == 2),
                    table.c.state == 'open'
                )) \
                .final()
            query = stmt.prewhere_rewrite()
            self.assertEqual(
                self.compile(query),
                'SELECT test_prewhere_final.id, test_prewhere_final.state '
                'FROM test_prewhere_final FINAL '
                'PREWHERE test_prewhere_final.id = %(id_1)s '
                'OR test_prewhere_final.id = %(id_2)s '
                'WHERE test_prewhere_final.state = %(state_1)s'
            )
            self.assertEqual(self.session.execute(query).fetchall(), [])

            # counterfactual: had the mutable state condition been moved
            # to PREWHERE, the replaced row version would be resurrected
            stale = self.session.execute(text(
                'SELECT id, ver, state FROM test_prewhere_final '
                "FINAL PREWHERE state = 'open'"
            )).fetchall()
            self.assertEqual(stale, [(1, 1, 'open')])
