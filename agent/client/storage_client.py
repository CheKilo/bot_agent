# -*- coding: utf-8 -*-
"""
Storage gRPC 客户端

提供 MySQL 和 Milvus 存储服务的 Python 客户端封装。
"""

import grpc
from typing import Any, Dict, List, Optional, Union, Iterator

from agent.pb import storage_pb2
from agent.pb import storage_pb2_grpc


class StorageRequestError(Exception):
    """Storage 请求错误"""

    def __init__(self, message: str, error: str = ""):
        self.message = message
        self.error = error
        super().__init__(f"{message}: {error}" if error else message)


class StorageClient:
    """
    Storage gRPC 客户端

    提供 MySQL 和 Milvus 存储操作的封装。

    使用示例：
        with StorageClient("localhost:50051") as client:
            # MySQL 操作
            result = client.execute([
                client.insert_op("test_db", "users", [{"id": 1, "name": "Alice"}])
            ])

            # Milvus 向量操作
            result = client.execute_vector([
                client.vector_insert_op("collection", "partition", vectors)
            ])
    """

    def __init__(self, address: str = "localhost:50051"):
        """
        初始化客户端

        Args:
            address: gRPC 服务地址
        """
        self.address = address
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[storage_pb2_grpc.StorageServiceStub] = None

    def connect(self):
        """建立连接"""
        self._channel = grpc.insecure_channel(self.address)
        self._stub = storage_pb2_grpc.StorageServiceStub(self._channel)

    def close(self):
        """关闭连接"""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ========================================================================
    # 类型转换辅助方法
    # ========================================================================

    @staticmethod
    def _to_typed_value(value: Any) -> storage_pb2.TypedValue:
        """将 Python 值转换为 TypedValue"""
        tv = storage_pb2.TypedValue()

        if value is None:
            tv.null_value = storage_pb2.NULL_VALUE
        elif isinstance(value, bool):
            tv.bool_value = value
        elif isinstance(value, int):
            tv.int_value = value
        elif isinstance(value, float):
            tv.double_value = value
        elif isinstance(value, str):
            tv.string_value = value
        elif isinstance(value, bytes):
            tv.bytes_value = value
        else:
            # 默认转为字符串
            tv.string_value = str(value)

        return tv

    @staticmethod
    def _from_typed_value(tv: storage_pb2.TypedValue) -> Any:
        """将 TypedValue 转换为 Python 值"""
        which = tv.WhichOneof("value")

        if which == "string_value":
            return tv.string_value
        elif which == "int_value":
            return tv.int_value
        elif which == "double_value":
            return tv.double_value
        elif which == "bool_value":
            return tv.bool_value
        elif which == "bytes_value":
            return tv.bytes_value
        elif which == "timestamp_value":
            return tv.timestamp_value
        elif which == "null_value":
            return None
        else:
            return None

    @staticmethod
    def _dict_to_typed_values(d: Dict[str, Any]) -> Dict[str, storage_pb2.TypedValue]:
        """将字典转换为 TypedValue 字典"""
        return {k: StorageClient._to_typed_value(v) for k, v in d.items()}

    @staticmethod
    def _copy_to_map(target_map, source_dict: Dict[str, Any]):
        """将字典复制到 protobuf map 字段（正确处理 Message 对象赋值）"""
        for k, v in source_dict.items():
            target_map[k].CopyFrom(StorageClient._to_typed_value(v))

    @staticmethod
    def _typed_values_to_dict(tvs: Dict[str, storage_pb2.TypedValue]) -> Dict[str, Any]:
        """将 TypedValue 字典转换为普通字典"""
        return {k: StorageClient._from_typed_value(v) for k, v in tvs.items()}

    # ========================================================================
    # MySQL 操作构建器
    # ========================================================================

    def insert_op(
        self, database: str, table: str, rows: List[Dict[str, Any]]
    ) -> storage_pb2.Operation:
        """
        构建插入操作

        Args:
            database: 数据库名
            table: 表名
            rows: 要插入的行数据列表

        Returns:
            Operation 对象
        """
        insert_rows = []
        for row in rows:
            insert_row = storage_pb2.InsertRow(fields=self._dict_to_typed_values(row))
            insert_rows.append(insert_row)

        return storage_pb2.Operation(
            database=database,
            table=table,
            insert=storage_pb2.InsertOperation(rows=insert_rows),
        )

    def update_op(
        self,
        database: str,
        table: str,
        set_fields: Optional[Dict[str, Any]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        raw_clause: str = "",
        raw_params: Optional[List[Any]] = None,
        raw_set: str = "",
        raw_set_params: Optional[List[Any]] = None,
    ) -> storage_pb2.Operation:
        """
        构建更新操作

        Args:
            database: 数据库名
            table: 表名
            set_fields: 要更新的字段（简单赋值）
            conditions: WHERE 条件（简单等值条件）
            raw_clause: 复杂 WHERE 子句
            raw_params: 复杂条件的参数
            raw_set: 原始 SET 子句，支持 SQL 表达式（如 "access_count = access_count + 1"）
            raw_set_params: raw_set 中占位符(?)对应的参数

        Returns:
            Operation 对象

        注意:
            当 raw_set 不为空时，优先使用 raw_set，忽略 set_fields
        """
        where = storage_pb2.WhereClause()
        if conditions:
            self._copy_to_map(where.conditions, conditions)
        if raw_clause:
            where.raw_clause = raw_clause
        if raw_params:
            where.raw_params.extend([self._to_typed_value(p) for p in raw_params])

        update_op = storage_pb2.UpdateOperation(where=where)

        # 优先使用 raw_set（支持 SQL 表达式）
        if raw_set:
            update_op.raw_set = raw_set
            if raw_set_params:
                update_op.raw_set_params.extend(
                    [self._to_typed_value(p) for p in raw_set_params]
                )
        elif set_fields:
            self._copy_to_map(update_op.set_fields, set_fields)

        return storage_pb2.Operation(
            database=database,
            table=table,
            update=update_op,
        )

    def delete_op(
        self,
        database: str,
        table: str,
        conditions: Optional[Dict[str, Any]] = None,
        raw_clause: str = "",
        raw_params: Optional[List[Any]] = None,
    ) -> storage_pb2.Operation:
        """
        构建删除操作

        Args:
            database: 数据库名
            table: 表名
            conditions: WHERE 条件（简单等值条件）
            raw_clause: 复杂 WHERE 子句
            raw_params: 复杂条件的参数

        Returns:
            Operation 对象
        """
        where = storage_pb2.WhereClause()
        if conditions:
            self._copy_to_map(where.conditions, conditions)
        if raw_clause:
            where.raw_clause = raw_clause
        if raw_params:
            where.raw_params.extend([self._to_typed_value(p) for p in raw_params])

        return storage_pb2.Operation(
            database=database,
            table=table,
            delete=storage_pb2.DeleteOperation(where=where),
        )

    def select_op(
        self,
        database: str,
        table: str,
        fields: Optional[List[str]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        raw_clause: str = "",
        raw_params: Optional[List[Any]] = None,
        order_by: Optional[str] = None,
        descending: bool = False,
        limit: int = 0,
        offset: int = 0,
    ) -> storage_pb2.Operation:
        """
        构建查询操作

        Args:
            database: 数据库名
            table: 表名
            fields: 要返回的字段（空则返回全部）
            conditions: WHERE 条件（简单等值条件）
            raw_clause: 复杂 WHERE 子句
            raw_params: 复杂条件的参数
            order_by: 排序字段
            descending: 是否降序
            limit: 限制数量
            offset: 偏移量

        Returns:
            Operation 对象
        """
        select = storage_pb2.SelectOperation()

        if fields:
            select.fields.extend(fields)

        # WHERE 条件
        if conditions or raw_clause:
            where = storage_pb2.WhereClause()
            if conditions:
                self._copy_to_map(where.conditions, conditions)
            if raw_clause:
                where.raw_clause = raw_clause
            if raw_params:
                where.raw_params.extend([self._to_typed_value(p) for p in raw_params])
            select.where.CopyFrom(where)

        # 排序
        if order_by:
            select.order_by.CopyFrom(
                storage_pb2.OrderBy(field=order_by, descending=descending)
            )

        # 分页
        if limit > 0 or offset > 0:
            select.pagination.CopyFrom(
                storage_pb2.Pagination(limit=limit, offset=offset)
            )

        return storage_pb2.Operation(database=database, table=table, select=select)

    # ========================================================================
    # MySQL 执行方法
    # ========================================================================

    def execute(
        self, operations: List[storage_pb2.Operation], use_transaction: bool = False
    ) -> storage_pb2.ExecuteResponse:
        """
        执行 MySQL 操作

        Args:
            operations: 操作列表
            use_transaction: 是否使用事务

        Returns:
            ExecuteResponse 响应

        Raises:
            StorageRequestError: 请求失败时抛出
        """
        if not self._stub:
            raise StorageRequestError(
                "Client not connected, please call connect() first"
            )

        request = storage_pb2.ExecuteRequest(
            operations=operations, use_transaction=use_transaction
        )

        try:
            response = self._stub.Execute(request)
            return response
        except grpc.RpcError as e:
            raise StorageRequestError(
                "MySQL execute request failed",
                f"code={e.code()}, details={e.details()}",
            )

    # ========================================================================
    # Milvus 向量操作构建器
    # ========================================================================

    def vector_insert_op(
        self, collection: str, partition: str, vectors: List[Dict[str, Any]]
    ) -> storage_pb2.VectorOperation:
        """
        构建向量插入操作

        Args:
            collection: Collection 名称
            partition: Partition 名称
            vectors: 向量数据列表，每个元素包含:
                - id: 向量 ID
                - vector: 向量数据（List[float]）
                - metadata: 元数据（可选）

        Returns:
            VectorOperation 对象
        """
        vector_data_list = []
        for v in vectors:
            vd = storage_pb2.VectorData(id=v.get("id", ""), vector=v.get("vector", []))
            if "metadata" in v and v["metadata"]:
                self._copy_to_map(vd.metadata, v["metadata"])
            vector_data_list.append(vd)

        return storage_pb2.VectorOperation(
            collection=collection,
            partition=partition,
            insert=storage_pb2.VectorInsertOperation(vectors=vector_data_list),
        )

    def vector_upsert_op(
        self, collection: str, partition: str, vectors: List[Dict[str, Any]]
    ) -> storage_pb2.VectorOperation:
        """
        构建向量 Upsert 操作（存在则更新，否则插入）

        Args:
            collection: Collection 名称
            partition: Partition 名称
            vectors: 向量数据列表

        Returns:
            VectorOperation 对象
        """
        vector_data_list = []
        for v in vectors:
            vd = storage_pb2.VectorData(id=v.get("id", ""), vector=v.get("vector", []))
            if "metadata" in v and v["metadata"]:
                self._copy_to_map(vd.metadata, v["metadata"])
            vector_data_list.append(vd)

        return storage_pb2.VectorOperation(
            collection=collection,
            partition=partition,
            upsert=storage_pb2.VectorUpsertOperation(vectors=vector_data_list),
        )

    def vector_search_op(
        self,
        collection: str,
        partition: str,
        query_vector: List[float],
        top_k: int = 10,
        min_score: float = 0.0,
        filter_conditions: Optional[Dict[str, Any]] = None,
        filter_expr: str = "",
        output_fields: Optional[List[str]] = None,
    ) -> storage_pb2.VectorOperation:
        """
        构建向量搜索操作

        Args:
            collection: Collection 名称
            partition: Partition 名称
            query_vector: 查询向量
            top_k: 返回数量
            min_score: 最小相似度阈值
            filter_conditions: 简单过滤条件
            filter_expr: 复杂过滤表达式
            output_fields: 要返回的字段

        Returns:
            VectorOperation 对象
        """
        search = storage_pb2.VectorSearchOperation(
            query_vector=query_vector, top_k=top_k, min_score=min_score
        )

        if filter_conditions:
            self._copy_to_map(search.filter, filter_conditions)
        if filter_expr:
            search.filter_expr = filter_expr
        if output_fields:
            search.output_fields.extend(output_fields)

        return storage_pb2.VectorOperation(
            collection=collection, partition=partition, search=search
        )

    def vector_delete_op(
        self,
        collection: str,
        partition: str,
        ids: Optional[List[str]] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        filter_expr: str = "",
    ) -> storage_pb2.VectorOperation:
        """
        构建向量删除操作

        Args:
            collection: Collection 名称
            partition: Partition 名称
            ids: 要删除的向量 ID 列表
            filter_conditions: 简单过滤条件
            filter_expr: 复杂过滤表达式

        Returns:
            VectorOperation 对象
        """
        delete = storage_pb2.VectorDeleteOperation()

        if ids:
            delete.ids.extend(ids)
        if filter_conditions:
            self._copy_to_map(delete.filter, filter_conditions)
        if filter_expr:
            delete.filter_expr = filter_expr

        return storage_pb2.VectorOperation(
            collection=collection, partition=partition, delete=delete
        )

    # ========================================================================
    # Milvus 执行方法
    # ========================================================================

    def execute_vector(
        self, operations: List[storage_pb2.VectorOperation]
    ) -> storage_pb2.ExecuteVectorResponse:
        """
        执行 Milvus 向量操作

        Args:
            operations: 向量操作列表

        Returns:
            ExecuteVectorResponse 响应

        Raises:
            StorageRequestError: 请求失败时抛出
        """
        if not self._stub:
            raise StorageRequestError(
                "Client not connected, please call connect() first"
            )

        request = storage_pb2.ExecuteVectorRequest(operations=operations)

        try:
            response = self._stub.ExecuteVector(request)
            return response
        except grpc.RpcError as e:
            raise StorageRequestError(
                "Milvus execute request failed",
                f"code={e.code()}, details={e.details()}",
            )

    # ========================================================================
    # 便捷方法
    # ========================================================================

    def insert(
        self,
        database: str,
        table: str,
        rows: List[Dict[str, Any]],
        use_transaction: bool = False,
    ) -> storage_pb2.ExecuteResponse:
        """便捷插入方法"""
        return self.execute([self.insert_op(database, table, rows)], use_transaction)

    def update(
        self,
        database: str,
        table: str,
        set_fields: Optional[Dict[str, Any]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        raw_set: str = "",
        raw_set_params: Optional[List[Any]] = None,
        use_transaction: bool = False,
    ) -> storage_pb2.ExecuteResponse:
        """
        便捷更新方法

        Args:
            database: 数据库名
            table: 表名
            set_fields: 要更新的字段（简单赋值）
            conditions: WHERE 条件
            raw_set: 原始 SET 子句，支持 SQL 表达式（如 "access_count = access_count + 1"）
            raw_set_params: raw_set 中占位符(?)对应的参数
            use_transaction: 是否使用事务
        """
        return self.execute(
            [
                self.update_op(
                    database,
                    table,
                    set_fields,
                    conditions,
                    raw_set=raw_set,
                    raw_set_params=raw_set_params,
                )
            ],
            use_transaction,
        )

    def delete(
        self,
        database: str,
        table: str,
        conditions: Optional[Dict[str, Any]] = None,
        use_transaction: bool = False,
    ) -> storage_pb2.ExecuteResponse:
        """便捷删除方法"""
        return self.execute(
            [self.delete_op(database, table, conditions)], use_transaction
        )

    def select(
        self,
        database: str,
        table: str,
        fields: Optional[List[str]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        raw_clause: str = "",
        raw_params: Optional[List[Any]] = None,
        order_by: Optional[str] = None,
        descending: bool = False,
        limit: int = 0,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        便捷查询方法，直接返回字典列表

        Returns:
            查询结果的字典列表
        """
        response = self.execute(
            [
                self.select_op(
                    database,
                    table,
                    fields,
                    conditions,
                    raw_clause,
                    raw_params,
                    order_by=order_by,
                    descending=descending,
                    limit=limit,
                    offset=offset,
                )
            ]
        )

        results = []
        for result in response.results:
            select_result = result.select_result
            if select_result:
                for row in select_result.rows:
                    results.append(self._typed_values_to_dict(row.fields))

        return results

    def vector_insert(
        self, collection: str, partition: str, vectors: List[Dict[str, Any]]
    ) -> int:
        """
        便捷向量插入方法

        Returns:
            插入的向量数量
        """
        response = self.execute_vector(
            [self.vector_insert_op(collection, partition, vectors)]
        )

        for result in response.results:
            if result.insert_result:
                return result.insert_result.inserted_count
        return 0

    def vector_search(
        self,
        collection: str,
        partition: str,
        query_vector: List[float],
        top_k: int = 10,
        min_score: float = 0.0,
        filter_expr: str = "",
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        便捷向量搜索方法

        Returns:
            搜索结果列表，每个元素包含 id, score, metadata
        """
        response = self.execute_vector(
            [
                self.vector_search_op(
                    collection,
                    partition,
                    query_vector,
                    top_k,
                    min_score,
                    filter_expr=filter_expr,
                    output_fields=output_fields,
                )
            ]
        )

        results = []
        for result in response.results:
            if result.search_result:
                for match in result.search_result.matches:
                    results.append(
                        {
                            "id": match.id,
                            "score": match.score,
                            "metadata": self._typed_values_to_dict(match.metadata),
                        }
                    )

        return results

    def vector_delete(self, collection: str, partition: str, ids: List[str]) -> int:
        """
        便捷向量删除方法

        Returns:
            删除的向量数量
        """
        response = self.execute_vector(
            [self.vector_delete_op(collection, partition, ids)]
        )

        for result in response.results:
            if result.delete_result:
                return result.delete_result.deleted_count
        return 0
