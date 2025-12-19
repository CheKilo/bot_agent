package storage

import (
	"bot_agent/gateway/internal/logger"
	"bot_agent/gateway/internal/pb"
	"context"
	"database/sql"
	"fmt"
	"strings"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

// 对于调用链中的某个批量操作，默认是对相同列进行批量插入，如果涉及到不同列，应该于调用链上再追加一个批量操作
type MySQLClient struct {
    db *sql.DB
} 

// 辅助接口, 使得事务和非事务执行可以共用同一套方法
type executor interface {
    ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error)
    QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error)
} 

func NewMySQLClient(dsn string) (*MySQLClient, error) {
	db, err := sql.Open("mysql", dsn)
	if err != nil {
		return nil, fmt.Errorf("sql open error: %w", err)
	}
	
	
	if err := db.Ping(); err != nil {
		db.Close()
		return nil, fmt.Errorf("sql ping error: %w", err)
	}
	return &MySQLClient{db: db}, nil
}

func (c *MySQLClient) Close() error {
	return c.db.Close()
}

func (c *MySQLClient) ExecuteBatch(ctx context.Context, ops []*pb.Operation, useTx bool) (results []*pb.OperationResult, err error)  {
	var exec executor
	var tx *sql.Tx

	// 根据是否使用事务, 初始化执行器
	if useTx {
		tx, err = c.db.BeginTx(ctx, nil)
		if err != nil {
			return nil, fmt.Errorf("begin tx error: %w", err)
		}
		// 出现异常时, 回滚事务
		defer func(){
			if err != nil && tx != nil {
				if rbErr := tx.Rollback(); rbErr != nil {
					logger.Warn("transaction rollback failed: %v (original error: %v)", rbErr, err)
				} else {
					logger.Warn("transaction rolled back due to error: %v", err)
				}
			}
		}()
		
		exec = tx
	} else {
		exec = c.db
	}

	results = make([]*pb.OperationResult, len(ops))

	for i, op := range ops {
		var res *pb.OperationResult

		// 使用 oneof 类型断言判断操作类型
		switch op.GetOperation().(type) {
		case *pb.Operation_Insert:
			res = c.executeInsert(ctx, exec, op, i)
		case *pb.Operation_Update:
			res = c.executeUpdate(ctx, exec, op, i)
		case *pb.Operation_Delete:
			res = c.executeDelete(ctx, exec, op, i)
		case *pb.Operation_Select:
			res = c.executeSelect(ctx, exec, op, i)
		default:
			return nil, fmt.Errorf("unknown operation type")
		}

		if !res.Success && useTx {
			// 返回错误，defer 会自动执行 Rollback
			return nil, fmt.Errorf("execute operation error: %s", res.Error)
		}
		results[i] = res
	}

	// 事务成功，提交
	if useTx {
		if err = tx.Commit(); err != nil {
			return nil, fmt.Errorf("commit tx error: %w", err)
		}
	}

	return results, nil
}

// 私有方法, 提供CRUD功能给ExecuteBatch 调用
func (c *MySQLClient) executeInsert(ctx context.Context, exec executor, op *pb.Operation, index int) *pb.OperationResult {
	database := op.GetDatabase()
	if database == "" {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "database is empty",
		}
	}
	table := op.GetTable()
	if table == "" {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "table is empty",
		}
	}

	insertOp := op.GetInsert()
	if insertOp == nil || len(insertOp.Rows) == 0 {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "insert rows is empty",
		}
	}

	// 从第一行获取列名（约定所有行列一致，由调用方保证）
	firstRow := insertOp.Rows[0]
	if len(firstRow.Fields) == 0 {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "first row has no fields",
		}
	}

	// 提取列名并保证顺序一致
	columns := make([]string, 0, len(firstRow.Fields))
	for col := range firstRow.Fields {
		columns = append(columns, col)
	}

	// 构建批量 VALUES 占位符和参数
	rowPlaceholders := make([]string, len(insertOp.Rows))
	values := make([]interface{}, 0, len(insertOp.Rows)*len(columns))

	singleRowPlaceholder := "(" + strings.Repeat("?, ", len(columns)-1) + "?)"
	for i, row := range insertOp.Rows {
		rowPlaceholders[i] = singleRowPlaceholder
		// 按照 columns 的顺序取值，保证顺序一致
		for _, col := range columns {
			values = append(values, extractTypedValue(row.Fields[col]))
		}
	}

	// 构建批量 INSERT 语句: INSERT INTO db.table (a, b) VALUES (?, ?), (?, ?)
	query := fmt.Sprintf("INSERT INTO %s.%s (%s) VALUES %s",
		database,
		table,
		strings.Join(columns, ", "),
		strings.Join(rowPlaceholders, ", "))

	result, err := exec.ExecContext(ctx, query, values...)
	if err != nil {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("batch insert error: %v", err),
		}
	}

	affected, _ := result.RowsAffected()

	return &pb.OperationResult{
		Index:   int32(index),
		Success: true,
		Result: &pb.OperationResult_InsertResult{
			InsertResult: &pb.InsertResult{
				InsertedCount: int32(affected),
			},
		},
	}
}

func (c *MySQLClient) executeUpdate(ctx context.Context, exec executor, op *pb.Operation, index int) *pb.OperationResult {
	database := op.GetDatabase()
	if database == "" {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "database is empty",
		}
	}
	table := op.GetTable()
	if table == "" {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "table is empty",
		}
	}

	updateOp := op.GetUpdate()
	if updateOp == nil {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "update operation is nil",
		}
	}

	// 构建 SET 子句
	var setClause string
	var values []interface{}

	// 优先使用 raw_set（支持 SQL 表达式，如 access_count = access_count + 1）
	if updateOp.RawSet != "" {
		setClause = updateOp.RawSet
		for _, p := range updateOp.RawSetParams {
			values = append(values, extractTypedValue(p))
		}
	} else if len(updateOp.SetFields) > 0 {
		// 使用简单字段赋值
		setClauses := make([]string, 0, len(updateOp.SetFields))
		for col, typedVal := range updateOp.SetFields {
			setClauses = append(setClauses, fmt.Sprintf("%s = ?", col))
			values = append(values, extractTypedValue(typedVal))
		}
		setClause = strings.Join(setClauses, ", ")
	} else {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "update fields is empty (neither set_fields nor raw_set provided)",
		}
	}

	// 构建 WHERE 子句
	whereClause, whereParams := buildWhereClause(updateOp.Where)
	values = append(values, whereParams...)

	query := fmt.Sprintf("UPDATE %s.%s SET %s%s",
		database,
		table,
		setClause,
		whereClause)

	result, err := exec.ExecContext(ctx, query, values...)
	if err != nil {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("update error: %v", err),
		}
	}

	affected, _ := result.RowsAffected()

	return &pb.OperationResult{
		Index:   int32(index),
		Success: true,
		Result: &pb.OperationResult_UpdateResult{
			UpdateResult: &pb.UpdateResult{
				AffectedRows: int32(affected),
			},
		},
	}
}

func (c *MySQLClient) executeDelete(ctx context.Context, exec executor, op *pb.Operation, index int) *pb.OperationResult {
	database := op.GetDatabase()
	if database == "" {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "database is empty",
		}
	}
	table := op.GetTable()
	if table == "" {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "table is empty",
		}
	}

	deleteOp := op.GetDelete()
	if deleteOp == nil || deleteOp.Where == nil {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "where clause is required for delete (to prevent accidental full table deletion)",
		}
	}

	// 构建 WHERE 子句
	whereClause, whereParams := buildWhereClause(deleteOp.Where)
	if whereClause == "" {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "where clause cannot be empty for delete",
		}
	}

	query := fmt.Sprintf("DELETE FROM %s.%s%s", database, table, whereClause)

	result, err := exec.ExecContext(ctx, query, whereParams...)
	if err != nil {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("delete error: %v", err),
		}
	}

	affected, _ := result.RowsAffected()

	return &pb.OperationResult{
		Index:   int32(index),
		Success: true,
		Result: &pb.OperationResult_DeleteResult{
			DeleteResult: &pb.DeleteResult{
				AffectedRows: int32(affected),
			},
		},
	}
}

func (c *MySQLClient) executeSelect(ctx context.Context, exec executor, op *pb.Operation, index int) *pb.OperationResult {
	database := op.GetDatabase()
	if database == "" {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "database is empty",
		}
	}
	table := op.GetTable()
	if table == "" {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "table is empty",
		}
	}

	selectOp := op.GetSelect()
	if selectOp == nil {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "select operation is nil",
		}
	}

	// 构建 SELECT 字段
	fields := "*"
	if len(selectOp.Fields) > 0 {
		fields = strings.Join(selectOp.Fields, ", ")
	}

	// 构建 WHERE 子句
	whereClause, whereParams := buildWhereClause(selectOp.Where)

	// 构建 ORDER BY
	orderClause := ""
	if selectOp.OrderBy != nil && selectOp.OrderBy.Field != "" {
		direction := "ASC"
		if selectOp.OrderBy.Descending {
			direction = "DESC"
		}
		orderClause = fmt.Sprintf(" ORDER BY %s %s", selectOp.OrderBy.Field, direction)
	}

	// 构建 LIMIT/OFFSET
	limitClause := ""
	if selectOp.Pagination != nil && selectOp.Pagination.Limit > 0 {
		limitClause = fmt.Sprintf(" LIMIT %d", selectOp.Pagination.Limit)
		if selectOp.Pagination.Offset > 0 {
			limitClause += fmt.Sprintf(" OFFSET %d", selectOp.Pagination.Offset)
		}
	}

	query := fmt.Sprintf("SELECT %s FROM %s.%s%s%s%s",
		fields, database, table, whereClause, orderClause, limitClause)

	rows, err := exec.QueryContext(ctx, query, whereParams...)
	if err != nil {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("select error: %v", err),
		}
	}
	defer rows.Close()

	// 获取列信息
	columns, err := rows.Columns()
	if err != nil {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("get columns error: %v", err),
		}
	}

	// 获取列类型信息
	columnTypes, err := rows.ColumnTypes()
	if err != nil {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("get column types error: %v", err),
		}
	}

	// 读取结果
	var resultRows []*pb.ResultRow
	for rows.Next() {
		// 创建扫描目标，使用 interface{} 接收原始类型
		scanDest := make([]interface{}, len(columns))
		for i := range scanDest {
			scanDest[i] = new(interface{})
		}

		if err := rows.Scan(scanDest...); err != nil {
			return &pb.OperationResult{
				Index:   int32(index),
				Success: false,
				Error:   fmt.Sprintf("scan row error: %v", err),
			}
		}

		// 构建结果行，根据列类型转换为对应的 TypedValue
		fields := make(map[string]*pb.TypedValue)
		for i, col := range columns {
			val := *(scanDest[i].(*interface{}))
			// 安全获取列类型，防止越界访问
			var colType *sql.ColumnType
			if columnTypes != nil && i < len(columnTypes) {
				colType = columnTypes[i]
			}
			fields[col] = convertToTypedValue(val, colType)
		}

		resultRows = append(resultRows, &pb.ResultRow{Fields: fields})
	}

	// 检查迭代过程中是否发生错误（如网络中断等）
	if err := rows.Err(); err != nil {
		return &pb.OperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("rows iteration error: %v", err),
		}
	}

	return &pb.OperationResult{
		Index:   int32(index),
		Success: true,
		Result: &pb.OperationResult_SelectResult{
			SelectResult: &pb.SelectResult{
				Rows:  resultRows,
				Total: int32(len(resultRows)),
			},
		},
	}
}

// 辅助函数：将数据库值转换为 TypedValue（保留原始类型）
func convertToTypedValue(val interface{}, colType *sql.ColumnType) *pb.TypedValue {
	if val == nil {
		return &pb.TypedValue{
			Value: &pb.TypedValue_NullValue{NullValue: pb.NullValue_NULL_VALUE},
		}
	}

	// 根据实际值类型进行转换
	switch v := val.(type) {
	case time.Time:
		// 处理时间类型，转换为 Unix 毫秒时间戳
		return &pb.TypedValue{
			Value: &pb.TypedValue_TimestampValue{TimestampValue: v.UnixMilli()},
		}
	case []byte:
		// []byte 需要根据列类型判断是字符串还是二进制
		// 如果 colType 为 nil，默认当作字符串处理
		if colType != nil {
			dbType := strings.ToUpper(colType.DatabaseTypeName())
			if strings.Contains(dbType, "BLOB") || strings.Contains(dbType, "BINARY") {
				return &pb.TypedValue{
					Value: &pb.TypedValue_BytesValue{BytesValue: v},
				}
			}
		}
		return &pb.TypedValue{
			Value: &pb.TypedValue_StringValue{StringValue: string(v)},
		}
	case string:
		return &pb.TypedValue{
			Value: &pb.TypedValue_StringValue{StringValue: v},
		}
	case int64:
		return &pb.TypedValue{
			Value: &pb.TypedValue_IntValue{IntValue: v},
		}
	case int32:
		return &pb.TypedValue{
			Value: &pb.TypedValue_IntValue{IntValue: int64(v)},
		}
	case int:
		return &pb.TypedValue{
			Value: &pb.TypedValue_IntValue{IntValue: int64(v)},
		}
	case float64:
		return &pb.TypedValue{
			Value: &pb.TypedValue_DoubleValue{DoubleValue: v},
		}
	case float32:
		return &pb.TypedValue{
			Value: &pb.TypedValue_DoubleValue{DoubleValue: float64(v)},
		}
	case bool:
		return &pb.TypedValue{
			Value: &pb.TypedValue_BoolValue{BoolValue: v},
		}
	default:
		// 其他类型转为字符串
		return &pb.TypedValue{
			Value: &pb.TypedValue_StringValue{StringValue: fmt.Sprintf("%v", v)},
		}
	}
}

// 辅助函数：从 TypedValue 中提取实际值
func extractTypedValue(tv *pb.TypedValue) interface{} {
	if tv == nil {
		return nil
	}
	switch v := tv.GetValue().(type) {
	case *pb.TypedValue_StringValue:
		return v.StringValue
	case *pb.TypedValue_IntValue:
		return v.IntValue
	case *pb.TypedValue_DoubleValue:
		return v.DoubleValue
	case *pb.TypedValue_BoolValue:
		return v.BoolValue
	case *pb.TypedValue_BytesValue:
		return v.BytesValue
	case *pb.TypedValue_TimestampValue:
		return v.TimestampValue
	case *pb.TypedValue_NullValue:
		return nil
	default:
		return nil
	}
}

// 辅助构建WHERE
func buildWhereClause(where *pb.WhereClause) (string, []interface{}) {
	if where == nil {
		return "", nil
	}

	var params []interface{}

	// 优先使用 raw_clause（复杂条件）
	if where.RawClause != "" {
		for _, p := range where.RawParams {
			params = append(params, extractTypedValue(p))
		}
		return " WHERE " + where.RawClause, params
	}

	// 使用简单等值条件
	if len(where.Conditions) == 0 {
		return "", nil
	}

	clauses := make([]string, 0, len(where.Conditions))
	for col, typedVal := range where.Conditions {
		clauses = append(clauses, fmt.Sprintf("%s = ?", col))
		params = append(params, extractTypedValue(typedVal))
	}

	return " WHERE " + strings.Join(clauses, " AND "), params
}