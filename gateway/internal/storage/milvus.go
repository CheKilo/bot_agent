// milvus.go
package storage

import (
	pb "bot_agent/gateway/internal/pb"
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

type MilvusClient struct {
	client     client.Client // 注意：不是指针，SDK 返回的是接口
	dbPath     string
	dimension  int
	collection string // 默认 collection 名称
}

// NewMilvusClient 使用 context.Background() 初始化连接
// collection 参数指定默认使用的 collection 名称
func NewMilvusClient(dbPath string, dimension int, collection string) (*MilvusClient, error) {
	ctx := context.Background()

	c, err := client.NewClient(ctx, client.Config{
		Address: dbPath,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to connect milvus: %w", err)
	}

	// 如果未指定 collection，使用默认值
	if collection == "" {
		collection = "memory_vectors"
	}

	m := &MilvusClient{
		client:     c,
		dbPath:     dbPath,
		dimension:  dimension,
		collection: collection,
	}

	// 初始化配置的 Collection
	if err := m.ensureCollection(ctx, collection); err != nil {
		c.Close()
		return nil, fmt.Errorf("failed to ensure collection: %w", err)
	}

	return m, nil
}

// Close 关闭 Milvus 连接
func (m *MilvusClient) Close() error {
	return m.client.Close()
}

// ExecuteBatch 批量执行向量操作（按顺序执行多个 Insert/Search/Delete/Upsert 操作）
func (m *MilvusClient) ExecuteBatch(ctx context.Context, ops []*pb.VectorOperation) ([]*pb.VectorOperationResult, error) {
	results := make([]*pb.VectorOperationResult, len(ops))

	for i, op := range ops {
		var res *pb.VectorOperationResult

		// 确保 partition 存在（如果指定了 partition）
		if op.Partition != "" {
			if err := m.ensurePartition(ctx, op.Collection, op.Partition); err != nil {
				results[i] = &pb.VectorOperationResult{
					Index:   int32(i),
					Success: false,
					Error:   fmt.Sprintf("ensure partition error: %v", err),
				}
				continue
			}
		}

		// 使用 oneof 类型断言判断操作类型
		switch op.GetOperation().(type) {
		case *pb.VectorOperation_Insert:
			res = m.executeInsert(ctx, op, i)
		case *pb.VectorOperation_Search:
			res = m.executeSearch(ctx, op, i)
		case *pb.VectorOperation_Delete:
			res = m.executeDelete(ctx, op, i)
		case *pb.VectorOperation_Upsert:
			res = m.executeUpsert(ctx, op, i)
		default:
			res = &pb.VectorOperationResult{
				Index:   int32(i),
				Success: false,
				Error:   "unknown operation type",
			}
		}

		results[i] = res
	}

	return results, nil
}

// executeInsert 执行向量插入操作
func (m *MilvusClient) executeInsert(ctx context.Context, op *pb.VectorOperation, index int) *pb.VectorOperationResult {
	collection := op.GetCollection()
	if collection == "" {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "collection is empty",
		}
	}

	insertOp := op.GetInsert()
	if insertOp == nil || len(insertOp.Vectors) == 0 {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "insert vectors is empty",
		}
	}

	// 构建列数据
	columns, err := m.buildColumns(insertOp.Vectors)
	if err != nil {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("build columns error: %v", err),
		}
	}

	// 加载 Collection 到内存（如果未加载）
	if err := m.ensureCollectionLoaded(ctx, collection); err != nil {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("load collection error: %v", err),
		}
	}

	// 执行插入
	var insertErr error
	if op.Partition != "" {
		_, insertErr = m.client.Insert(ctx, collection, op.Partition, columns...)
	} else {
		_, insertErr = m.client.Insert(ctx, collection, "", columns...)
	}

	if insertErr != nil {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("insert error: %v", insertErr),
		}
	}

	// Flush 确保数据写入磁盘，使其可被搜索
	if err := m.client.Flush(ctx, collection, false); err != nil {
		// Flush 失败不影响插入结果，只记录警告
		// 数据最终会被自动 flush
	}

	return &pb.VectorOperationResult{
		Index:   int32(index),
		Success: true,
		Result: &pb.VectorOperationResult_InsertResult{
			InsertResult: &pb.VectorInsertResult{
				InsertedCount: int32(len(insertOp.Vectors)),
			},
		},
	}
}

// executeUpsert 执行向量更新或插入操作
func (m *MilvusClient) executeUpsert(ctx context.Context, op *pb.VectorOperation, index int) *pb.VectorOperationResult {
	collection := op.GetCollection()
	if collection == "" {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "collection is empty",
		}
	}

	upsertOp := op.GetUpsert()
	if upsertOp == nil || len(upsertOp.Vectors) == 0 {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "upsert vectors is empty",
		}
	}

	// 构建列数据
	columns, err := m.buildColumns(upsertOp.Vectors)
	if err != nil {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("build columns error: %v", err),
		}
	}

	// 加载 Collection 到内存（如果未加载）
	if err := m.ensureCollectionLoaded(ctx, collection); err != nil {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("load collection error: %v", err),
		}
	}

	// 执行 Upsert
	var upsertErr error
	if op.Partition != "" {
		_, upsertErr = m.client.Upsert(ctx, collection, op.Partition, columns...)
	} else {
		_, upsertErr = m.client.Upsert(ctx, collection, "", columns...)
	}

	if upsertErr != nil {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("upsert error: %v", upsertErr),
		}
	}

	// Flush 确保数据写入磁盘，使其可被搜索
	if err := m.client.Flush(ctx, collection, false); err != nil {
		// Flush 失败不影响 upsert 结果，只记录警告
		// 数据最终会被自动 flush
	}

	return &pb.VectorOperationResult{
		Index:   int32(index),
		Success: true,
		Result: &pb.VectorOperationResult_UpsertResult{
			UpsertResult: &pb.VectorUpsertResult{
				UpsertedCount: int32(len(upsertOp.Vectors)),
			},
		},
	}
}

// executeSearch 执行向量搜索操作
func (m *MilvusClient) executeSearch(ctx context.Context, op *pb.VectorOperation, index int) *pb.VectorOperationResult {
	collection := op.GetCollection()
	if collection == "" {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "collection is empty",
		}
	}

	searchOp := op.GetSearch()
	if searchOp == nil || len(searchOp.QueryVector) == 0 {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "search query vector is empty",
		}
	}

	// 加载 Collection 到内存（如果未加载）
	if err := m.ensureCollectionLoaded(ctx, collection); err != nil {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("load collection error: %v", err),
		}
	}

	// 构建查询向量
	queryVectors := []entity.Vector{entity.FloatVector(searchOp.QueryVector)}

	// 构建搜索参数
	sp, err := entity.NewIndexIvfFlatSearchParam(16) // nprobe 参数
	if err != nil {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("create search param error: %v", err),
		}
	}

	// 构建过滤表达式
	filterExpr := m.buildFilterExpr(searchOp.Filter, searchOp.FilterExpr)

	// 设置 top_k
	topK := int(searchOp.TopK)
	if topK <= 0 {
		topK = 10 // 默认值
	}

	// 设置输出字段
	outputFields := searchOp.OutputFields
	if len(outputFields) == 0 {
		outputFields = []string{"*"} // 返回所有字段
	}

	// 构建分区列表
	partitions := []string{}
	if op.Partition != "" {
		partitions = []string{op.Partition}
	}

	// 执行搜索
	searchResult, err := m.client.Search(
		ctx,
		collection,
		partitions,
		filterExpr,
		outputFields,
		queryVectors,
		"vector", // 向量字段名
		entity.L2,
		topK,
		sp,
	)
	if err != nil {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("search error: %v", err),
		}
	}

	// 解析搜索结果
	matches := make([]*pb.VectorMatch, 0)
	for _, result := range searchResult {
		for i := 0; i < result.ResultCount; i++ {
			// 获取 ID
			var id string
			if result.IDs != nil {
				switch ids := result.IDs.(type) {
				case *entity.ColumnInt64:
					if i < ids.Len() {
						val, _ := ids.ValueByIdx(i)
						id = strconv.FormatInt(val, 10)
					}
				case *entity.ColumnVarChar:
					if i < ids.Len() {
						val, _ := ids.ValueByIdx(i)
						id = val
					}
				}
			}

			// 获取分数
			score := float32(0)
			if i < len(result.Scores) {
				score = result.Scores[i]
			}

			// 过滤最小分数阈值
			if searchOp.MinScore > 0 && score < searchOp.MinScore {
				continue
			}

			// 获取元数据
			metadata := make(map[string]*pb.TypedValue)
			for _, field := range result.Fields {
				fieldName := field.Name()
				if fieldName == "vector" || fieldName == "id" {
					continue // 跳过向量和 ID 字段
				}
				if i < field.Len() {
					metadata[fieldName] = m.fieldToTypedValue(field, i)
				}
			}

			matches = append(matches, &pb.VectorMatch{
				Id:       id,
				Score:    score,
				Metadata: metadata,
			})
		}
	}

	return &pb.VectorOperationResult{
		Index:   int32(index),
		Success: true,
		Result: &pb.VectorOperationResult_SearchResult{
			SearchResult: &pb.VectorSearchResult{
				Matches: matches,
			},
		},
	}
}

// executeDelete 执行向量删除操作
func (m *MilvusClient) executeDelete(ctx context.Context, op *pb.VectorOperation, index int) *pb.VectorOperationResult {
	collection := op.GetCollection()
	if collection == "" {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "collection is empty",
		}
	}

	deleteOp := op.GetDelete()
	if deleteOp == nil {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   "delete operation is nil",
		}
	}

	// 加载 Collection 到内存（如果未加载）
	if err := m.ensureCollectionLoaded(ctx, collection); err != nil {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("load collection error: %v", err),
		}
	}

	var deleteErr error
	var deletedCount int

	// 优先使用 IDs 删除
	if len(deleteOp.Ids) > 0 {
		// 将字符串 ID 转换为表达式
		idExpr := fmt.Sprintf("id in [%s]", strings.Join(quoteStrings(deleteOp.Ids), ","))
		deleteErr = m.client.Delete(ctx, collection, op.Partition, idExpr)
		deletedCount = len(deleteOp.Ids)
	} else {
		// 使用过滤表达式删除
		filterExpr := m.buildFilterExpr(deleteOp.Filter, deleteOp.FilterExpr)
		if filterExpr == "" {
			return &pb.VectorOperationResult{
				Index:   int32(index),
				Success: false,
				Error:   "delete requires ids or filter expression (to prevent accidental full collection deletion)",
			}
		}
		deleteErr = m.client.Delete(ctx, collection, op.Partition, filterExpr)
		deletedCount = 1 // 无法确定实际删除数量
	}

	if deleteErr != nil {
		return &pb.VectorOperationResult{
			Index:   int32(index),
			Success: false,
			Error:   fmt.Sprintf("delete error: %v", deleteErr),
		}
	}

	// Flush 确保删除操作立即生效
	if err := m.client.Flush(ctx, collection, false); err != nil {
		// Flush 失败不影响删除结果，只记录警告
		// 数据最终会被自动 flush
	}

	return &pb.VectorOperationResult{
		Index:   int32(index),
		Success: true,
		Result: &pb.VectorOperationResult_DeleteResult{
			DeleteResult: &pb.VectorDeleteResult{
				DeletedCount: int32(deletedCount),
			},
		},
	}
}

// ensureCollection 确保 Collection 存在
// 注意：此方法只检查 Collection 是否存在，不会自动创建
// 用户需要提前通过 Milvus 管理工具或 API 创建 Collection 并定义 schema
// 这样可以保持 Milvus 客户端的通用性，不 hardcode 任何业务字段
func (m *MilvusClient) ensureCollection(ctx context.Context, collectionName string) error {
	exists, err := m.client.HasCollection(ctx, collectionName)
	if err != nil {
		return fmt.Errorf("failed to check collection: %w", err)
	}

	if !exists {
		return fmt.Errorf("collection '%s' does not exist, please create it first with your desired schema", collectionName)
	}

	return nil
}

// ensurePartition 确保分区存在
func (m *MilvusClient) ensurePartition(ctx context.Context, collection, partition string) error {
	exists, err := m.client.HasPartition(ctx, collection, partition)
	if err != nil {
		return fmt.Errorf("failed to check partition: %w", err)
	}

	if !exists {
		err = m.client.CreatePartition(ctx, collection, partition)
		if err != nil {
			return fmt.Errorf("failed to create partition: %w", err)
		}
	}

	return nil
}

// ensureCollectionLoaded 确保 Collection 已加载到内存
func (m *MilvusClient) ensureCollectionLoaded(ctx context.Context, collection string) error {
	// 检查是否已加载
	loaded, err := m.client.GetLoadState(ctx, collection, nil)
	if err != nil {
		return fmt.Errorf("failed to get load state: %w", err)
	}

	if loaded != entity.LoadStateLoaded {
		err = m.client.LoadCollection(ctx, collection, false)
		if err != nil {
			return fmt.Errorf("failed to load collection: %w", err)
		}
	}

	return nil
}

// buildColumns 根据 VectorData 动态构建 Milvus 列数据
// 不做任何 hardcode，完全根据用户传递的元数据字段动态构建
// 用户需要确保传递的字段与 Collection schema 中定义的字段匹配
// 固定字段：id (VarChar, PK), vector (FloatVector)
// 动态字段：根据用户传递的 metadata 自动推断类型
func (m *MilvusClient) buildColumns(vectors []*pb.VectorData) ([]entity.Column, error) {
	if len(vectors) == 0 {
		return nil, fmt.Errorf("vectors is empty")
	}

	n := len(vectors)

	// 构建 id 和 vector 列（固定字段）
	ids := make([]string, n)
	vectorData := make([][]float32, n)
	for i, v := range vectors {
		ids[i] = v.Id
		vectorData[i] = v.Vector
	}

	columns := []entity.Column{
		entity.NewColumnVarChar("id", ids),
		entity.NewColumnFloatVector("vector", m.dimension, vectorData),
	}

	// 收集所有元数据字段名及其类型（从第一个非空值推断）
	fieldTypes := make(map[string]string) // fieldName -> type
	for _, v := range vectors {
		for fieldName, tv := range v.Metadata {
			if tv != nil {
				if _, exists := fieldTypes[fieldName]; !exists {
					fieldTypes[fieldName] = inferTypedValueType(tv)
				}
			}
		}
	}

	// 为每个元数据字段构建对应的列
	for fieldName, fieldType := range fieldTypes {
		col := buildTypedColumn(fieldName, fieldType, vectors)
		if col != nil {
			columns = append(columns, col)
		}
	}

	return columns, nil
}

// inferTypedValueType 推断 TypedValue 的实际类型
// 对于字符串类型，会智能检测是否为有效 JSON 格式
func inferTypedValueType(tv *pb.TypedValue) string {
	if tv == nil {
		return "string"
	}
	switch v := tv.GetValue().(type) {
	case *pb.TypedValue_IntValue:
		return "int64"
	case *pb.TypedValue_TimestampValue:
		return "int64" // 时间戳存储为 int64
	case *pb.TypedValue_DoubleValue:
		return "double"
	case *pb.TypedValue_BoolValue:
		return "bool"
	case *pb.TypedValue_BytesValue:
		return "bytes"
	case *pb.TypedValue_StringValue:
		// 智能检测字符串是否为有效 JSON 格式（对象或数组）
		if isValidJSON(v.StringValue) {
			return "json"
		}
		return "string"
	default:
		return "string"
	}
}

// isValidJSON 检测字符串是否为有效的 JSON 对象或数组
// 只有 {} 或 [] 开头的才被认为是 JSON，普通字符串不会被误判
func isValidJSON(s string) bool {
	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return false
	}
	// 只检测 JSON 对象或数组，不检测纯字符串/数字等
	if (s[0] == '{' && s[len(s)-1] == '}') || (s[0] == '[' && s[len(s)-1] == ']') {
		return json.Valid([]byte(s))
	}
	return false
}

// buildTypedColumn 根据字段类型构建对应的 Milvus 列
func buildTypedColumn(fieldName, fieldType string, vectors []*pb.VectorData) entity.Column {
	n := len(vectors)

	switch fieldType {
	case "int64":
		values := make([]int64, n)
		for i, v := range vectors {
			if tv, ok := v.Metadata[fieldName]; ok && tv != nil {
				values[i] = extractTypedValueAsInt64(tv)
			}
		}
		return entity.NewColumnInt64(fieldName, values)

	case "double":
		values := make([]float64, n)
		for i, v := range vectors {
			if tv, ok := v.Metadata[fieldName]; ok && tv != nil {
				values[i] = extractTypedValueAsDouble(tv)
			}
		}
		return entity.NewColumnDouble(fieldName, values)

	case "bool":
		values := make([]bool, n)
		for i, v := range vectors {
			if tv, ok := v.Metadata[fieldName]; ok && tv != nil {
				values[i] = extractTypedValueAsBool(tv)
			}
		}
		return entity.NewColumnBool(fieldName, values)

	case "json":
		// JSON 类型：将字符串值转换为 []byte
		values := make([][]byte, n)
		for i, v := range vectors {
			if tv, ok := v.Metadata[fieldName]; ok && tv != nil {
				jsonStr := extractTypedValueAsString(tv)
				if jsonStr == "" {
					jsonStr = "{}" // 默认空 JSON 对象
				}
				values[i] = []byte(jsonStr)
			} else {
				values[i] = []byte("{}") // 默认空 JSON 对象
			}
		}
		return entity.NewColumnJSONBytes(fieldName, values)

	default: // string
		values := make([]string, n)
		for i, v := range vectors {
			if tv, ok := v.Metadata[fieldName]; ok && tv != nil {
				values[i] = extractTypedValueAsString(tv)
			}
		}
		return entity.NewColumnVarChar(fieldName, values)
	}
}

// extractTypedValueAsDouble 从 TypedValue 中提取 float64 值
func extractTypedValueAsDouble(tv *pb.TypedValue) float64 {
	if tv == nil {
		return 0.0
	}
	switch v := tv.GetValue().(type) {
	case *pb.TypedValue_DoubleValue:
		return v.DoubleValue
	case *pb.TypedValue_IntValue:
		return float64(v.IntValue)
	case *pb.TypedValue_TimestampValue:
		return float64(v.TimestampValue)
	case *pb.TypedValue_StringValue:
		val, _ := strconv.ParseFloat(v.StringValue, 64)
		return val
	default:
		return 0.0
	}
}

// extractTypedValueAsBool 从 TypedValue 中提取 bool 值
func extractTypedValueAsBool(tv *pb.TypedValue) bool {
	if tv == nil {
		return false
	}
	switch v := tv.GetValue().(type) {
	case *pb.TypedValue_BoolValue:
		return v.BoolValue
	case *pb.TypedValue_IntValue:
		return v.IntValue != 0
	case *pb.TypedValue_StringValue:
		return v.StringValue == "true" || v.StringValue == "1"
	default:
		return false
	}
}

// extractTypedValueAsInt64 从 TypedValue 中提取 int64 值
func extractTypedValueAsInt64(tv *pb.TypedValue) int64 {
	if tv == nil {
		return 0
	}
	switch v := tv.GetValue().(type) {
	case *pb.TypedValue_IntValue:
		return v.IntValue
	case *pb.TypedValue_TimestampValue:
		return v.TimestampValue
	case *pb.TypedValue_DoubleValue:
		return int64(v.DoubleValue)
	case *pb.TypedValue_StringValue:
		val, _ := strconv.ParseInt(v.StringValue, 10, 64)
		return val
	default:
		return 0
	}
}

// buildFilterExpr 构建过滤表达式
func (m *MilvusClient) buildFilterExpr(filter map[string]*pb.TypedValue, filterExpr string) string {
	// 优先使用复杂表达式
	if filterExpr != "" {
		return filterExpr
	}

	// 使用简单等值条件构建表达式
	if len(filter) == 0 {
		return ""
	}

	conditions := make([]string, 0, len(filter))
	for key, tv := range filter {
		val := extractTypedValueAsString(tv)
		// 对字符串值加引号
		switch tv.GetValue().(type) {
		case *pb.TypedValue_StringValue:
			conditions = append(conditions, fmt.Sprintf("%s == \"%s\"", key, val))
		case *pb.TypedValue_IntValue:
			conditions = append(conditions, fmt.Sprintf("%s == %s", key, val))
		case *pb.TypedValue_DoubleValue:
			conditions = append(conditions, fmt.Sprintf("%s == %s", key, val))
		case *pb.TypedValue_BoolValue:
			conditions = append(conditions, fmt.Sprintf("%s == %s", key, val))
		default:
			conditions = append(conditions, fmt.Sprintf("%s == \"%s\"", key, val))
		}
	}

	return strings.Join(conditions, " && ")
}

// fieldToTypedValue 将 Milvus 字段值转换为 TypedValue
func (m *MilvusClient) fieldToTypedValue(field entity.Column, idx int) *pb.TypedValue {
	switch col := field.(type) {
	case *entity.ColumnVarChar:
		val, _ := col.ValueByIdx(idx)
		return &pb.TypedValue{Value: &pb.TypedValue_StringValue{StringValue: val}}
	case *entity.ColumnInt64:
		val, _ := col.ValueByIdx(idx)
		return &pb.TypedValue{Value: &pb.TypedValue_IntValue{IntValue: val}}
	case *entity.ColumnInt32:
		val, _ := col.ValueByIdx(idx)
		return &pb.TypedValue{Value: &pb.TypedValue_IntValue{IntValue: int64(val)}}
	case *entity.ColumnFloat:
		val, _ := col.ValueByIdx(idx)
		return &pb.TypedValue{Value: &pb.TypedValue_DoubleValue{DoubleValue: float64(val)}}
	case *entity.ColumnDouble:
		val, _ := col.ValueByIdx(idx)
		return &pb.TypedValue{Value: &pb.TypedValue_DoubleValue{DoubleValue: val}}
	case *entity.ColumnBool:
		val, _ := col.ValueByIdx(idx)
		return &pb.TypedValue{Value: &pb.TypedValue_BoolValue{BoolValue: val}}
	case *entity.ColumnJSONBytes:
		// JSON 类型：将 []byte 转换为字符串返回
		val, _ := col.ValueByIdx(idx)
		return &pb.TypedValue{Value: &pb.TypedValue_StringValue{StringValue: string(val)}}
	default:
		// 默认转为字符串
		return &pb.TypedValue{Value: &pb.TypedValue_StringValue{StringValue: fmt.Sprintf("%v", field)}}
	}
}

// extractTypedValueAsString 从 TypedValue 中提取值并转换为字符串
func extractTypedValueAsString(tv *pb.TypedValue) string {
	if tv == nil {
		return ""
	}
	switch v := tv.GetValue().(type) {
	case *pb.TypedValue_StringValue:
		return v.StringValue
	case *pb.TypedValue_IntValue:
		return strconv.FormatInt(v.IntValue, 10)
	case *pb.TypedValue_DoubleValue:
		return strconv.FormatFloat(v.DoubleValue, 'f', -1, 64)
	case *pb.TypedValue_BoolValue:
		return strconv.FormatBool(v.BoolValue)
	case *pb.TypedValue_TimestampValue:
		return strconv.FormatInt(v.TimestampValue, 10)
	case *pb.TypedValue_NullValue:
		return ""
	default:
		return ""
	}
}

// quoteStrings 为字符串数组添加引号
func quoteStrings(strs []string) []string {
	result := make([]string, len(strs))
	for i, s := range strs {
		result[i] = fmt.Sprintf("\"%s\"", s)
	}
	return result
}
