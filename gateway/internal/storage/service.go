package storage

import (
	"bot_agent/gateway/internal/logger"
	"bot_agent/gateway/internal/pb"
	"context"
	"errors"
	"fmt"
	"strings"
)

type StorageService struct {
	pb.UnimplementedStorageServiceServer // 这是 gRPC 的向前兼容机制。如果后续在 proto 中新增方法但没实现，服务仍然可以正常运行。
	mysqlClient                          *MySQLClient
	milvusClient                         *MilvusClient
}

// NewStorageService 创建空的存储服务，后续通过 SetXXX 方法设置客户端
func NewStorageService() *StorageService {
	return &StorageService{}
}

// SetMySQLClient 设置 MySQL 客户端
func (s *StorageService) SetMySQLClient(client *MySQLClient) {
	s.mysqlClient = client
}

// SetMilvusClient 设置 Milvus 客户端
func (s *StorageService) SetMilvusClient(client *MilvusClient) {
	s.milvusClient = client
}

func (s *StorageService) Execute(ctx context.Context, req *pb.ExecuteRequest) (*pb.ExecuteResponse, error) {
	logger.Info("Execute req: %v", req)

	// 检查 MySQL 客户端是否已初始化
	if s.mysqlClient == nil {
		return &pb.ExecuteResponse{
			Success: false,
			Error:   "MySQL client not initialized",
		}, nil
	}

	results, err := s.mysqlClient.ExecuteBatch(ctx, req.Operations, req.UseTransaction)
	if err != nil {
		return &pb.ExecuteResponse{
			Success: false,
			Error:   err.Error(),
		}, nil
	}

	// 非事务模式下，需要检查每个操作的结果，汇总失败信息
	var failedIndices []int32
	var failedErrors []string
	for _, res := range results {
		if !res.Success {
			failedIndices = append(failedIndices, res.Index)
			failedErrors = append(failedErrors, res.Error)
		}
	}

	// 如果有失败的操作，汇总错误信息
	if len(failedIndices) > 0 {
		var errorMsg string
		if len(failedIndices) == 1 {
			errorMsg = fmt.Sprintf("operation %d failed: %s", failedIndices[0], failedErrors[0])
		} else {
			// 多个失败时，汇总所有失败信息
			var details []string
			for i, idx := range failedIndices {
				details = append(details, fmt.Sprintf("[%d]: %s", idx, failedErrors[i]))
			}
			errorMsg = fmt.Sprintf("%d operations failed: %s", len(failedIndices), strings.Join(details, "; "))
		}
		return &pb.ExecuteResponse{
			Results: results,
			Success: false,
			Error:   errorMsg,
		}, nil
	}

	return &pb.ExecuteResponse{
		Results: results,
		Success: true,
	}, nil
}

func (s *StorageService) ExecuteVector(ctx context.Context, req *pb.ExecuteVectorRequest) (*pb.ExecuteVectorResponse, error) {
	logger.Info("ExecuteVector req: %v", req)

	// 检查 Milvus 客户端是否已初始化
	if s.milvusClient == nil {
		return &pb.ExecuteVectorResponse{
			Success: false,
			Error:   "Milvus client not initialized",
		}, nil
	}

	// 执行向量操作
	results, err := s.milvusClient.ExecuteBatch(ctx, req.Operations)
	if err != nil {
		return &pb.ExecuteVectorResponse{
			Success: false,
			Error:   err.Error(),
		}, nil
	}

	// 检查每个操作的结果，汇总失败信息
	var failedIndices []int32
	var failedErrors []string
	for _, res := range results {
		if !res.Success {
			failedIndices = append(failedIndices, res.Index)
			failedErrors = append(failedErrors, res.Error)
		}
	}

	// 如果有失败的操作，汇总错误信息
	if len(failedIndices) > 0 {
		var errorMsg string
		if len(failedIndices) == 1 {
			errorMsg = fmt.Sprintf("operation %d failed: %s", failedIndices[0], failedErrors[0])
		} else {
			// 多个失败时，汇总所有失败信息
			var details []string
			for i, idx := range failedIndices {
				details = append(details, fmt.Sprintf("[%d]: %s", idx, failedErrors[i]))
			}
			errorMsg = fmt.Sprintf("%d operations failed: %s", len(failedIndices), strings.Join(details, "; "))
		}
		return &pb.ExecuteVectorResponse{
			Results: results,
			Success: false,
			Error:   errorMsg,
		}, nil
	}

	return &pb.ExecuteVectorResponse{
		Results: results,
		Success: true,
	}, nil
}

func (s *StorageService) Close() error {
	var errs []string

	if s.mysqlClient != nil {
		if err := s.mysqlClient.Close(); err != nil {
			errs = append(errs, fmt.Sprintf("mysql close error: %v", err))
		}
	}

	if s.milvusClient != nil {
		if err := s.milvusClient.Close(); err != nil {
			errs = append(errs, fmt.Sprintf("milvus close error: %v", err))
		}
	}

	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}
