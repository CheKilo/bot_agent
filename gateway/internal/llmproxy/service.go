package llmproxy

import (
	"context"
	"encoding/json"
	"fmt"

	"bot_agent/gateway/internal/logger"
	"bot_agent/gateway/internal/pb"

	"google.golang.org/grpc"
)

// LLMProxyService 实现 LLMProxyServiceServer 接口
type LLMProxyService struct {
	pb.UnimplementedLLMProxyServiceServer
	client *LLMClient
}

// NewLLMProxyService 创建 LLM 代理服务
func NewLLMProxyService(config LLMProxyConfig) *LLMProxyService {
	return &LLMProxyService{
		client: NewLLMClient(config),
	}
}

// SetClient 设置 LLM 客户端（用于测试或延迟初始化）
func (s *LLMProxyService) SetClient(client *LLMClient) {
	s.client = client
}

// ChatCompletion 非流式对话
func (s *LLMProxyService) ChatCompletion(ctx context.Context, req *pb.ChatCompletionRequest) (*pb.ChatCompletionResponse, error) {
	logger.Info("ChatCompletion request: deployment_id=%s, messages=%d", req.DeploymentId, len(req.Messages))

	if s.client == nil {
		return nil, fmt.Errorf("LLM client not initialized")
	}

	// 转换 gRPC 请求为 HTTP 请求
	httpReq := s.convertToHTTPRequest(req)

	// 调用 LLM API
	httpResp, err := s.client.ChatCompletion(ctx, req.DeploymentId, req.ApiVersion, httpReq)
	if err != nil {
		logger.Error("ChatCompletion failed: %v", err)
		return nil, fmt.Errorf("chat completion failed: %w", err)
	}

	// 转换 HTTP 响应为 gRPC 响应
	resp := s.convertToGRPCResponse(httpResp)

	logger.Info("ChatCompletion success: id=%s, choices=%d", resp.Id, len(resp.Choices))
	return resp, nil
}

// ChatCompletionStream 流式对话
func (s *LLMProxyService) ChatCompletionStream(req *pb.ChatCompletionRequest, stream grpc.ServerStreamingServer[pb.ChatCompletionChunk]) error {
	logger.Info("ChatCompletionStream request: deployment_id=%s, messages=%d", req.DeploymentId, len(req.Messages))

	if s.client == nil {
		return fmt.Errorf("LLM client not initialized")
	}

	// 转换 gRPC 请求为 HTTP 请求
	httpReq := s.convertToHTTPRequest(req)

	ctx := stream.Context()

	// 调用流式 API
	err := s.client.ChatCompletionStream(ctx, req.DeploymentId, req.ApiVersion, httpReq, func(chunk *ChatCompletionChunkHTTP) error {
		// 转换为 gRPC chunk 并发送
		grpcChunk := s.convertToGRPCChunk(chunk)
		if err := stream.Send(grpcChunk); err != nil {
			return fmt.Errorf("send chunk failed: %w", err)
		}
		return nil
	})

	if err != nil {
		logger.Error("ChatCompletionStream failed: %v", err)
		return fmt.Errorf("chat completion stream failed: %w", err)
	}

	logger.Info("ChatCompletionStream completed")
	return nil
}

// GetEmbedding 获取 Embedding 向量
func (s *LLMProxyService) GetEmbedding(ctx context.Context, req *pb.EmbeddingRequest) (*pb.EmbeddingResponse, error) {
	logger.Info("GetEmbedding request: deployment_id=%s, input_count=%d", req.DeploymentId, len(req.Input))

	if s.client == nil {
		return nil, fmt.Errorf("LLM client not initialized")
	}

	// 调用 Embedding API
	httpResp, err := s.client.GetEmbedding(ctx, req.DeploymentId, req.ApiVersion, req.Input)
	if err != nil {
		logger.Error("GetEmbedding failed: %v", err)
		return nil, fmt.Errorf("get embedding failed: %w", err)
	}

	// 转换响应
	resp := &pb.EmbeddingResponse{
		Object: httpResp.Object,
		Model:  httpResp.Model,
		Data:   make([]*pb.EmbeddingData, len(httpResp.Data)),
		Usage: &pb.Usage{
			PromptTokens:     httpResp.Usage.PromptTokens,
			CompletionTokens: httpResp.Usage.CompletionTokens,
			TotalTokens:      httpResp.Usage.TotalTokens,
		},
	}

	for i, d := range httpResp.Data {
		resp.Data[i] = &pb.EmbeddingData{
			Index:     d.Index,
			Object:    d.Object,
			Embedding: d.Embedding,
		}
	}

	logger.Info("GetEmbedding success: data_count=%d", len(resp.Data))
	return resp, nil
}

// Close 关闭服务
func (s *LLMProxyService) Close() error {
	if s.client != nil {
		return s.client.Close()
	}
	return nil
}

// ==================== 转换方法 ====================

// convertToHTTPRequest 将 gRPC 请求转换为 HTTP 请求
// 注意：只有用户明确设置的参数才会被传递给 API，避免某些模型不支持特定参数的问题
func (s *LLMProxyService) convertToHTTPRequest(req *pb.ChatCompletionRequest) *ChatCompletionHTTPRequest {
	httpReq := &ChatCompletionHTTPRequest{
		Messages: s.convertMessages(req.Messages),
		Stop:     req.Stop,
		User:     req.User,
	}

	// 只在用户明确设置非零值时才传递这些参数（某些模型如 GPT-5 不支持自定义值）
	if req.Temperature != 0 {
		httpReq.Temperature = &req.Temperature
	}
	if req.MaxTokens != 0 {
		httpReq.MaxCompletionTokens = &req.MaxTokens
	}
	if req.TopP != 0 {
		httpReq.TopP = &req.TopP
	}
	if req.FrequencyPenalty != 0 {
		httpReq.FrequencyPenalty = &req.FrequencyPenalty
	}
	if req.PresencePenalty != 0 {
		httpReq.PresencePenalty = &req.PresencePenalty
	}
	if req.N != 0 {
		httpReq.N = &req.N
	}
	if req.Seed != 0 {
		httpReq.Seed = &req.Seed
	}

	// 响应格式
	if req.ResponseFormat != "" {
		httpReq.ResponseFormat = &ResponseFormat{Type: req.ResponseFormat}
	}

	// 工具
	if len(req.Tools) > 0 {
		httpReq.Tools = s.convertTools(req.Tools)
	}

	// 工具选择
	if req.ToolChoice != "" {
		httpReq.ToolChoice = req.ToolChoice
	}

	return httpReq
}

// convertMessages 转换消息列表
func (s *LLMProxyService) convertMessages(messages []*pb.ChatMessage) []ChatMessageHTTP {
	result := make([]ChatMessageHTTP, len(messages))
	for i, msg := range messages {
		httpMsg := ChatMessageHTTP{
			Role:       msg.Role,
			Name:       msg.Name,
			ToolCallID: msg.ToolCallId,
		}

		// 处理内容
		switch content := msg.ContentType.(type) {
		case *pb.ChatMessage_Content:
			// 简单文本内容
			httpMsg.Content = content.Content
		case *pb.ChatMessage_ContentParts:
			// 多模态内容
			parts := make([]ContentPartHTTP, len(content.ContentParts.Parts))
			for j, part := range content.ContentParts.Parts {
				parts[j] = ContentPartHTTP{
					Type: part.Type,
					Text: part.Text,
				}
				if part.ImageUrl != nil {
					parts[j].ImageURL = &ImageURLHTTP{
						URL:    part.ImageUrl.Url,
						Detail: part.ImageUrl.Detail,
					}
				}
			}
			httpMsg.Content = parts
		}

		// 处理工具调用
		if len(msg.ToolCalls) > 0 {
			httpMsg.ToolCalls = make([]ToolCallHTTP, len(msg.ToolCalls))
			for j, tc := range msg.ToolCalls {
				httpMsg.ToolCalls[j] = ToolCallHTTP{
					ID:   tc.Id,
					Type: tc.Type,
					Function: FunctionCallHTTP{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}

		result[i] = httpMsg
	}
	return result
}

// convertTools 转换工具列表
func (s *LLMProxyService) convertTools(tools []*pb.Tool) []ToolHTTP {
	result := make([]ToolHTTP, len(tools))
	for i, tool := range tools {
		httpTool := ToolHTTP{
			Type: tool.Type,
			Function: FunctionDefHTTP{
				Name:        tool.Function.Name,
				Description: tool.Function.Description,
			},
		}

		// 解析 parameters JSON
		if tool.Function.Parameters != "" {
			var params interface{}
			if err := json.Unmarshal([]byte(tool.Function.Parameters), &params); err == nil {
				httpTool.Function.Parameters = params
			} else {
				// 如果解析失败，直接使用字符串
				httpTool.Function.Parameters = tool.Function.Parameters
			}
		}

		result[i] = httpTool
	}
	return result
}

// convertToGRPCResponse 将 HTTP 响应转换为 gRPC 响应
func (s *LLMProxyService) convertToGRPCResponse(httpResp *ChatCompletionHTTPResponse) *pb.ChatCompletionResponse {
	resp := &pb.ChatCompletionResponse{
		Id:      httpResp.ID,
		Object:  httpResp.Object,
		Created: httpResp.Created,
		Model:   httpResp.Model,
		Choices: make([]*pb.Choice, len(httpResp.Choices)),
		Usage: &pb.Usage{
			PromptTokens:     httpResp.Usage.PromptTokens,
			CompletionTokens: httpResp.Usage.CompletionTokens,
			TotalTokens:      httpResp.Usage.TotalTokens,
		},
	}

	for i, choice := range httpResp.Choices {
		resp.Choices[i] = &pb.Choice{
			Index:        choice.Index,
			Message:      s.convertToGRPCMessage(&choice.Message),
			FinishReason: choice.FinishReason,
		}
	}

	return resp
}

// convertToGRPCMessage 将 HTTP 消息转换为 gRPC 消息
func (s *LLMProxyService) convertToGRPCMessage(httpMsg *ChatMessageHTTP) *pb.ChatMessage {
	msg := &pb.ChatMessage{
		Role:       httpMsg.Role,
		Name:       httpMsg.Name,
		ToolCallId: httpMsg.ToolCallID,
	}

	// 处理内容
	switch content := httpMsg.Content.(type) {
	case string:
		msg.ContentType = &pb.ChatMessage_Content{Content: content}
	case []interface{}:
		// 多模态内容（通常响应不会返回多模态，但保留处理逻辑）
		parts := &pb.ContentList{Parts: make([]*pb.ContentPart, len(content))}
		for i, part := range content {
			if partMap, ok := part.(map[string]interface{}); ok {
				contentPart := &pb.ContentPart{}
				if t, ok := partMap["type"].(string); ok {
					contentPart.Type = t
				}
				if text, ok := partMap["text"].(string); ok {
					contentPart.Text = text
				}
				parts.Parts[i] = contentPart
			}
		}
		msg.ContentType = &pb.ChatMessage_ContentParts{ContentParts: parts}
	}

	// 处理工具调用
	if len(httpMsg.ToolCalls) > 0 {
		msg.ToolCalls = make([]*pb.ToolCall, len(httpMsg.ToolCalls))
		for i, tc := range httpMsg.ToolCalls {
			msg.ToolCalls[i] = &pb.ToolCall{
				Id:   tc.ID,
				Type: tc.Type,
				Function: &pb.FunctionCall{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			}
		}
	}

	return msg
}

// convertToGRPCChunk 将 HTTP 流式响应块转换为 gRPC 块
func (s *LLMProxyService) convertToGRPCChunk(httpChunk *ChatCompletionChunkHTTP) *pb.ChatCompletionChunk {
	chunk := &pb.ChatCompletionChunk{
		Id:      httpChunk.ID,
		Object:  httpChunk.Object,
		Created: httpChunk.Created,
		Model:   httpChunk.Model,
		Choices: make([]*pb.StreamChoice, len(httpChunk.Choices)),
	}

	// Usage（仅最后一个 chunk 有值）
	if httpChunk.Usage != nil {
		chunk.Usage = &pb.Usage{
			PromptTokens:     httpChunk.Usage.PromptTokens,
			CompletionTokens: httpChunk.Usage.CompletionTokens,
			TotalTokens:      httpChunk.Usage.TotalTokens,
		}
	}

	for i, choice := range httpChunk.Choices {
		chunk.Choices[i] = &pb.StreamChoice{
			Index: choice.Index,
			Delta: &pb.ChatMessageDelta{
				Role:    choice.Delta.Role,
				Content: choice.Delta.Content,
			},
			FinishReason: choice.FinishReason,
		}
	}

	return chunk
}
