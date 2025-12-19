package llmproxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"bot_agent/gateway/internal/logger"
)

// LLMProxyConfig 配置信息
type LLMProxyConfig struct {
	Endpoint   string        // API 端点，如 http://api.gameai-llm.woa.com/llm-service/azure/public
	APIKey     string        // API Key
	Timeout    time.Duration // 请求超时时间
	MaxRetries int           // 最大重试次数
}

// LLMClient LLM API 客户端
type LLMClient struct {
	config     LLMProxyConfig
	httpClient *http.Client
}

// NewLLMClient 创建新的 LLM 客户端
func NewLLMClient(config LLMProxyConfig) *LLMClient {
	if config.Timeout == 0 {
		config.Timeout = 60 * time.Second
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}

	return &LLMClient{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}
}

// ==================== HTTP 请求/响应结构体 ====================

// ChatCompletionHTTPRequest Azure OpenAI Chat Completion 请求格式
// 注意：可选参数使用指针类型，这样零值时不会被序列化到 JSON 中，避免某些模型不支持特定参数的问题
type ChatCompletionHTTPRequest struct {
	Messages            []ChatMessageHTTP `json:"messages"`
	Temperature         *float32          `json:"temperature,omitempty"`           // 使用指针，不传则使用模型默认值
	MaxCompletionTokens *int32            `json:"max_completion_tokens,omitempty"` // GPT-5 使用 max_completion_tokens
	TopP                *float32          `json:"top_p,omitempty"`
	FrequencyPenalty    *float32          `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float32          `json:"presence_penalty,omitempty"`
	Stop                []string          `json:"stop,omitempty"`
	User                string            `json:"user,omitempty"`
	N                   *int32            `json:"n,omitempty"`
	Seed                *int32            `json:"seed,omitempty"`
	ResponseFormat      *ResponseFormat   `json:"response_format,omitempty"`
	Tools               []ToolHTTP        `json:"tools,omitempty"`
	ToolChoice          interface{}       `json:"tool_choice,omitempty"` // string 或 object
	Stream              bool              `json:"stream,omitempty"`
}

// ResponseFormat 响应格式
type ResponseFormat struct {
	Type string `json:"type"` // "text" 或 "json_object"
}

// ChatMessageHTTP HTTP 请求中的消息格式
type ChatMessageHTTP struct {
	Role       string        `json:"role"`
	Content    interface{}   `json:"content"` // string 或 []ContentPartHTTP
	Name       string        `json:"name,omitempty"`
	ToolCalls  []ToolCallHTTP `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

// ContentPartHTTP 多模态内容部分
type ContentPartHTTP struct {
	Type     string       `json:"type"`
	Text     string       `json:"text,omitempty"`
	ImageURL *ImageURLHTTP `json:"image_url,omitempty"`
}

// ImageURLHTTP 图片 URL
type ImageURLHTTP struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// ToolHTTP 工具定义
type ToolHTTP struct {
	Type     string             `json:"type"`
	Function FunctionDefHTTP `json:"function"`
}

// FunctionDefHTTP 函数定义
type FunctionDefHTTP struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters,omitempty"` // JSON Schema
}

// ToolCallHTTP 工具调用
type ToolCallHTTP struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function FunctionCallHTTP `json:"function"`
}

// FunctionCallHTTP 函数调用
type FunctionCallHTTP struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ChatCompletionHTTPResponse Chat Completion 响应
type ChatCompletionHTTPResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChoiceHTTP `json:"choices"`
	Usage   UsageHTTP    `json:"usage"`
	Error   *ErrorHTTP   `json:"error,omitempty"`
}

// ChoiceHTTP 响应选项
type ChoiceHTTP struct {
	Index        int32           `json:"index"`
	Message      ChatMessageHTTP `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

// UsageHTTP Token 使用统计
type UsageHTTP struct {
	PromptTokens     int32 `json:"prompt_tokens"`
	CompletionTokens int32 `json:"completion_tokens"`
	TotalTokens      int32 `json:"total_tokens"`
}

// ErrorHTTP 错误响应
type ErrorHTTP struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// ChatCompletionChunkHTTP 流式响应块
type ChatCompletionChunkHTTP struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []StreamChoiceHTTP `json:"choices"`
	Usage   *UsageHTTP         `json:"usage,omitempty"`
}

// StreamChoiceHTTP 流式选项
type StreamChoiceHTTP struct {
	Index        int32              `json:"index"`
	Delta        ChatMessageDeltaHTTP `json:"delta"`
	FinishReason string             `json:"finish_reason,omitempty"`
}

// ChatMessageDeltaHTTP 增量消息
type ChatMessageDeltaHTTP struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// EmbeddingHTTPRequest Embedding 请求
type EmbeddingHTTPRequest struct {
	Input []string `json:"input"`
}

// EmbeddingHTTPResponse Embedding 响应
type EmbeddingHTTPResponse struct {
	Object string              `json:"object"`
	Model  string              `json:"model"`
	Data   []EmbeddingDataHTTP `json:"data"`
	Usage  UsageHTTP           `json:"usage"`
	Error  *ErrorHTTP          `json:"error,omitempty"`
}

// EmbeddingDataHTTP Embedding 数据
type EmbeddingDataHTTP struct {
	Index     int32     `json:"index"`
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
}

// ==================== API 方法 ====================

// buildURL 构建请求 URL
func (c *LLMClient) buildURL(deploymentID, apiVersion, endpoint string) string {
	return fmt.Sprintf("%s/openai/deployments/%s/%s?api-version=%s",
		c.config.Endpoint, deploymentID, endpoint, apiVersion)
}

// doRequest 执行 HTTP 请求
func (c *LLMClient) doRequest(ctx context.Context, method, url string, body interface{}) (*http.Response, error) {
	var bodyReader io.Reader
	if body != nil {
		jsonBytes, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("marshal request body failed: %w", err)
		}
		bodyReader = bytes.NewReader(jsonBytes)
		logger.Debug("LLM request URL: %s, body: %s", url, string(jsonBytes))
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("create request failed: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("api-key", c.config.APIKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do request failed: %w", err)
	}

	return resp, nil
}

// ChatCompletion 非流式对话请求
func (c *LLMClient) ChatCompletion(ctx context.Context, deploymentID, apiVersion string, req *ChatCompletionHTTPRequest) (*ChatCompletionHTTPResponse, error) {
	req.Stream = false

	url := c.buildURL(deploymentID, apiVersion, "v1/chat/completions")

	resp, err := c.doRequest(ctx, http.MethodPost, url, req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body failed: %w", err)
	}

	logger.Debug("LLM response status: %d, body: %s", resp.StatusCode, string(bodyBytes))

	var result ChatCompletionHTTPResponse
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w, body: %s", err, string(bodyBytes))
	}

	if result.Error != nil {
		return nil, fmt.Errorf("LLM API error: %s (type: %s, code: %s)",
			result.Error.Message, result.Error.Type, result.Error.Code)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	return &result, nil
}

// StreamChunkHandler 流式响应处理回调
type StreamChunkHandler func(chunk *ChatCompletionChunkHTTP) error

// ChatCompletionStream 流式对话请求
func (c *LLMClient) ChatCompletionStream(ctx context.Context, deploymentID, apiVersion string, req *ChatCompletionHTTPRequest, handler StreamChunkHandler) error {
	req.Stream = true

	url := c.buildURL(deploymentID, apiVersion, "chat/completions")

	// 流式请求使用独立的 HTTP 客户端，不设置超时
	httpClient := &http.Client{}

	var bodyReader io.Reader
	jsonBytes, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("marshal request body failed: %w", err)
	}
	bodyReader = bytes.NewReader(jsonBytes)
	logger.Debug("LLM stream request URL: %s, body: %s", url, string(jsonBytes))

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bodyReader)
	if err != nil {
		return fmt.Errorf("create request failed: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("api-key", c.config.APIKey)

	resp, err := httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("do request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("unexpected status code: %d, body: %s", resp.StatusCode, string(bodyBytes))
	}

	// 解析 SSE 流
	reader := bufio.NewReader(resp.Body)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("read stream failed: %w", err)
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// SSE 格式: data: {...}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			return nil
		}

		var chunk ChatCompletionChunkHTTP
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			logger.Warn("unmarshal stream chunk failed: %v, data: %s", err, data)
			continue
		}

		if err := handler(&chunk); err != nil {
			return fmt.Errorf("handle chunk failed: %w", err)
		}
	}
}

// GetEmbedding 获取 Embedding 向量
func (c *LLMClient) GetEmbedding(ctx context.Context, deploymentID, apiVersion string, input []string) (*EmbeddingHTTPResponse, error) {
	req := &EmbeddingHTTPRequest{
		Input: input,
	}

	url := c.buildURL(deploymentID, apiVersion, "embeddings")

	resp, err := c.doRequest(ctx, http.MethodPost, url, req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response body failed: %w", err)
	}

	logger.Debug("Embedding response status: %d, body length: %d", resp.StatusCode, len(bodyBytes))

	var result EmbeddingHTTPResponse
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		return nil, fmt.Errorf("unmarshal response failed: %w", err)
	}

	if result.Error != nil {
		return nil, fmt.Errorf("Embedding API error: %s (type: %s, code: %s)",
			result.Error.Message, result.Error.Type, result.Error.Code)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	return &result, nil
}

// Close 关闭客户端
func (c *LLMClient) Close() error {
	c.httpClient.CloseIdleConnections()
	return nil
}
