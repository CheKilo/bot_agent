package main

import (
	"bot_agent/gateway/internal/config"
	"bot_agent/gateway/internal/llmproxy"
	"bot_agent/gateway/internal/logger"
	"bot_agent/gateway/internal/pb"
	"bot_agent/gateway/internal/storage"
	"net"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

func main() {
	// 加载配置文件
	cfg := make(map[string]interface{})
	config.Load("./config/dev.yaml", &cfg)

	// 初始化日志器
	logCfg := &logger.Config{
		Level:       logger.ParseLevel(config.Get(cfg, "logger.level", "INFO")),
		FilePath:    config.Get(cfg, "logger.file_path", "./logs/gateway.log"),
		EnableFile:  config.GetBool(cfg, "logger.enable_file", true),
		EnableStdio: config.GetBool(cfg, "logger.enable_stdio", true),
		MaxSize:     config.GetInt(cfg, "logger.max_size", 100),
		MaxBackups:  config.GetInt(cfg, "logger.max_backups", 7),
		MaxAge:      config.GetInt(cfg, "logger.max_age", 30),
		Compress:    config.GetBool(cfg, "logger.compress", true),
	}

	if err := logger.Init(logCfg); err != nil {
		panic("init logger error: " + err.Error())
	}
	defer logger.Close()

	// 创建存储服务
	storageService := storage.NewStorageService()
	defer storageService.Close()

	// 初始化 MySQL 客户端
	mysqlDSN := config.Get(cfg, "mysql.dsn", "")
	if mysqlDSN != "" {
		logger.Info("Initializing MySQL client, DSN: %s", mysqlDSN)
		mysqlClient, err := storage.NewMySQLClient(mysqlDSN)
		if err != nil {
			logger.Fatal("Failed to create MySQL client: %v", err)
		}
		storageService.SetMySQLClient(mysqlClient)
		logger.Info("MySQL client initialized successfully")
	} else {
		logger.Warn("MySQL DSN not configured, MySQL storage disabled")
	}

	// 初始化 Milvus 客户端
	milvusAddr := config.Get(cfg, "milvus.addr", "")
	if milvusAddr != "" {
		milvusDimension := config.GetInt(cfg, "milvus.dimension", 1536)
		milvusCollection := config.Get(cfg, "milvus.collection", "memory_vectors")
		logger.Info("Initializing Milvus client, addr: %s, dimension: %d, collection: %s", milvusAddr, milvusDimension, milvusCollection)

		milvusClient, err := storage.NewMilvusClient(milvusAddr, milvusDimension, milvusCollection)
		if err != nil {
			logger.Fatal("Failed to create Milvus client: %v", err)
		}
		storageService.SetMilvusClient(milvusClient)
		logger.Info("Milvus client initialized successfully")
	} else {
		logger.Warn("Milvus addr not configured, Milvus storage disabled")
	}

	// 初始化 LLM 代理服务
	llmEndpoint := config.Get(cfg, "llm.endpoint", "")
	llmAPIKey := config.Get(cfg, "llm.api_key", "")
	var llmService *llmproxy.LLMProxyService
	if llmEndpoint != "" && llmAPIKey != "" {
		llmTimeout := config.GetInt(cfg, "llm.timeout", 60)
		llmMaxRetries := config.GetInt(cfg, "llm.max_retries", 3)
		logger.Info("Initializing LLM Proxy service, endpoint: %s", llmEndpoint)

		llmConfig := llmproxy.LLMProxyConfig{
			Endpoint:   llmEndpoint,
			APIKey:     llmAPIKey,
			Timeout:    time.Duration(llmTimeout) * time.Second,
			MaxRetries: llmMaxRetries,
		}
		llmService = llmproxy.NewLLMProxyService(llmConfig)
		logger.Info("LLM Proxy service initialized successfully")
	} else {
		logger.Warn("LLM endpoint or api_key not configured, LLM Proxy disabled")
	}
	defer func() {
		if llmService != nil {
			llmService.Close()
		}
	}()

	grpcServer := grpc.NewServer()
	pb.RegisterStorageServiceServer(grpcServer, storageService)
	// 注册 LLM 代理服务
	if llmService != nil {
		pb.RegisterLLMProxyServiceServer(grpcServer, llmService)
		logger.Info("LLM Proxy service registered")
	}
	reflection.Register(grpcServer)

	addr := config.Get(cfg, "server.addr", ":50051")
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		logger.Fatal("Failed to listen: %v", err)
	}

	logger.Info("gRPC Storage Server listening on %s", addr)
	if err := grpcServer.Serve(listener); err != nil {
		logger.Fatal("Failed to serve: %v", err)
	}
}
