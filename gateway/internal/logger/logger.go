package logger

import (
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"gopkg.in/natefinch/lumberjack.v2"
)

// Level 定义日志级别
type Level int

const (
	DEBUG Level = iota
	INFO
	WARN
	ERROR
	FATAL
)

// levelNames 日志级别名称映射
var levelNames = map[Level]string{
	DEBUG: "DEBUG",
	INFO:  "INFO",
	WARN:  "WARN",
	ERROR: "ERROR",
	FATAL: "FATAL",
}

// Logger 日志器结构体
type Logger struct {
	mu           sync.Mutex
	level        Level
	logger       *log.Logger
	lumberjack   *lumberjack.Logger // 使用 lumberjack 进行日志轮转
	enableFile   bool
	enableStdio  bool
}

// Config 日志配置
type Config struct {
	Level       Level  // 日志级别
	FilePath    string // 日志文件路径（为空则不写入文件）
	EnableFile  bool   // 是否启用文件输出
	EnableStdio bool   // 是否启用控制台输出

	// 日志归档配置
	MaxSize    int  // 单个日志文件最大大小（MB），默认 100MB
	MaxBackups int  // 保留的旧日志文件最大数量，默认 7 个
	MaxAge     int  // 保留旧日志文件的最大天数，默认 30 天
	Compress   bool // 是否压缩旧日志文件，默认 true
}

// DefaultConfig 返回默认配置
func DefaultConfig() *Config {
	return &Config{
		Level:       INFO,
		FilePath:    "",
		EnableFile:  false,
		EnableStdio: true,
		MaxSize:     100, // 100MB
		MaxBackups:  7,   // 保留 7 个备份
		MaxAge:      30,  // 保留 30 天
		Compress:    true,
	}
}

var (
	defaultLogger *Logger
	once          sync.Once
)

// Init 初始化默认日志器
func Init(cfg *Config) error {
	var err error
	once.Do(func() {
		defaultLogger, err = NewLogger(cfg)
	})
	return err
}

// NewLogger 创建新的日志器
func NewLogger(cfg *Config) (*Logger, error) {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	// 设置默认值
	if cfg.MaxSize <= 0 {
		cfg.MaxSize = 100
	}
	if cfg.MaxBackups <= 0 {
		cfg.MaxBackups = 7
	}
	if cfg.MaxAge <= 0 {
		cfg.MaxAge = 30
	}

	l := &Logger{
		level:       cfg.Level,
		enableFile:  cfg.EnableFile,
		enableStdio: cfg.EnableStdio,
	}

	var writers []io.Writer

	// 控制台输出
	if cfg.EnableStdio {
		writers = append(writers, os.Stdout)
	}

	// 文件输出（使用 lumberjack 进行日志轮转）
	if cfg.EnableFile && cfg.FilePath != "" {
		// 确保日志目录存在
		dir := filepath.Dir(cfg.FilePath)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, fmt.Errorf("mkdir %s: %w", dir, err)
		}

		// 创建 lumberjack 日志轮转器
		l.lumberjack = &lumberjack.Logger{
			Filename:   cfg.FilePath,   // 日志文件路径
			MaxSize:    cfg.MaxSize,    // 单文件最大 MB
			MaxBackups: cfg.MaxBackups, // 保留旧文件数量
			MaxAge:     cfg.MaxAge,     // 保留天数
			Compress:   cfg.Compress,   // 是否压缩
			LocalTime:  true,           // 使用本地时间命名备份文件
		}
		writers = append(writers, l.lumberjack)
	}

	// 如果没有任何输出，默认使用 stdout
	if len(writers) == 0 {
		writers = append(writers, os.Stdout)
	}

	// 创建多输出 writer
	multiWriter := io.MultiWriter(writers...)
	l.logger = log.New(multiWriter, "", 0)

	return l, nil
}

// Close 关闭日志器，释放文件资源
func (l *Logger) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.lumberjack != nil {
		return l.lumberjack.Close()
	}
	return nil
}

// Rotate 手动触发日志轮转
func (l *Logger) Rotate() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.lumberjack != nil {
		return l.lumberjack.Rotate()
	}
	return nil
}

// SetLevel 设置日志级别
func (l *Logger) SetLevel(level Level) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.level = level
}

// formatMessage 格式化日志消息
func (l *Logger) formatMessage(level Level, format string, args ...interface{}) string {
	timestamp := time.Now().Format("2006-01-02 15:04:05.000")
	levelStr := levelNames[level]
	message := fmt.Sprintf(format, args...)
	return fmt.Sprintf("[%s] [%s] %s", timestamp, levelStr, message)
}

// log 内部日志记录方法
func (l *Logger) log(level Level, format string, args ...interface{}) {
	if level < l.level {
		return
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	msg := l.formatMessage(level, format, args...)
	l.logger.Println(msg)
}

// Debug 记录 DEBUG 级别日志
func (l *Logger) Debug(format string, args ...interface{}) {
	l.log(DEBUG, format, args...)
}

// Info 记录 INFO 级别日志
func (l *Logger) Info(format string, args ...interface{}) {
	l.log(INFO, format, args...)
}

// Warn 记录 WARN 级别日志
func (l *Logger) Warn(format string, args ...interface{}) {
	l.log(WARN, format, args...)
}

// Error 记录 ERROR 级别日志
func (l *Logger) Error(format string, args ...interface{}) {
	l.log(ERROR, format, args...)
}

// Fatal 记录 FATAL 级别日志并退出程序
func (l *Logger) Fatal(format string, args ...interface{}) {
	l.log(FATAL, format, args...)
	os.Exit(1)
}

// ========== 包级别的便捷函数 ==========

// GetDefault 获取默认日志器
func GetDefault() *Logger {
	if defaultLogger == nil {
		// 如果未初始化，使用默认配置创建
		defaultLogger, _ = NewLogger(DefaultConfig())
	}
	return defaultLogger
}

// Debug 包级别 DEBUG 日志
func Debug(format string, args ...interface{}) {
	GetDefault().Debug(format, args...)
}

// Info 包级别 INFO 日志
func Info(format string, args ...interface{}) {
	GetDefault().Info(format, args...)
}

// Warn 包级别 WARN 日志
func Warn(format string, args ...interface{}) {
	GetDefault().Warn(format, args...)
}

// Error 包级别 ERROR 日志
func Error(format string, args ...interface{}) {
	GetDefault().Error(format, args...)
}

// Fatal 包级别 FATAL 日志
func Fatal(format string, args ...interface{}) {
	GetDefault().Fatal(format, args...)
}

// Close 关闭默认日志器
func Close() error {
	if defaultLogger != nil {
		return defaultLogger.Close()
	}
	return nil
}

// Rotate 手动触发默认日志器轮转
func Rotate() error {
	if defaultLogger != nil {
		return defaultLogger.Rotate()
	}
	return nil
}

// ParseLevel 从字符串解析日志级别
func ParseLevel(levelStr string) Level {
	switch levelStr {
	case "DEBUG", "debug":
		return DEBUG
	case "INFO", "info":
		return INFO
	case "WARN", "warn":
		return WARN
	case "ERROR", "error":
		return ERROR
	case "FATAL", "fatal":
		return FATAL
	default:
		return INFO
	}
}
