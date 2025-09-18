package logging

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/log"
)

// FileLogger handles writing logs to specific files
type FileLogger struct {
	mu       sync.Mutex
	logDir   string
	files    map[string]*os.File
	enabled  bool
}

var (
	globalFileLogger *FileLogger
	once             sync.Once
)

// InitFileLogger initializes the global file logger
func InitFileLogger(logDir string) error {
	var err error
	once.Do(func() {
		globalFileLogger = &FileLogger{
			logDir:  logDir,
			files:   make(map[string]*os.File),
			enabled: true,
		}
		
		// Create log directory if it doesn't exist
		if err = os.MkdirAll(logDir, 0755); err != nil {
			return
		}
		
		log.Info("File logger initialized", "logDir", logDir)
	})
	return err
}

// GetFileLogger returns the global file logger
func GetFileLogger() *FileLogger {
	if globalFileLogger == nil {
		// Initialize with default directory if not already initialized
		InitFileLogger("/root/blockchain-logs")
	}
	return globalFileLogger
}

// LogToFile writes a log entry to a specific file
func (fl *FileLogger) LogToFile(filename, level, message string, fields ...interface{}) {
	if !fl.enabled {
		return
	}
	
	fl.mu.Lock()
	defer fl.mu.Unlock()
	
	// Get or create file handle
	file, err := fl.getOrCreateFile(filename)
	if err != nil {
		log.Error("Failed to open log file", "filename", filename, "error", err)
		return
	}
	
	// Format log entry
	timestamp := time.Now().Format("2006-01-02 15:04:05.000")
	logEntry := fmt.Sprintf("[%s] %s: %s", timestamp, level, message)
	
	// Add fields if provided
	if len(fields) > 0 {
		for i := 0; i < len(fields); i += 2 {
			if i+1 < len(fields) {
				logEntry += fmt.Sprintf(" %v=%v", fields[i], fields[i+1])
			}
		}
	}
	
	logEntry += "\n"
	
	// Write to file
	if _, err := file.WriteString(logEntry); err != nil {
		log.Error("Failed to write to log file", "filename", filename, "error", err)
	}
	
	// Flush to ensure data is written
	file.Sync()
}

// getOrCreateFile gets an existing file handle or creates a new one
func (fl *FileLogger) getOrCreateFile(filename string) (*os.File, error) {
	if file, exists := fl.files[filename]; exists {
		return file, nil
	}
	
	// Create new file
	fullPath := filepath.Join(fl.logDir, filename)
	file, err := os.OpenFile(fullPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return nil, err
	}
	
	fl.files[filename] = file
	return file, nil
}

// Close closes all open log files
func (fl *FileLogger) Close() error {
	fl.mu.Lock()
	defer fl.mu.Unlock()
	
	for filename, file := range fl.files {
		if err := file.Close(); err != nil {
			log.Error("Failed to close log file", "filename", filename, "error", err)
		}
	}
	
	fl.files = make(map[string]*os.File)
	fl.enabled = false
	return nil
}

// Convenience functions for different log types

// LogGPU logs GPU-related events
func LogGPU(level, message string, fields ...interface{}) {
	fl := GetFileLogger()
	fl.LogToFile("gpu.log", level, message, fields...)
}

// LogPerformance logs performance metrics
func LogPerformance(level, message string, fields ...interface{}) {
	fl := GetFileLogger()
	fl.LogToFile("performance.log", level, message, fields...)
}

// LogError logs error events
func LogError(level, message string, fields ...interface{}) {
	fl := GetFileLogger()
	fl.LogToFile("errors.log", level, message, fields...)
}

// LogTransaction logs transaction processing events
func LogTransaction(level, message string, fields ...interface{}) {
	fl := GetFileLogger()
	fl.LogToFile("transactions.log", level, message, fields...)
}

// LogHybrid logs hybrid processor events
func LogHybrid(level, message string, fields ...interface{}) {
	fl := GetFileLogger()
	fl.LogToFile("hybrid.log", level, message, fields...)
}

// LogMining logs mining-related events
func LogMining(level, message string, fields ...interface{}) {
	fl := GetFileLogger()
	fl.LogToFile("mining.log", level, message, fields...)
}

// LogNetwork logs network-related events
func LogNetwork(level, message string, fields ...interface{}) {
	fl := GetFileLogger()
	fl.LogToFile("network.log", level, message, fields...)
}

// LogSystem logs general system events
func LogSystem(level, message string, fields ...interface{}) {
	fl := GetFileLogger()
	fl.LogToFile("system.log", level, message, fields...)
}
