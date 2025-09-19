package hybrid

import (
	"context"
	"errors"
	"runtime"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/gopool"
	"github.com/ethereum/go-ethereum/common/gpu"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

// LoggingConfig controls how hybrid processor events are emitted.
type LoggingConfig struct {
	EnableDebug           bool          `json:"enableDebug"`
	StrategyCooldown      time.Duration `json:"strategyCooldown"`
	MetricsSampleInterval time.Duration `json:"metricsSampleInterval"`
	WarningCooldown       time.Duration `json:"warningCooldown"`
	SuccessCooldown       time.Duration `json:"successCooldown"`
	BatchLogInterval      time.Duration `json:"batchLogInterval"`
}

func defaultLoggingConfig() *LoggingConfig {
	return &LoggingConfig{
		EnableDebug:           false,
		StrategyCooldown:      10 * time.Second,
		MetricsSampleInterval: time.Second,
		WarningCooldown:       5 * time.Second,
		SuccessCooldown:       30 * time.Second,
		BatchLogInterval:      2 * time.Second,
	}
}

func mergeLoggingConfig(cfg *LoggingConfig) *LoggingConfig {
	def := defaultLoggingConfig()
	if cfg == nil {
		return def
	}
	merged := *cfg
	if merged.StrategyCooldown <= 0 {
		merged.StrategyCooldown = def.StrategyCooldown
	}
	if merged.MetricsSampleInterval <= 0 {
		merged.MetricsSampleInterval = def.MetricsSampleInterval
	}
	if merged.WarningCooldown <= 0 {
		merged.WarningCooldown = def.WarningCooldown
	}
	if merged.SuccessCooldown <= 0 {
		merged.SuccessCooldown = def.SuccessCooldown
	}
	if merged.BatchLogInterval < 0 {
		merged.BatchLogInterval = def.BatchLogInterval
	}
	return &merged
}

// HybridProcessor combines CPU and GPU processing with intelligent load balancing
type HybridProcessor struct {
	cpuProcessor *gopool.ParallelProcessor
	gpuProcessor *gpu.GPUProcessor

	loadBalancer *LoadBalancer
	config       *HybridConfig
	logging      *LoggingConfig

	mu    sync.RWMutex
	stats HybridStats

	strategyMu            sync.Mutex
	lastStrategy          ProcessingStrategy
	lastStrategyReason    string
	lastStrategyLoggedAt  time.Time
	lastStrategyBatch     int
	lastThroughputWarning time.Time
	lastThroughputSuccess time.Time
	lastImbalanceWarning  time.Time
	lastMetricsLog        time.Time

	loggingMu    sync.Mutex
	lastBatchLog time.Time

	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// HybridConfig holds configuration for hybrid processing
type HybridConfig struct {
	CPUConfig *gopool.ProcessorConfig `json:"cpuConfig"`
	GPUConfig *gpu.GPUConfig          `json:"gpuConfig"`

	EnableGPU             bool    `json:"enableGpu"`
	GPUThreshold          int     `json:"gpuThreshold"`
	CPUGPURatio           float64 `json:"cpuGpuRatio"`
	AdaptiveLoadBalancing bool    `json:"adaptiveLoadBalancing"`
	PerformanceMonitoring bool    `json:"performanceMonitoring"`

	MaxCPUUtilization float64       `json:"maxCpuUtilization"`
	MaxGPUUtilization float64       `json:"maxGpuUtilization"`
	LatencyThreshold  time.Duration `json:"latencyThreshold"`
	ThroughputTarget  uint64        `json:"throughputTarget"`

	MaxMemoryUsage       uint64 `json:"maxMemoryUsage"`
	GPUMemoryReservation uint64 `json:"gpuMemoryReservation"`

	Logging *LoggingConfig `json:"logging,omitempty"`
}

// DefaultHybridConfig returns optimized hybrid configuration for i5-13500 (20 threads) + RTX 4000 SFF Ada
// GPU-first strategy to maximize accelerator utilization while protecting AI workloads
func DefaultHybridConfig() *HybridConfig {
	return &HybridConfig{
		CPUConfig:             gopool.DefaultProcessorConfig(),
		GPUConfig:             gpu.DefaultGPUConfig(),
		EnableGPU:             true,
		GPUThreshold:          500,
		CPUGPURatio:           0.90,
		AdaptiveLoadBalancing: true,
		PerformanceMonitoring: true,
		MaxCPUUtilization:     0.95,
		MaxGPUUtilization:     0.98,
		LatencyThreshold:      25 * time.Millisecond,
		ThroughputTarget:      2000000,
		MaxMemoryUsage:        64 * 1024 * 1024 * 1024,
		GPUMemoryReservation:  18 * 1024 * 1024 * 1024,
		Logging:               defaultLoggingConfig(),
	}
}

// LoadBalancer manages workload distribution between CPU and GPU
type LoadBalancer struct {
	mu                 sync.RWMutex
	cpuUtilization     float64
	gpuUtilization     float64
	avgCPULatency      time.Duration
	avgGPULatency      time.Duration
	cpuThroughput      uint64
	gpuThroughput      uint64
	adaptiveRatio      float64
	lastAdjustment     time.Time
	performanceHistory []PerformanceSnapshot
}

// PerformanceSnapshot captures performance metrics at a point in time
type PerformanceSnapshot struct {
	Timestamp     time.Time
	CPULatency    time.Duration
	GPULatency    time.Duration
	CPUThroughput uint64
	GPUThroughput uint64
	TotalTPS      uint64
}

// HybridStats holds statistics for hybrid processing
type HybridStats struct {
	TotalProcessed     uint64        `json:"totalProcessed"`
	CPUProcessed       uint64        `json:"cpuProcessed"`
	GPUProcessed       uint64        `json:"gpuProcessed"`
	AvgLatency         time.Duration `json:"avgLatency"`
	CurrentTPS         uint64        `json:"currentTps"`
	CPUUtilization     float64       `json:"cpuUtilization"`
	GPUUtilization     float64       `json:"gpuUtilization"`
	LoadBalancingRatio float64       `json:"loadBalancingRatio"`
	MemoryUsage        uint64        `json:"memoryUsage"`
	GPUMemoryUsage     uint64        `json:"gpuMemoryUsage"`
}

// GPUStatus describes the availability of GPU acceleration from the hybrid processor.
type GPUStatus struct {
	ConfigEnabled     bool        `json:"configEnabled"`
	Available         bool        `json:"available"`
	Type              gpu.GPUType `json:"type"`
	DeviceCount       int         `json:"deviceCount"`
	UnavailableReason string      `json:"unavailableReason,omitempty"`
}

// NewHybridProcessor creates a new hybrid processor
func NewHybridProcessor(config *HybridConfig) (*HybridProcessor, error) {
	if config == nil {
		config = DefaultHybridConfig()
	}
	logging := mergeLoggingConfig(config.Logging)
	config.Logging = logging

	ctx, cancel := context.WithCancel(context.Background())

	cpuProcessor, err := gopool.NewParallelProcessor(config.CPUConfig)
	if err != nil {
		cancel()
		return nil, err
	}

	var gpuProcessor *gpu.GPUProcessor
	if config.EnableGPU {
		gpuProcessor, err = gpu.NewGPUProcessor(config.GPUConfig)
		if err != nil {
			log.Warn("GPU processor initialization failed, continuing with CPU only", "error", err)
			config.EnableGPU = false
		}
	}

	loadBalancer := &LoadBalancer{
		adaptiveRatio:      config.CPUGPURatio,
		lastAdjustment:     time.Now(),
		performanceHistory: make([]PerformanceSnapshot, 0, 100),
	}

	processor := &HybridProcessor{
		cpuProcessor: cpuProcessor,
		gpuProcessor: gpuProcessor,
		loadBalancer: loadBalancer,
		config:       config,
		logging:      logging,
		ctx:          ctx,
		cancel:       cancel,
	}

	if config.PerformanceMonitoring {
		processor.wg.Add(1)
		go processor.performanceMonitor()
	}

	if config.AdaptiveLoadBalancing {
		processor.wg.Add(1)
		go processor.adaptiveLoadBalancer()
	}

	log.Info("Hybrid processor initialized",
		"enableGPU", config.EnableGPU,
		"cpuWorkers", func() int {
			if config.CPUConfig != nil {
				return config.CPUConfig.TxWorkers
			}
			return 0
		}(),
		"gpuWorkers", func() int {
			if config.GPUConfig != nil {
				return config.GPUConfig.TxWorkers
			}
			return 0
		}(),
		"targetTPS", config.ThroughputTarget,
	)

	return processor, nil
}

// ProcessingStrategy represents different processing approaches
type ProcessingStrategy int

const (
	ProcessingStrategyCPUOnly ProcessingStrategy = iota
	ProcessingStrategyGPUOnly
	ProcessingStrategyHybrid
)

func (p ProcessingStrategy) String() string {
	switch p {
	case ProcessingStrategyCPUOnly:
		return "cpu-only"
	case ProcessingStrategyGPUOnly:
		return "gpu-only"
	case ProcessingStrategyHybrid:
		return "hybrid"
	default:
		return "unknown"
	}
}

func (h *HybridProcessor) debugLogsEnabled() bool {
	return h.logging != nil && h.logging.EnableDebug
}

func (h *HybridProcessor) logDebug(msg string, ctx ...interface{}) {
	if h.debugLogsEnabled() {
		log.Debug(msg, ctx...)
	}
}

// determineProcessingStrategy decides the optimal processing strategy
func (h *HybridProcessor) determineProcessingStrategy(batchSize int) (ProcessingStrategy, string) {
	strategy := ProcessingStrategyCPUOnly
	reason := "gpu_unavailable"

	if !h.config.EnableGPU || h.gpuProcessor == nil {
		h.recordStrategyDecision(strategy, batchSize, reason, 0, 0, 0)
		return strategy, reason
	}

	if batchSize < h.config.GPUThreshold {
		reason = "batch_below_gpu_threshold"
		h.recordStrategyDecision(strategy, batchSize, reason, 0, 0, 0)
		return strategy, reason
	}

	h.loadBalancer.mu.RLock()
	cpuUtil := h.loadBalancer.cpuUtilization
	gpuUtil := h.loadBalancer.gpuUtilization
	adaptiveRatio := h.loadBalancer.adaptiveRatio
	h.loadBalancer.mu.RUnlock()

	if cpuUtil > h.config.MaxCPUUtilization {
		if gpuUtil < h.config.MaxGPUUtilization {
			strategy = ProcessingStrategyGPUOnly
			reason = "cpu_overloaded"
		} else {
			strategy = ProcessingStrategyHybrid
			reason = "cpu_hot_gpu_hot"
		}
		h.recordStrategyDecision(strategy, batchSize, reason, cpuUtil, gpuUtil, adaptiveRatio)
		return strategy, reason
	}

	if batchSize > h.config.GPUThreshold*2 && gpuUtil < 0.5 {
		strategy = ProcessingStrategyHybrid
		reason = "large_batch_low_gpu_utilization"
		h.recordStrategyDecision(strategy, batchSize, reason, cpuUtil, gpuUtil, adaptiveRatio)
		return strategy, reason
	}

	if batchSize > h.config.GPUThreshold*5 {
		strategy = ProcessingStrategyGPUOnly
		reason = "very_large_batch"
		h.recordStrategyDecision(strategy, batchSize, reason, cpuUtil, gpuUtil, adaptiveRatio)
		return strategy, reason
	}

	h.recordStrategyDecision(strategy, batchSize, "default_cpu_path", cpuUtil, gpuUtil, adaptiveRatio)
	return ProcessingStrategyCPUOnly, "default_cpu_path"
}

func (h *HybridProcessor) recordStrategyDecision(strategy ProcessingStrategy, batchSize int, reason string, cpuUtil, gpuUtil, adaptiveRatio float64) {
	interval := h.logging.StrategyCooldown
	if interval <= 0 {
		interval = 10 * time.Second
	}
	now := time.Now()

	var (
		sinceLastLog     time.Duration
		shouldLogDetail  bool
		previousStrategy ProcessingStrategy
	)

	h.strategyMu.Lock()
	previousStrategy = h.lastStrategy
	if strategy != h.lastStrategy || reason != h.lastStrategyReason || batchSize != h.lastStrategyBatch || now.Sub(h.lastStrategyLoggedAt) >= interval {
		shouldLogDetail = true
		if !h.lastStrategyLoggedAt.IsZero() {
			sinceLastLog = now.Sub(h.lastStrategyLoggedAt)
		}
		h.lastStrategyLoggedAt = now
		h.lastStrategyBatch = batchSize
		h.lastStrategyReason = reason
	}
	h.lastStrategy = strategy
	h.strategyMu.Unlock()

	if strategy != previousStrategy {
		log.Info("Hybrid strategy change",
			"from", previousStrategy.String(),
			"to", strategy.String(),
			"reason", reason,
			"batchSize", batchSize,
			"cpuUtil", cpuUtil,
			"gpuUtil", gpuUtil,
			"adaptiveRatio", adaptiveRatio,
		)

	}

	if shouldLogDetail {
		switch reason {
		case "gpu_unavailable":
			status := h.GetGPUStatus()
			log.Warn("GPU acceleration unavailable",
				"configured", status.ConfigEnabled,
				"available", status.Available,
				"type", status.Type.String(),
				"devices", status.DeviceCount,
				"detail", status.UnavailableReason,
			)
		case "batch_below_gpu_threshold":
			threshold := 0
			if h.config != nil {
				threshold = h.config.GPUThreshold
			}
			log.Info("GPU batch skipped below threshold",
				"batchSize", batchSize,
				"threshold", threshold,
			)
		}
	}

	if !shouldLogDetail || !h.debugLogsEnabled() {
		return
	}

	h.mu.RLock()
	currentTPS := h.stats.CurrentTPS
	avgLatency := h.stats.AvgLatency
	loadRatio := h.stats.LoadBalancingRatio
	h.mu.RUnlock()

	h.logDebug("Hybrid strategy decision",
		"strategy", strategy.String(),
		"reason", reason,
		"batchSize", batchSize,
		"cpuUtil", cpuUtil,
		"gpuUtil", gpuUtil,
		"adaptiveRatio", adaptiveRatio,
		"currentTPS", currentTPS,
		"avgLatency", avgLatency,
		"loadRatio", loadRatio,
		"sinceLastLog", sinceLastLog,
	)
}

// processCPUOnly processes transactions using CPU only
func (h *HybridProcessor) processCPUOnly(txs []*types.Transaction, callback func([]*TransactionResult, error), start time.Time) error {
	h.logDebug("Processing batch on CPU", "size", len(txs))

	results := make([]*TransactionResult, len(txs))
	var wg sync.WaitGroup
	var mu sync.Mutex

	batchSize := 100
	for i := 0; i < len(txs); i += batchSize {
		end := i + batchSize
		if end > len(txs) {
			end = len(txs)
		}

		wg.Add(1)
		batch := txs[i:end]
		batchStart := i

		err := h.cpuProcessor.SubmitTxTask(func() error {
			defer wg.Done()

			batchResults := make([]*TransactionResult, len(batch))
			for j, tx := range batch {
				batchResults[j] = &TransactionResult{
					Hash:      tx.Hash(),
					Valid:     true,
					GasUsed:   tx.Gas(),
					Processed: true,
				}
			}

			mu.Lock()
			copy(results[batchStart:batchStart+len(batch)], batchResults)
			mu.Unlock()
			return nil
		}, nil)

		if err != nil {
			log.Error("Failed to submit CPU batch", "error", err, "batchStart", i, "batchSize", len(batch))
			return err
		}
	}

	wg.Wait()

	duration := time.Since(start)
	h.updateStats(uint64(len(txs)), 0, duration, ProcessingStrategyCPUOnly)

	if duration > 0 {
		throughput := float64(len(txs)) / duration.Seconds()
		h.logDebug("CPU batch completed", "size", len(txs), "duration", duration, "throughput", throughput)
	} else {
		h.logDebug("CPU batch completed", "size", len(txs), "duration", duration)
	}

	callback(results, nil)
	return nil
}

// processGPUOnly processes transactions using GPU only
func (h *HybridProcessor) processGPUOnly(txs []*types.Transaction, callback func([]*TransactionResult, error), start time.Time) error {
	if h.gpuProcessor == nil {
		log.Debug("GPU processor unavailable, falling back to CPU", "size", len(txs))
		return h.processCPUOnly(txs, callback, start)
	}

	h.logDebug("Processing batch on GPU", "size", len(txs))

	err := h.gpuProcessor.ProcessTransactionsBatch(txs, func(results []*gpu.TxResult, err error) {
		if err != nil {
			log.Error("GPU batch processing failed", "size", len(txs), "error", err)
			callback(nil, err)
			return
		}

		hybridResults := make([]*TransactionResult, len(results))
		for i, result := range results {
			hybridResults[i] = &TransactionResult{
				Hash:      result.Hash,
				Valid:     result.Valid,
				GasUsed:   result.GasUsed,
				Error:     result.Error,
				Processed: true,
			}
		}

		duration := time.Since(start)
		h.updateStats(0, uint64(len(txs)), duration, ProcessingStrategyGPUOnly)

		if duration > 0 {
			throughput := float64(len(results)) / duration.Seconds()
			h.logDebug("GPU batch completed", "size", len(results), "duration", duration, "throughput", throughput)
		} else {
			h.logDebug("GPU batch completed", "size", len(results), "duration", duration)
		}

		callback(hybridResults, nil)
	})

	if err != nil {
		log.Error("Failed to submit batch to GPU", "size", len(txs), "error", err)
	}
	return err
}

// processHybrid processes transactions using both CPU and GPU
func (h *HybridProcessor) processHybrid(txs []*types.Transaction, callback func([]*TransactionResult, error), start time.Time) error {
	if h.gpuProcessor == nil {
		log.Debug("GPU processor unavailable during hybrid path, falling back to CPU", "size", len(txs))
		return h.processCPUOnly(txs, callback, start)
	}

	h.loadBalancer.mu.RLock()
	ratio := h.loadBalancer.adaptiveRatio
	h.loadBalancer.mu.RUnlock()

	gpuCount := int(float64(len(txs)) * ratio)
	cpuCount := len(txs) - gpuCount

	h.logDebug("Processing hybrid batch", "size", len(txs), "gpuCount", gpuCount, "cpuCount", cpuCount, "ratio", ratio)

	gpuTxs := txs[:gpuCount]
	cpuTxs := txs[gpuCount:]

	results := make([]*TransactionResult, len(txs))
	var wg sync.WaitGroup
	var processingError error
	var mu sync.Mutex

	if len(gpuTxs) > 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()

			err := h.gpuProcessor.ProcessTransactionsBatch(gpuTxs, func(gpuResults []*gpu.TxResult, err error) {
				mu.Lock()
				defer mu.Unlock()

				if err != nil {
					processingError = err
					log.Error("GPU segment failed during hybrid batch", "error", err, "gpuCount", len(gpuTxs))
					return
				}

				for i, result := range gpuResults {
					results[i] = &TransactionResult{
						Hash:      result.Hash,
						Valid:     result.Valid,
						GasUsed:   result.GasUsed,
						Error:     result.Error,
						Processed: true,
					}
				}
			})

			if err != nil {
				mu.Lock()
				processingError = err
				mu.Unlock()
				log.Error("Failed to submit hybrid GPU segment", "error", err, "gpuCount", len(gpuTxs))
			}
		}()
	}

	if len(cpuTxs) > 0 {
		wg.Add(1)
		go func() {
			defer wg.Done()

			err := h.processCPUBatch(cpuTxs, func(cpuResults []*TransactionResult, err error) {
				mu.Lock()
				defer mu.Unlock()

				if err != nil {
					processingError = err
					log.Error("CPU segment failed during hybrid batch", "error", err, "cpuCount", len(cpuTxs))
					return
				}

				copy(results[gpuCount:], cpuResults)
			})

			if err != nil {
				mu.Lock()
				processingError = err
				mu.Unlock()
				log.Error("Failed to execute hybrid CPU segment", "error", err, "cpuCount", len(cpuTxs))
			}
		}()
	}

	wg.Wait()

	if processingError != nil {
		log.Warn("Hybrid batch failed", "error", processingError, "gpuCount", gpuCount, "cpuCount", cpuCount)
		callback(nil, processingError)
		return processingError
	}

	duration := time.Since(start)
	h.updateStats(uint64(cpuCount), uint64(gpuCount), duration, ProcessingStrategyHybrid)

	if duration > 0 {
		throughput := float64(len(txs)) / duration.Seconds()
		h.logDebug("Hybrid batch completed", "size", len(txs), "gpuCount", gpuCount, "cpuCount", cpuCount, "duration", duration, "throughput", throughput)
	} else {
		h.logDebug("Hybrid batch completed", "size", len(txs), "gpuCount", gpuCount, "cpuCount", cpuCount, "duration", duration)
	}

	callback(results, nil)
	return nil
}

// processCPUBatch processes a batch using CPU
func (h *HybridProcessor) processCPUBatch(txs []*types.Transaction, callback func([]*TransactionResult, error)) error {
	if h.debugLogsEnabled() {
		log.Trace("Processing CPU sub-batch", "size", len(txs))
	}

	results := make([]*TransactionResult, len(txs))
	for i, tx := range txs {
		results[i] = &TransactionResult{
			Hash:      tx.Hash(),
			Valid:     true,
			GasUsed:   tx.Gas(),
			Processed: true,
		}
	}

	callback(results, nil)

	if h.debugLogsEnabled() {
		log.Trace("CPU sub-batch completed", "size", len(txs))
	}
	return nil
}

// TransactionResult holds the result of hybrid transaction processing
type TransactionResult struct {
	Hash      common.Hash `json:"hash"`
	Valid     bool        `json:"valid"`
	GasUsed   uint64      `json:"gasUsed"`
	Error     error       `json:"error,omitempty"`
	Processed bool        `json:"processed"`
}

// updateStats updates processing statistics
func (h *HybridProcessor) updateStats(cpuProcessed, gpuProcessed uint64, duration time.Duration, strategy ProcessingStrategy) {
	totalProcessed := cpuProcessed + gpuProcessed
	now := time.Now()

	var (
		currentTPS  uint64
		avgLatency  time.Duration
		loadRatio   float64
		shouldWarn  bool
		shouldCheer bool
	)

	h.mu.Lock()
	h.stats.TotalProcessed += totalProcessed
	h.stats.CPUProcessed += cpuProcessed
	h.stats.GPUProcessed += gpuProcessed

	if duration > 0 {
		if h.stats.AvgLatency == 0 {
			h.stats.AvgLatency = duration
		} else {
			h.stats.AvgLatency = (h.stats.AvgLatency + duration) / 2
		}
	}

	if duration > 0 && totalProcessed > 0 {
		currentTPS = uint64(float64(totalProcessed) / duration.Seconds())
		h.stats.CurrentTPS = currentTPS
	} else {
		currentTPS = h.stats.CurrentTPS
	}

	if h.stats.TotalProcessed > 0 {
		h.stats.LoadBalancingRatio = float64(h.stats.GPUProcessed) / float64(h.stats.TotalProcessed)
	}

	avgLatency = h.stats.AvgLatency
	loadRatio = h.stats.LoadBalancingRatio

	warnCooldown := h.logging.WarningCooldown
	if warnCooldown <= 0 {
		warnCooldown = 5 * time.Second
	}
	successCooldown := h.logging.SuccessCooldown
	if successCooldown <= 0 {
		successCooldown = 30 * time.Second
	}

	if h.config != nil && h.config.ThroughputTarget > 0 && totalProcessed > 0 {
		target := h.config.ThroughputTarget
		if currentTPS >= target && now.Sub(h.lastThroughputSuccess) >= successCooldown {
			shouldCheer = true
			h.lastThroughputSuccess = now
		} else if currentTPS < target && totalProcessed >= uint64(h.config.GPUThreshold) && now.Sub(h.lastThroughputWarning) >= warnCooldown {
			shouldWarn = true
			h.lastThroughputWarning = now
		}
	}
	h.mu.Unlock()

	h.logBatchCompletion(strategy, cpuProcessed, gpuProcessed, duration, currentTPS, avgLatency, loadRatio)

	if shouldWarn {
		log.Warn("Hybrid throughput below target", "strategy", strategy.String(), "tps", currentTPS, "target", h.config.ThroughputTarget, "avgLatency", avgLatency, "cpuProcessed", cpuProcessed, "gpuProcessed", gpuProcessed, "duration", duration, "loadRatio", loadRatio)
	}

	if shouldCheer {
		log.Info("Hybrid throughput met target", "strategy", strategy.String(), "tps", currentTPS, "target", h.config.ThroughputTarget, "avgLatency", avgLatency, "cpuProcessed", cpuProcessed, "gpuProcessed", gpuProcessed, "duration", duration, "loadRatio", loadRatio)
	}
}

func (h *HybridProcessor) logBatchCompletion(strategy ProcessingStrategy, cpuProcessed, gpuProcessed uint64, duration time.Duration, currentTPS uint64, avgLatency time.Duration, loadRatio float64) {
	if !h.debugLogsEnabled() {
		return
	}
	interval := h.logging.BatchLogInterval
	if interval > 0 {
		h.loggingMu.Lock()
		if !h.lastBatchLog.IsZero() && time.Since(h.lastBatchLog) < interval {
			h.loggingMu.Unlock()
			return
		}
		h.lastBatchLog = time.Now()
		h.loggingMu.Unlock()
	}
	h.logDebug("Hybrid batch statistics", "strategy", strategy.String(), "cpuProcessed", cpuProcessed, "gpuProcessed", gpuProcessed, "duration", duration, "tps", currentTPS, "avgLatency", avgLatency, "loadRatio", loadRatio)
}

// performanceMonitor continuously monitors system performance
func (h *HybridProcessor) performanceMonitor() {
	defer h.wg.Done()

	interval := h.logging.MetricsSampleInterval
	if interval <= 0 {
		interval = time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-h.ctx.Done():
			return
		case <-ticker.C:
			h.collectPerformanceMetrics()
		}
	}
}

// collectPerformanceMetrics collects current performance metrics
func (h *HybridProcessor) collectPerformanceMetrics() {
	if h == nil || h.loadBalancer == nil {
		if h.debugLogsEnabled() {
			log.Trace("Hybrid performance monitor waiting for initialization")
		}
		return
	}

	var (
		cpuUtil       float64
		avgCPULatency time.Duration
		cpuRunning    int
	)
	if h.cpuProcessor != nil && h.config != nil && h.config.CPUConfig != nil {
		cpuStats := h.cpuProcessor.GetStats()
		if h.config.CPUConfig.TxWorkers > 0 {
			cpuUtil = float64(cpuStats.TxPoolRunning) / float64(h.config.CPUConfig.TxWorkers)
		}
		avgCPULatency = cpuStats.AvgProcessTime
		cpuRunning = cpuStats.TxPoolRunning
	}

	var (
		gpuUtil       float64
		avgGPULatency time.Duration
		gpuQueueSize  int
	)
	if h.gpuProcessor != nil && h.config != nil && h.config.GPUConfig != nil {
		if h.gpuProcessor.IsGPUAvailable() {
			gpuStats := h.gpuProcessor.GetStats()
			if h.config.GPUConfig.TxWorkers > 0 {
				gpuUtil = float64(gpuStats.TxQueueSize) / float64(h.config.GPUConfig.TxWorkers)
			}
			avgGPULatency = gpuStats.AvgTxTime
			gpuQueueSize = gpuStats.TxQueueSize
		}
	}

	if h.loadBalancer != nil {
		h.loadBalancer.mu.Lock()
		h.loadBalancer.cpuUtilization = cpuUtil
		h.loadBalancer.gpuUtilization = gpuUtil
		h.loadBalancer.avgCPULatency = avgCPULatency
		h.loadBalancer.avgGPULatency = avgGPULatency

		snapshot := PerformanceSnapshot{
			Timestamp:     time.Now(),
			CPULatency:    avgCPULatency,
			GPULatency:    avgGPULatency,
			CPUThroughput: h.stats.CPUProcessed,
			GPUThroughput: h.stats.GPUProcessed,
			TotalTPS:      h.stats.CurrentTPS,
		}

		h.loadBalancer.performanceHistory = append(h.loadBalancer.performanceHistory, snapshot)
		if len(h.loadBalancer.performanceHistory) > 100 {
			h.loadBalancer.performanceHistory = h.loadBalancer.performanceHistory[1:]
		}
		h.loadBalancer.mu.Unlock()
	}

	var (
		currentTPS       uint64
		loadRatio        float64
		throughputTarget uint64
		memoryUsage      uint64
		shouldWarn       bool
		warnReason       string
		shouldTrace      bool
	)

	now := time.Now()

	h.mu.Lock()
	h.stats.CPUUtilization = cpuUtil
	h.stats.GPUUtilization = gpuUtil

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	h.stats.MemoryUsage = m.Alloc
	memoryUsage = h.stats.MemoryUsage

	currentTPS = h.stats.CurrentTPS
	loadRatio = h.stats.LoadBalancingRatio
	if h.config != nil {
		throughputTarget = h.config.ThroughputTarget

		if h.config.EnableGPU {
			if cpuUtil > h.config.MaxCPUUtilization && gpuUtil < 0.5 && now.Sub(h.lastImbalanceWarning) > h.logging.WarningCooldown {
				shouldWarn = true
				warnReason = "cpu_utilization_spike"
				h.lastImbalanceWarning = now
			} else if gpuUtil > h.config.MaxGPUUtilization && now.Sub(h.lastImbalanceWarning) > h.logging.WarningCooldown {
				shouldWarn = true
				warnReason = "gpu_saturated"
				h.lastImbalanceWarning = now
			}
		}
	}

	interval := h.logging.MetricsSampleInterval
	if interval <= 0 {
		interval = time.Second
	}
	if now.Sub(h.lastMetricsLog) >= interval {
		shouldTrace = true
		h.lastMetricsLog = now
	}

	h.mu.Unlock()

	if shouldTrace && h.debugLogsEnabled() {
		log.Trace("Hybrid performance snapshot", "cpuUtil", cpuUtil, "gpuUtil", gpuUtil, "cpuLatency", avgCPULatency, "gpuLatency", avgGPULatency, "cpuActive", cpuRunning, "gpuQueue", gpuQueueSize, "currentTPS", currentTPS, "targetTPS", throughputTarget, "loadRatio", loadRatio, "memory", memoryUsage)
	}

	if shouldWarn {
		log.Warn("Hybrid load imbalance detected", "reason", warnReason, "cpuUtil", cpuUtil, "gpuUtil", gpuUtil, "cpuLatency", avgCPULatency, "gpuLatency", avgGPULatency, "currentTPS", currentTPS, "targetTPS", throughputTarget, "cpuActive", cpuRunning, "gpuQueue", gpuQueueSize)
	}
}

// adaptiveLoadBalancer adjusts the CPU/GPU ratio based on performance
func (h *HybridProcessor) adaptiveLoadBalancer() {
	defer h.wg.Done()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-h.ctx.Done():
			return
		case <-ticker.C:
			h.adjustLoadBalancing()
		}
	}
}

// adjustLoadBalancing adjusts the load balancing ratio
func (h *HybridProcessor) adjustLoadBalancing() {
	h.loadBalancer.mu.Lock()
	defer h.loadBalancer.mu.Unlock()

	if time.Since(h.loadBalancer.lastAdjustment) < 10*time.Second {
		return
	}

	cpuUtil := h.loadBalancer.cpuUtilization
	gpuUtil := h.loadBalancer.gpuUtilization
	currentRatio := h.loadBalancer.adaptiveRatio

	newRatio := currentRatio

	if cpuUtil > h.config.MaxCPUUtilization && gpuUtil < h.config.MaxGPUUtilization {
		newRatio = min(1.0, currentRatio+0.1)
	}

	if gpuUtil > h.config.MaxGPUUtilization && cpuUtil < h.config.MaxCPUUtilization {
		newRatio = max(0.0, currentRatio-0.1)
	}

	if cpuUtil < 0.5 && gpuUtil < 0.5 && h.stats.CurrentTPS < h.config.ThroughputTarget {
		newRatio = min(1.0, currentRatio+0.05)
	}

	if newRatio != currentRatio {
		h.loadBalancer.adaptiveRatio = newRatio
		h.loadBalancer.lastAdjustment = time.Now()

		h.logDebug("Load balancing ratio adjusted",
			"oldRatio", currentRatio,
			"newRatio", newRatio,
			"cpuUtil", cpuUtil,
			"gpuUtil", gpuUtil,
		)
	}
}

// ProcessTransactionsBatch processes a batch of transactions using the optimal strategy
func (h *HybridProcessor) ProcessTransactionsBatch(txs []*types.Transaction, callback func([]*TransactionResult, error)) error {
	if callback == nil {
		return errors.New("hybrid processor requires a callback")
	}

	start := time.Now()
	var once sync.Once
	safeCallback := func(results []*TransactionResult, err error) {
		once.Do(func() {
			callback(results, err)
		})
	}

	if len(txs) == 0 {
		safeCallback([]*TransactionResult{}, nil)
		return nil
	}

	strategy, reason := h.determineProcessingStrategy(len(txs))

	h.logDebug("Processing transaction batch",
		"size", len(txs),
		"strategy", strategy.String(),
		"reason", reason)

	var err error
	switch strategy {
	case ProcessingStrategyCPUOnly:
		err = h.processCPUOnly(txs, safeCallback, start)
	case ProcessingStrategyGPUOnly:
		err = h.processGPUOnly(txs, safeCallback, start)
	case ProcessingStrategyHybrid:
		err = h.processHybrid(txs, safeCallback, start)
	default:
		err = h.processCPUOnly(txs, safeCallback, start)
	}

	if err != nil {
		duration := time.Since(start)
		h.logDebug("Hybrid batch failed",
			"size", len(txs),
			"strategy", strategy.String(),
			"reason", reason,
			"duration", duration,
			"error", err)
		h.logBatchCompletion(strategy, 0, 0, duration, 0, 0, 0)
		safeCallback(nil, err)
		return err
	}

	return nil
}

// GetStats returns current hybrid processor statistics
func (h *HybridProcessor) GetStats() HybridStats {
	h.mu.RLock()
	defer h.mu.RUnlock()

	return h.stats
}

// GetGPUStatus exposes the current GPU availability and configuration state.
func (h *HybridProcessor) GetGPUStatus() GPUStatus {
	status := GPUStatus{}

	if h == nil {
		status.UnavailableReason = "hybrid_processor_nil"
		return status
	}

	if h.config != nil {
		status.ConfigEnabled = h.config.EnableGPU
	}

	if !status.ConfigEnabled {
		status.UnavailableReason = "disabled_in_config"
		return status
	}

	if h.gpuProcessor == nil {
		status.UnavailableReason = "not_initialized"
		return status
	}

	status.Type = h.gpuProcessor.GetGPUType()
	status.DeviceCount = h.gpuProcessor.GetDeviceCount()
	status.Available = h.gpuProcessor.IsGPUAvailable() && status.DeviceCount > 0

	if !status.Available {
		status.UnavailableReason = "no_device_detected"
	}

	return status
}

// Close gracefully shuts down the hybrid processor
func (h *HybridProcessor) Close() error {
	log.Info("Shutting down hybrid processor...")

	h.cancel()
	h.wg.Wait()

	if h.cpuProcessor != nil {
		h.cpuProcessor.Close()
	}

	if h.gpuProcessor != nil {
		h.gpuProcessor.Close()
	}

	log.Info("Hybrid processor shutdown complete")
	return nil
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

var globalHybridProcessor *HybridProcessor

// InitGlobalHybridProcessor initializes the global hybrid processor
func InitGlobalHybridProcessor(config *HybridConfig) error {
	if config == nil {
		log.Info("Initializing global hybrid processor with default configuration")
	} else {
		log.Info("Initializing global hybrid processor", "enableGPU", config.EnableGPU, "cpuWorkers", func() int {
			if config.CPUConfig != nil {
				return config.CPUConfig.TxWorkers
			}
			return 0
		}(), "gpuWorkers", func() int {
			if config.GPUConfig != nil {
				return config.GPUConfig.TxWorkers
			}
			return 0
		}(), "targetTPS", config.ThroughputTarget)
	}

	if globalHybridProcessor != nil {
		log.Info("Reinitializing global hybrid processor")
		globalHybridProcessor.Close()
	}

	var err error
	globalHybridProcessor, err = NewHybridProcessor(config)
	return err
}

// GetGlobalHybridProcessor returns the global hybrid processor
func GetGlobalHybridProcessor() *HybridProcessor {
	return globalHybridProcessor
}

// CloseGlobalHybridProcessor closes the global hybrid processor
func CloseGlobalHybridProcessor() error {
	if globalHybridProcessor != nil {
		log.Info("Closing global hybrid processor")
		return globalHybridProcessor.Close()
	}
	return nil
}
