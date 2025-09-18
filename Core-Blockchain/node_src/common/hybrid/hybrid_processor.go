package hybrid

import (
	"context"
	"runtime"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/gopool"
	"github.com/ethereum/go-ethereum/common/gpu"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/log"
)

// HybridProcessor combines CPU and GPU processing with intelligent load balancing
type HybridProcessor struct {
	// Processing components
	cpuProcessor *gopool.ParallelProcessor
	gpuProcessor *gpu.GPUProcessor

	// Load balancing
	loadBalancer *LoadBalancer

	// Configuration
	config *HybridConfig

	// Statistics
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

>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
=======
	strategyMu            sync.Mutex
	lastStrategy          ProcessingStrategy
	lastStrategyReason    string
	lastStrategyLoggedAt  time.Time
	lastStrategyBatch     int
	lastThroughputWarning time.Time
	lastThroughputSuccess time.Time
	lastImbalanceWarning  time.Time
	lastMetricsLog        time.Time

>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
	// Shutdown coordination
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// HybridConfig holds configuration for hybrid processing
type HybridConfig struct {
	// CPU Configuration
	CPUConfig *gopool.ProcessorConfig `json:"cpuConfig"`

	// GPU Configuration
	GPUConfig *gpu.GPUConfig `json:"gpuConfig"`

	// Load Balancing Configuration
	EnableGPU             bool    `json:"enableGpu"`
	GPUThreshold          int     `json:"gpuThreshold"`          // Minimum batch size for GPU
	CPUGPURatio           float64 `json:"cpuGpuRatio"`           // 0.0 = all CPU, 1.0 = all GPU
	AdaptiveLoadBalancing bool    `json:"adaptiveLoadBalancing"` // Enable adaptive load balancing
	PerformanceMonitoring bool    `json:"performanceMonitoring"` // Enable performance monitoring

	// Performance Thresholds
	MaxCPUUtilization float64       `json:"maxCpuUtilization"` // Max CPU utilization before GPU offload
	MaxGPUUtilization float64       `json:"maxGpuUtilization"` // Max GPU utilization
	LatencyThreshold  time.Duration `json:"latencyThreshold"`  // Max acceptable latency
	ThroughputTarget  uint64        `json:"throughputTarget"`  // Target TPS

	// Memory Management
	MaxMemoryUsage       uint64 `json:"maxMemoryUsage"`       // Max total memory usage
	GPUMemoryReservation uint64 `json:"gpuMemoryReservation"` // Reserved GPU memory
}

// DefaultHybridConfig returns optimized hybrid configuration for i5-13500 (20 threads) + RTX 4000 SFF Ada
// GPU-first strategy to maximize accelerator utilization while protecting AI workloads
func DefaultHybridConfig() *HybridConfig {
	return &HybridConfig{
		CPUConfig:             gopool.DefaultProcessorConfig(),
		GPUConfig:             gpu.DefaultGPUConfig(),
		EnableGPU:             true,
		GPUThreshold:          500,  // Aggressive threshold - use GPU for batches >= 500
		CPUGPURatio:           0.90, // 90% GPU, 10% CPU for maximum GPU utilization
		AdaptiveLoadBalancing: true,
		PerformanceMonitoring: true,
		MaxCPUUtilization:     0.95,                    // Allow CPU to reach 95% before shifting load
		MaxGPUUtilization:     0.98,                    // Push RTX 4000 SFF Ada to 98% utilization
		LatencyThreshold:      25 * time.Millisecond,   // Tighter latency target for faster confirmation
		ThroughputTarget:      2000000,                 // 2M TPS stretch target for GPU-first execution
>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
		ThroughputTarget:      2000000,                 // 2M TPS stretch target for GPU-first execution
=======
		ThroughputTarget:      2000000,                 // 2M TPS stretch target for GPU-first execution
>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
		MaxMemoryUsage:        64 * 1024 * 1024 * 1024, // 64GB RAM available for blockchain workloads
		GPUMemoryReservation:  18 * 1024 * 1024 * 1024, // Reserve 18GB GPU memory (leave 2GB for TinyLlama)
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

// NewHybridProcessor creates a new hybrid processor
func NewHybridProcessor(config *HybridConfig) (*HybridProcessor, error) {
	if config == nil {
		config = DefaultHybridConfig()
	}

	ctx, cancel := context.WithCancel(context.Background())

	// Initialize CPU processor
	cpuProcessor, err := gopool.NewParallelProcessor(config.CPUConfig)
	if err != nil {
		cancel()
		return nil, err
	}

	// Initialize GPU processor if enabled
	var gpuProcessor *gpu.GPUProcessor
	if config.EnableGPU {
		gpuProcessor, err = gpu.NewGPUProcessor(config.GPUConfig)
		if err != nil {
			log.Warn("GPU processor initialization failed, continuing with CPU only", "error", err)
			config.EnableGPU = false
		}
	}

	// Initialize load balancer
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
		ctx:          ctx,
		cancel:       cancel,
	}

	// Start monitoring and load balancing
	if config.PerformanceMonitoring {
		processor.wg.Add(1)
		go processor.performanceMonitor()
	}

	if config.AdaptiveLoadBalancing {
		processor.wg.Add(1)
		go processor.adaptiveLoadBalancer()
	}

	log.Info("Hybrid processor initialized",
		"cpuEnabled", true,
		"gpuEnabled", config.EnableGPU,
		"gpuType", func() string {
			if gpuProcessor != nil {
				switch gpuProcessor.GetGPUType() {
				case gpu.GPUTypeCUDA:
					return "CUDA"
				case gpu.GPUTypeOpenCL:
					return "OpenCL"
				default:
					return "None"
				}
			}
			return "None"
		}(),
	)

	return processor, nil
}

// ProcessTransactionsBatch processes a batch of transactions using optimal CPU/GPU distribution
func (h *HybridProcessor) ProcessTransactionsBatch(txs []*types.Transaction, callback func([]*TransactionResult, error)) error {
	if len(txs) == 0 {
		callback(nil, nil)
		return nil
	}

	start := time.Now()
	batchSize := len(txs)

	strategy, reason := h.determineProcessingStrategy(batchSize)

	h.mu.RLock()
	currentTPS := h.stats.CurrentTPS
	h.mu.RUnlock()

	log.Trace("Hybrid processor received batch", "size", batchSize, "strategy", strategy.String(), "reason", reason, "currentTPS", currentTPS)
>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
	strategy, reason := h.determineProcessingStrategy(batchSize)

	h.mu.RLock()
	currentTPS := h.stats.CurrentTPS
	h.mu.RUnlock()

	log.Trace("Hybrid processor received batch", "size", batchSize, "strategy", strategy.String(), "reason", reason, "currentTPS", currentTPS)
=======
	strategy, reason := h.determineProcessingStrategy(batchSize)

	h.mu.RLock()
	currentTPS := h.stats.CurrentTPS
	h.mu.RUnlock()

	log.Trace("Hybrid processor received batch", "size", batchSize, "strategy", strategy.String(), "reason", reason, "currentTPS", currentTPS)
>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan

	switch strategy {
	case ProcessingStrategyCPUOnly:
		return h.processCPUOnly(txs, callback, start)
	case ProcessingStrategyGPUOnly:
		return h.processGPUOnly(txs, callback, start)
	case ProcessingStrategyHybrid:
		return h.processHybrid(txs, callback, start)
	default:
		return h.processCPUOnly(txs, callback, start)
	}
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

// determineProcessingStrategy decides the optimal processing strategy
func (h *HybridProcessor) determineProcessingStrategy(batchSize int) (ProcessingStrategy, string) {
	strategy := ProcessingStrategyCPUOnly
	reason := "gpu_unavailable"

	if !h.config.EnableGPU || h.gpuProcessor == nil {
		h.recordStrategyDecision(strategy, batchSize, reason, 0, 0, 0)
		return strategy, reason
	}

<<<<<<< HEAD
	// Small batches go to CPU
=======
	// Small batches go to CPU to avoid GPU overhead
>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
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

<<<<<<< HEAD
=======
	strategy = ProcessingStrategyCPUOnly
	reason = "default_cpu_path"

>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
	// If CPU is overloaded, prefer GPU
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

	// If GPU is underutilized and batch is large, use hybrid
	if batchSize > h.config.GPUThreshold*2 && gpuUtil < 0.5 {
		strategy = ProcessingStrategyHybrid
		reason = "large_batch_low_gpu_utilization"
		h.recordStrategyDecision(strategy, batchSize, reason, cpuUtil, gpuUtil, adaptiveRatio)
		return strategy, reason
	}

<<<<<<< HEAD
	// Default to GPU for large batches
=======
	// Default to GPU for very large batches
>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
	if batchSize > h.config.GPUThreshold*5 {
		strategy = ProcessingStrategyGPUOnly
		reason = "very_large_batch"
		h.recordStrategyDecision(strategy, batchSize, reason, cpuUtil, gpuUtil, adaptiveRatio)
		return strategy, reason
	}

<<<<<<< HEAD
	return ProcessingStrategyCPUOnly
=======
	h.recordStrategyDecision(strategy, batchSize, reason, cpuUtil, gpuUtil, adaptiveRatio)
	return strategy, reason
}

func (h *HybridProcessor) recordStrategyDecision(strategy ProcessingStrategy, batchSize int, reason string, cpuUtil, gpuUtil, adaptiveRatio float64) {
	logNow := false
	var sinceLastLog time.Duration

	h.strategyMu.Lock()
	if strategy != h.lastStrategy || reason != h.lastStrategyReason || batchSize != h.lastStrategyBatch || time.Since(h.lastStrategyLoggedAt) > 10*time.Second {
		if !h.lastStrategyLoggedAt.IsZero() {
			sinceLastLog = time.Since(h.lastStrategyLoggedAt)
		}
		h.lastStrategy = strategy
		h.lastStrategyReason = reason
		h.lastStrategyBatch = batchSize
		h.lastStrategyLoggedAt = time.Now()
		logNow = true
	}
	h.strategyMu.Unlock()

	if !logNow {
		return
	}

	h.mu.RLock()
	currentTPS := h.stats.CurrentTPS
	avgLatency := h.stats.AvgLatency
	loadRatio := h.stats.LoadBalancingRatio
	h.mu.RUnlock()

	log.Debug("Hybrid strategy decision", "strategy", strategy.String(), "reason", reason, "batchSize", batchSize, "cpuUtil", cpuUtil, "gpuUtil", gpuUtil, "adaptiveRatio", adaptiveRatio, "currentTPS", currentTPS, "avgLatency", avgLatency, "loadRatio", loadRatio, "sinceLastLog", sinceLastLog)
>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
}

// processCPUOnly processes transactions using CPU only
func (h *HybridProcessor) processCPUOnly(txs []*types.Transaction, callback func([]*TransactionResult, error), start time.Time) error {
	log.Debug("Processing batch on CPU", "size", len(txs))

	results := make([]*TransactionResult, len(txs))
	var wg sync.WaitGroup
	var mu sync.Mutex

	// Process in parallel using CPU workers
	batchSize := 100 // Process in smaller batches
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

			// Process batch
			batchResults := make([]*TransactionResult, len(batch))
			for j, tx := range batch {
				batchResults[j] = &TransactionResult{
					Hash:      tx.Hash(),
					Valid:     true, // Simplified validation
					GasUsed:   tx.Gas(),
					Processed: true,
				}
			}

			// Store results
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

	// Wait for completion
	wg.Wait()

	// Update statistics
	duration := time.Since(start)
	h.updateStats(uint64(len(txs)), 0, duration, ProcessingStrategyCPUOnly)

<<<<<<< HEAD
=======
	throughput := float64(0)
	if duration > 0 {
		throughput = float64(len(txs)) / duration.Seconds()
	}
	log.Debug("CPU batch completed", "size", len(txs), "duration", duration, "throughput", throughput)

>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
	callback(results, nil)
	return nil
}

// processGPUOnly processes transactions using GPU only
func (h *HybridProcessor) processGPUOnly(txs []*types.Transaction, callback func([]*TransactionResult, error), start time.Time) error {
	if h.gpuProcessor == nil {
		log.Debug("GPU processor unavailable, falling back to CPU", "size", len(txs))
		return h.processCPUOnly(txs, callback, start)
	}

<<<<<<< HEAD
=======
	log.Debug("Processing batch on GPU", "size", len(txs))

>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
	// Process transactions using GPU
	err := h.gpuProcessor.ProcessTransactionsBatch(txs, func(results []*gpu.TxResult, err error) {
		if err != nil {
			log.Error("GPU batch processing failed", "size", len(txs), "error", err)
			callback(nil, err)
			return
		}

		// Convert GPU results to hybrid results
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

		// Update statistics
		duration := time.Since(start)
		h.updateStats(0, uint64(len(txs)), duration, ProcessingStrategyGPUOnly)

<<<<<<< HEAD
		callback(hybridResults, nil)
	})

=======
		throughput := float64(0)
		if duration > 0 {
			throughput = float64(len(txs)) / duration.Seconds()
		}
		log.Debug("GPU batch completed", "size", len(results), "duration", duration, "throughput", throughput)

		callback(hybridResults, nil)
	})

	if err != nil {
		log.Error("Failed to submit batch to GPU", "size", len(txs), "error", err)
	}

>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
	return err
}

// processHybrid processes transactions using both CPU and GPU
func (h *HybridProcessor) processHybrid(txs []*types.Transaction, callback func([]*TransactionResult, error), start time.Time) error {
	if h.gpuProcessor == nil {
		log.Debug("GPU processor unavailable during hybrid path, falling back to CPU", "size", len(txs))
		return h.processCPUOnly(txs, callback, start)
	}

	// Determine split ratio
	h.loadBalancer.mu.RLock()
	ratio := h.loadBalancer.adaptiveRatio
	h.loadBalancer.mu.RUnlock()

	// Split transactions
	gpuCount := int(float64(len(txs)) * ratio)
	cpuCount := len(txs) - gpuCount

<<<<<<< HEAD
=======
	log.Debug("Processing hybrid batch", "size", len(txs), "gpuCount", gpuCount, "cpuCount", cpuCount, "ratio", ratio)

>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
	gpuTxs := txs[:gpuCount]
	cpuTxs := txs[gpuCount:]

	results := make([]*TransactionResult, len(txs))
	var wg sync.WaitGroup
	var processingError error
	var mu sync.Mutex

	// Process GPU batch
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

				// Convert and store GPU results
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

	// Process CPU batch
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

				// Store CPU results
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

	// Wait for both to complete
	wg.Wait()

	if processingError != nil {
		log.Warn("Hybrid batch failed", "error", processingError, "gpuCount", gpuCount, "cpuCount", cpuCount)
		callback(nil, processingError)
		return processingError
	}

	// Update statistics
	duration := time.Since(start)
	h.updateStats(uint64(cpuCount), uint64(gpuCount), duration, ProcessingStrategyHybrid)

<<<<<<< HEAD
=======
	throughput := float64(0)
	if duration > 0 {
		throughput = float64(len(txs)) / duration.Seconds()
	}
	log.Debug("Hybrid batch completed", "size", len(txs), "gpuCount", gpuCount, "cpuCount", cpuCount, "duration", duration, "throughput", throughput)

>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
	callback(results, nil)
	return nil
}

// processCPUBatch processes a batch using CPU
func (h *HybridProcessor) processCPUBatch(txs []*types.Transaction, callback func([]*TransactionResult, error)) error {
	log.Trace("Processing CPU sub-batch", "size", len(txs))

	results := make([]*TransactionResult, len(txs))

	// Simple CPU processing
	for i, tx := range txs {
		results[i] = &TransactionResult{
			Hash:      tx.Hash(),
			Valid:     true, // Simplified validation
			GasUsed:   tx.Gas(),
			Processed: true,
		}
	}

	callback(results, nil)

	log.Trace("CPU sub-batch completed", "size", len(txs))
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

	var (
		currentTPS  uint64
		avgLatency  time.Duration
		loadRatio   float64
		shouldWarn  bool
		shouldCheer bool
	)

	now := time.Now()

	h.mu.Lock()
<<<<<<< HEAD
	defer h.mu.Unlock()

	h.stats.TotalProcessed += cpuProcessed + gpuProcessed
	h.stats.CPUProcessed += cpuProcessed
	h.stats.GPUProcessed += gpuProcessed

	// Update average latency
	if h.stats.AvgLatency == 0 {
		h.stats.AvgLatency = duration
	} else {
		h.stats.AvgLatency = (h.stats.AvgLatency + duration) / 2
	}

	// Calculate current TPS
=======
	h.stats.TotalProcessed += totalProcessed
	h.stats.CPUProcessed += cpuProcessed
	h.stats.GPUProcessed += gpuProcessed

>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
	if duration > 0 {
		if h.stats.AvgLatency == 0 {
			h.stats.AvgLatency = duration
		} else {
			h.stats.AvgLatency = (h.stats.AvgLatency + duration) / 2
		}
	}

<<<<<<< HEAD
	// Update load balancing ratio
=======
	if duration > 0 && totalProcessed > 0 {
		currentTPS = uint64(float64(totalProcessed) / duration.Seconds())
		h.stats.CurrentTPS = currentTPS
	} else {
		currentTPS = h.stats.CurrentTPS
	}

>>>>>>> remotes/origin/codex/resolve-merge-conflicts-and-implement-plan
	if h.stats.TotalProcessed > 0 {
		h.stats.LoadBalancingRatio = float64(h.stats.GPUProcessed) / float64(h.stats.TotalProcessed)
	}

	avgLatency = h.stats.AvgLatency
	loadRatio = h.stats.LoadBalancingRatio

	if h.config != nil && h.config.ThroughputTarget > 0 && totalProcessed > 0 {
		target := h.config.ThroughputTarget
		if currentTPS >= target && now.Sub(h.lastThroughputSuccess) > 30*time.Second {
			shouldCheer = true
			h.lastThroughputSuccess = now
		} else if currentTPS < target && totalProcessed >= uint64(h.config.GPUThreshold) && now.Sub(h.lastThroughputWarning) > 5*time.Second {
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
	log.Debug("Hybrid batch statistics", "strategy", strategy.String(), "cpuProcessed", cpuProcessed, "gpuProcessed", gpuProcessed, "duration", duration, "tps", currentTPS, "avgLatency", avgLatency, "loadRatio", loadRatio)
}

// performanceMonitor continuously monitors system performance
func (h *HybridProcessor) performanceMonitor() {
	defer h.wg.Done()

	ticker := time.NewTicker(1 * time.Second)
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
	// Safety check for nil pointers
	if h == nil || h.loadBalancer == nil {
		log.Trace("Hybrid performance monitor waiting for initialization")
		return
	}

	// Get CPU stats with nil check
	var cpuUtil float64
	var avgCPULatency time.Duration
	var cpuRunning int
	if h.cpuProcessor != nil && h.config != nil && h.config.CPUConfig != nil {
		cpuStats := h.cpuProcessor.GetStats()
		if h.config.CPUConfig.TxWorkers > 0 {
			cpuUtil = float64(cpuStats.TxPoolRunning) / float64(h.config.CPUConfig.TxWorkers)
		}
		avgCPULatency = cpuStats.AvgProcessTime
		cpuRunning = cpuStats.TxPoolRunning
	}

	// Get GPU stats with nil check
	var gpuUtil float64
	var avgGPULatency time.Duration
	var gpuQueueSize int
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

	// Update load balancer metrics with safety checks
	if h.loadBalancer != nil {
		h.loadBalancer.mu.Lock()
		h.loadBalancer.cpuUtilization = cpuUtil
		h.loadBalancer.gpuUtilization = gpuUtil
		h.loadBalancer.avgCPULatency = avgCPULatency
		h.loadBalancer.avgGPULatency = avgGPULatency

		// Add performance snapshot
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

	// Update hybrid stats with safety checks
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

	// Update memory usage (simplified)
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	h.stats.MemoryUsage = m.Alloc
	memoryUsage = h.stats.MemoryUsage
	// TODO: integrate GPU memory telemetry from GPU processor when available

	currentTPS = h.stats.CurrentTPS
	loadRatio = h.stats.LoadBalancingRatio
	if h.config != nil {
		throughputTarget = h.config.ThroughputTarget

		if h.config.EnableGPU {
			if cpuUtil > h.config.MaxCPUUtilization && gpuUtil < 0.5 && now.Sub(h.lastImbalanceWarning) > 5*time.Second {
				shouldWarn = true
				warnReason = "cpu_utilization_spike"
				h.lastImbalanceWarning = now
			} else if gpuUtil > h.config.MaxGPUUtilization && now.Sub(h.lastImbalanceWarning) > 5*time.Second {
				shouldWarn = true
				warnReason = "gpu_saturated"
				h.lastImbalanceWarning = now
			}
		}
	}

	if now.Sub(h.lastMetricsLog) >= time.Second {
		shouldTrace = true
		h.lastMetricsLog = now
	}

	h.mu.Unlock()

	if shouldTrace {
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

	// Don't adjust too frequently
	if time.Since(h.loadBalancer.lastAdjustment) < 10*time.Second {
		return
	}

	cpuUtil := h.loadBalancer.cpuUtilization
	gpuUtil := h.loadBalancer.gpuUtilization
	currentRatio := h.loadBalancer.adaptiveRatio

	// Adjustment logic
	newRatio := currentRatio

	// If CPU is overloaded, shift more work to GPU
	if cpuUtil > h.config.MaxCPUUtilization && gpuUtil < h.config.MaxGPUUtilization {
		newRatio = min(1.0, currentRatio+0.1)
	}

	// If GPU is overloaded, shift more work to CPU
	if gpuUtil > h.config.MaxGPUUtilization && cpuUtil < h.config.MaxCPUUtilization {
		newRatio = max(0.0, currentRatio-0.1)
	}

	// If both are underutilized, prefer GPU for better throughput
	if cpuUtil < 0.5 && gpuUtil < 0.5 && h.stats.CurrentTPS < h.config.ThroughputTarget {
		newRatio = min(1.0, currentRatio+0.05)
	}

	// Apply adjustment
	if newRatio != currentRatio {
		h.loadBalancer.adaptiveRatio = newRatio
		h.loadBalancer.lastAdjustment = time.Now()

		log.Debug("Load balancing ratio adjusted",
			"oldRatio", currentRatio,
			"newRatio", newRatio,
			"cpuUtil", cpuUtil,
			"gpuUtil", gpuUtil,
		)
	}
}

// GetStats returns current hybrid processor statistics
func (h *HybridProcessor) GetStats() HybridStats {
	h.mu.RLock()
	defer h.mu.RUnlock()

	return h.stats
}

// Close gracefully shuts down the hybrid processor
func (h *HybridProcessor) Close() error {
	log.Info("Shutting down hybrid processor...")

	// Cancel context
	h.cancel()

	// Wait for background goroutines
	h.wg.Wait()

	// Close CPU processor
	if h.cpuProcessor != nil {
		h.cpuProcessor.Close()
	}

	// Close GPU processor
	if h.gpuProcessor != nil {
		h.gpuProcessor.Close()
	}

	log.Info("Hybrid processor shutdown complete")
	return nil
}

// Helper functions
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

// Global hybrid processor instance
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
