package hybrid

import (
	"testing"

	"github.com/ethereum/go-ethereum/common/gpu"
)

func newStrategyTestHybrid(cfg *HybridConfig, cpuUtil, gpuUtil float64, gpuQueueDepth int, currentTPS uint64) *HybridProcessor {
	hp := &HybridProcessor{
		config:       cfg,
		logging:      defaultLoggingConfig(),
		loadBalancer: &LoadBalancer{adaptiveRatio: cfg.CPUGPURatio},
		gpuProcessor: &gpu.GPUProcessor{},
	}

	hp.loadBalancer.cpuUtilization = cpuUtil
	hp.loadBalancer.gpuUtilization = gpuUtil
	hp.loadBalancer.gpuQueueDepth = gpuQueueDepth

	hp.mu.Lock()
	hp.stats.CurrentTPS = currentTPS
	hp.mu.Unlock()

	return hp
}

func baseHybridConfig() *HybridConfig {
	return &HybridConfig{
		EnableGPU:             true,
		GPUThreshold:          384,
		CPUGPURatio:           0.5,
		AdaptiveLoadBalancing: false,
		PerformanceMonitoring: false,
		MaxCPUUtilization:     0.95,
		MaxGPUUtilization:     0.98,
		ThroughputTarget:      2000,
		GPUConfig: &gpu.GPUConfig{
			TxWorkers: 80,
		},
	}
}

func TestDetermineProcessingStrategy_ActivatesGPUAtThreshold(t *testing.T) {
	cfg := baseHybridConfig()
	hp := newStrategyTestHybrid(cfg, 0.45, 0.15, 0, 1500)

	strategy, reason := hp.determineProcessingStrategy(cfg.GPUThreshold)
	if strategy == ProcessingStrategyCPUOnly {
		t.Fatalf("expected GPU participation for batch >= threshold, got %s (%s)", strategy.String(), reason)
	}
	if strategy != ProcessingStrategyGPUOnly {
		t.Fatalf("expected GPU-only strategy when GPU is idle, got %s (%s)", strategy.String(), reason)
	}
}

func TestDetermineProcessingStrategy_UsesHybridForGPUBacklog(t *testing.T) {
	cfg := baseHybridConfig()
	hp := newStrategyTestHybrid(cfg, 0.55, 0.7, 60, 2200)

	strategy, reason := hp.determineProcessingStrategy(cfg.GPUThreshold * 2)
	if strategy != ProcessingStrategyHybrid {
		t.Fatalf("expected hybrid strategy for GPU backlog, got %s (%s)", strategy.String(), reason)
	}
}

func TestDetermineProcessingStrategy_ThroughputShortfallAddsCPU(t *testing.T) {
	cfg := baseHybridConfig()
	cfg.ThroughputTarget = 1500

	hp := newStrategyTestHybrid(cfg, 0.6, 0.6, 10, 500)

	strategy, reason := hp.determineProcessingStrategy(cfg.GPUThreshold)
	if strategy != ProcessingStrategyHybrid {
		t.Fatalf("expected hybrid strategy when throughput target is missed, got %s (%s)", strategy.String(), reason)
	}
}

func BenchmarkDetermineProcessingStrategy(b *testing.B) {
	cfg := baseHybridConfig()
	hp := newStrategyTestHybrid(cfg, 0.5, 0.2, 2, 1200)

	batchSize := cfg.GPUThreshold * 2
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		strategy, reason := hp.determineProcessingStrategy(batchSize)
		if strategy == ProcessingStrategyCPUOnly {
			b.Fatalf("unexpected CPU-only strategy for batch >= threshold (%s)", reason)
		}
	}
}
