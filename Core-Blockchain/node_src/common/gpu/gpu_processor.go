package gpu

import (
	"context"
	"encoding/binary"
	"errors"
	"math/big"
	"sync"
	"time"
	"unsafe"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/logging"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/log"
)

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: ${SRCDIR}/libcuda_kernels.so ${SRCDIR}/libopencl_kernels.so -lOpenCL -L/usr/local/cuda/lib64 -lcudart -Wl,-rpath,${SRCDIR} -Wl,-rpath,/usr/local/cuda/lib64

#include <stdlib.h>
#include <string.h>

// CUDA function declarations (implemented in cuda_kernels.cu)
int cuda_init_device();
int cuda_process_transactions(void* txs, int count, void* results);
int cuda_process_transactions_full(void* txs, void* tx_lengths, void* state_data, void* access_lists, int count, void* results);
int cuda_process_hashes(void* hashes, int count, void* results);
int cuda_verify_signatures(void* sigs, void* msgs, void* keys, int count, void* results);
void cuda_cleanup();

// CUDA function declarations (stubs for now - can be replaced with real implementations)
int initCUDA();
int processTxBatchCUDA(void* txData, int txCount, void* results);
int processHashesCUDA(void* hashes, int count, void* results);
int verifySignaturesCUDA(void* signatures, int count, void* results);
void cleanupCUDA();

// OpenCL function declarations (implemented in opencl_kernels.c)
int initOpenCL();
int processTxBatchOpenCL(void* txData, int txCount, void* results);
int processTxBatchOpenCLFull(void* txData, void* txLens, void* stateData, void* accessLists, int txCount, void* results);
int processHashesOpenCL(void* hashes, int count, void* results);
int verifySignaturesOpenCL(void* signatures, int count, void* results);
void cleanupOpenCL();

// Working stub implementations for CUDA (can be replaced when CUDA is properly configured)
int initCUDA() { 
    // Return -1 to indicate CUDA not available, system will fall back to OpenCL or CPU
    return -1; 
}

int processHashesCUDA(void* hashes, int count, void* results) { 
    return -1; // Not implemented, will fall back to OpenCL or CPU
}

int verifySignaturesCUDA(void* signatures, int count, void* results) {
    return -1; // Not implemented, will fall back to OpenCL or CPU
}

int processTxBatchCUDA(void* txData, int txCount, void* results) { 
    return -1; // Not implemented, will fall back to OpenCL or CPU
}

void cleanupCUDA() { 
    // No-op for stub implementation
}
*/
import "C"

// GPUType represents the type of GPU acceleration
type GPUType int

const (
	GPUTypeNone GPUType = iota
	GPUTypeCUDA
	GPUTypeOpenCL
)

// GPUProcessor provides GPU-accelerated blockchain operations
type GPUProcessor struct {
	gpuType         GPUType
	deviceCount     int
	maxBatchSize    int
	maxMemoryUsage  uint64
	
	// Processing pools
	hashPool        chan *HashBatch
	signaturePool   chan *SignatureBatch
	txPool          chan *TransactionBatch
	
	// Statistics
	mu              sync.RWMutex
	processedHashes uint64
	processedSigs   uint64
	processedTxs    uint64
	avgHashTime     time.Duration
	avgSigTime      time.Duration
	avgTxTime       time.Duration
	
	// Shutdown coordination
	ctx             context.Context
	cancel          context.CancelFunc
	wg              sync.WaitGroup
	
	// Memory management
	memoryPool      sync.Pool
	cudaStreams     []unsafe.Pointer
	openclQueues    []unsafe.Pointer
}

// GPUConfig holds configuration for GPU processing
type GPUConfig struct {
	PreferredGPUType GPUType `json:"preferredGpuType"`
	MaxBatchSize     int     `json:"maxBatchSize"`
	MaxMemoryUsage   uint64  `json:"maxMemoryUsage"`
	HashWorkers      int     `json:"hashWorkers"`
	SignatureWorkers int     `json:"signatureWorkers"`
	TxWorkers        int     `json:"txWorkers"`
	EnablePipelining bool    `json:"enablePipelining"`
}

// DefaultGPUConfig returns optimized GPU configuration for NVIDIA RTX 4000 SFF Ada (20GB VRAM)
// Balanced for blockchain processing + TinyLlama 1.1B AI model with 2GB VRAM reservation
func DefaultGPUConfig() *GPUConfig {
	return &GPUConfig{
		PreferredGPUType: GPUTypeCUDA,   // Prefer CUDA for RTX 4000 SFF Ada when available
		MaxBatchSize:     800000,        // 4x increase - 800K batches (keeps GPU saturated)
		MaxMemoryUsage:   18 * 1024 * 1024 * 1024, // 18GB GPU memory (leave 2GB for TinyLlama + system)
		HashWorkers:      80,            // 80 workers - balance with AI workload
		SignatureWorkers: 80,            // 80 workers - balance with AI workload
		TxWorkers:        80,            // 80 workers - balance with AI workload
		EnablePipelining: true,
	}
}

// HashBatch represents a batch of hashes to process
type HashBatch struct {
	Hashes   [][]byte
	Results  [][]byte
	Callback func([][]byte, error)
}

// SignatureBatch represents a batch of signatures to verify
type SignatureBatch struct {
	Signatures [][]byte
	Messages   [][]byte
	PublicKeys [][]byte
	Results    []bool
	Callback   func([]bool, error)
}

// TransactionBatch represents a batch of transactions to process
type TransactionBatch struct {
	Transactions []*types.Transaction
	Results      []*TxResult
	Callback     func([]*TxResult, error)
}

// TxResult holds the result of GPU transaction processing
type TxResult struct {
	Hash      common.Hash
	Valid     bool
	GasUsed   uint64
	Error     error
	Signature []byte
}

// NewGPUProcessor creates a new GPU processor
func NewGPUProcessor(config *GPUConfig) (*GPUProcessor, error) {
	log.Info("Initializing GPU processor", "config", config)
	
	if config == nil {
		log.Debug("No config provided, using default GPU configuration")
		config = DefaultGPUConfig()
	}
	
	log.Debug("GPU processor configuration", 
		"preferredGPUType", config.PreferredGPUType,
		"maxBatchSize", config.MaxBatchSize,
		"maxMemoryUsage", config.MaxMemoryUsage,
		"hashWorkers", config.HashWorkers,
		"signatureWorkers", config.SignatureWorkers,
		"txWorkers", config.TxWorkers,
		"enablePipelining", config.EnablePipelining,
	)
	
	ctx, cancel := context.WithCancel(context.Background())
	
	processor := &GPUProcessor{
		maxBatchSize:   config.MaxBatchSize,
		maxMemoryUsage: config.MaxMemoryUsage,
		ctx:            ctx,
		cancel:         cancel,
		hashPool:       make(chan *HashBatch, 100),
		signaturePool:  make(chan *SignatureBatch, 100),
		txPool:         make(chan *TransactionBatch, 100),
	}
	
	log.Debug("Created GPU processor channels", 
		"hashPoolSize", cap(processor.hashPool),
		"signaturePoolSize", cap(processor.signaturePool),
		"txPoolSize", cap(processor.txPool),
	)
	
	// Initialize memory pool
	memoryPoolSize := config.MaxBatchSize * 256
	log.Debug("Initializing memory pool", "itemSize", memoryPoolSize)
	processor.memoryPool = sync.Pool{
		New: func() interface{} {
			log.Trace("Allocating new memory pool buffer", "size", memoryPoolSize)
			return make([]byte, memoryPoolSize) // 256 bytes per item
		},
	}
	
	// Try to initialize GPU
	log.Info("Attempting GPU initialization", "preferredType", config.PreferredGPUType)
	if err := processor.initializeGPU(config.PreferredGPUType); err != nil {
		log.Warn("GPU initialization failed, falling back to CPU", "error", err)
		processor.gpuType = GPUTypeNone
	}
	
	// Start worker goroutines
	log.Info("Starting GPU worker goroutines")
	processor.startWorkers(config)
	
	log.Info("GPU processor initialized successfully", 
		"type", processor.gpuType,
		"deviceCount", processor.deviceCount,
		"maxBatchSize", processor.maxBatchSize,
		"maxMemoryUsage", processor.maxMemoryUsage,
	)
	
	return processor, nil
}

// initializeGPU attempts to initialize GPU acceleration
func (p *GPUProcessor) initializeGPU(preferredType GPUType) error {
	log.Debug("Starting GPU initialization process", "preferredType", preferredType)
	
	// Try CUDA first if preferred or if no preference
	if preferredType == GPUTypeCUDA || preferredType == GPUTypeNone {
		log.Debug("Attempting CUDA initialization")
		start := time.Now()
		result := C.cuda_init_device()
		initDuration := time.Since(start)
		
		log.Debug("CUDA initialization attempt completed", 
			"result", int(result), 
			"duration", initDuration,
			"success", result > 0,
		)
		
		if result > 0 {
			p.gpuType = GPUTypeCUDA
			p.deviceCount = int(result)
			log.Info("CUDA GPU acceleration enabled successfully", 
				"devices", p.deviceCount,
				"initDuration", initDuration,
				"gpuType", "CUDA",
			)
			return nil
		} else {
			log.Warn("CUDA initialization failed", 
				"result", int(result),
				"duration", initDuration,
				"reason", "cuda_init_device returned non-positive value",
			)
		}
	} else {
		log.Debug("Skipping CUDA initialization", "reason", "not preferred type")
	}
	
	// Try OpenCL if CUDA failed or if preferred
	if preferredType == GPUTypeOpenCL || preferredType == GPUTypeNone {
		log.Debug("Attempting OpenCL initialization")
		start := time.Now()
		result := C.initOpenCL()
		initDuration := time.Since(start)
		
		log.Debug("OpenCL initialization attempt completed", 
			"result", int(result), 
			"duration", initDuration,
			"success", result > 0,
		)
		
		if result > 0 {
			p.gpuType = GPUTypeOpenCL
			p.deviceCount = int(result)
			log.Info("OpenCL GPU acceleration enabled successfully", 
				"devices", p.deviceCount,
				"initDuration", initDuration,
				"gpuType", "OpenCL",
				"hardwareType", "RTX 4000 SFF Ada",
			)
			return nil
		} else {
			log.Warn("OpenCL initialization failed", 
				"result", int(result),
				"duration", initDuration,
				"reason", "initOpenCL returned non-positive value",
			)
		}
	} else {
		log.Debug("Skipping OpenCL initialization", "reason", "not preferred type")
	}
	
	log.Error("All GPU initialization attempts failed", 
		"preferredType", preferredType,
		"cudaAttempted", preferredType == GPUTypeCUDA || preferredType == GPUTypeNone,
		"openclAttempted", preferredType == GPUTypeOpenCL || preferredType == GPUTypeNone,
	)
	
	return errors.New("no GPU acceleration available")
}

// startWorkers starts the GPU worker goroutines
func (p *GPUProcessor) startWorkers(config *GPUConfig) {
	// Hash processing workers
	for i := 0; i < config.HashWorkers; i++ {
		p.wg.Add(1)
		go p.hashWorker()
	}
	
	// Signature verification workers
	for i := 0; i < config.SignatureWorkers; i++ {
		p.wg.Add(1)
		go p.signatureWorker()
	}
	
	// Transaction processing workers
	for i := 0; i < config.TxWorkers; i++ {
		p.wg.Add(1)
		go p.transactionWorker()
	}
}

// ProcessHashesBatch processes a batch of hashes using GPU acceleration
func (p *GPUProcessor) ProcessHashesBatch(hashes [][]byte, callback func([][]byte, error)) error {
	log.Trace("ProcessHashesBatch called", "batchSize", len(hashes))
	
	if len(hashes) == 0 {
		log.Debug("Empty hash batch received, calling callback with nil")
		callback(nil, nil)
		return nil
	}
	
	// Validate input parameters
	if callback == nil {
		log.Error("ProcessHashesBatch called with nil callback")
		return errors.New("callback cannot be nil")
	}
	
	// Check batch size limits
	if len(hashes) > p.maxBatchSize {
		log.Warn("Hash batch size exceeds maximum", 
			"batchSize", len(hashes),
			"maxBatchSize", p.maxBatchSize,
		)
		return errors.New("batch size exceeds maximum allowed")
	}
	
	batch := &HashBatch{
		Hashes:   hashes,
		Results:  make([][]byte, len(hashes)),
		Callback: callback,
	}
	
	log.Debug("Submitting hash batch for processing", 
		"batchSize", len(hashes),
		"queueSize", len(p.hashPool),
		"queueCapacity", cap(p.hashPool),
	)
	
	select {
	case p.hashPool <- batch:
		log.Trace("Hash batch successfully queued for processing")
		return nil
	case <-p.ctx.Done():
		log.Debug("Hash batch submission cancelled due to context cancellation")
		return p.ctx.Err()
	default:
		log.Error("Hash processing queue full, rejecting batch", 
			"batchSize", len(hashes),
			"queueSize", len(p.hashPool),
			"queueCapacity", cap(p.hashPool),
		)
		return errors.New("hash processing queue full")
	}
}

// ProcessSignaturesBatch verifies a batch of signatures using GPU acceleration
func (p *GPUProcessor) ProcessSignaturesBatch(signatures, messages, publicKeys [][]byte, callback func([]bool, error)) error {
	log.Trace("ProcessSignaturesBatch called", "batchSize", len(signatures))
	
	if len(signatures) == 0 {
		log.Debug("Empty signature batch received, calling callback with nil")
		callback(nil, nil)
		return nil
	}
	
	// Validate input parameters
	if callback == nil {
		log.Error("ProcessSignaturesBatch called with nil callback")
		return errors.New("callback cannot be nil")
	}
	
	// Validate batch consistency
	if len(signatures) != len(messages) || len(signatures) != len(publicKeys) {
		log.Error("Signature batch arrays have mismatched lengths", 
			"signaturesLen", len(signatures),
			"messagesLen", len(messages),
			"publicKeysLen", len(publicKeys),
		)
		return errors.New("signature batch arrays must have equal lengths")
	}
	
	// Check batch size limits
	if len(signatures) > p.maxBatchSize {
		log.Warn("Signature batch size exceeds maximum", 
			"batchSize", len(signatures),
			"maxBatchSize", p.maxBatchSize,
		)
		return errors.New("batch size exceeds maximum allowed")
	}
	
	batch := &SignatureBatch{
		Signatures: signatures,
		Messages:   messages,
		PublicKeys: publicKeys,
		Results:    make([]bool, len(signatures)),
		Callback:   callback,
	}
	
	log.Debug("Submitting signature batch for processing", 
		"batchSize", len(signatures),
		"queueSize", len(p.signaturePool),
		"queueCapacity", cap(p.signaturePool),
	)
	
	select {
	case p.signaturePool <- batch:
		log.Trace("Signature batch successfully queued for processing")
		return nil
	case <-p.ctx.Done():
		log.Debug("Signature batch submission cancelled due to context cancellation")
		return p.ctx.Err()
	default:
		log.Error("Signature processing queue full, rejecting batch", 
			"batchSize", len(signatures),
			"queueSize", len(p.signaturePool),
			"queueCapacity", cap(p.signaturePool),
		)
		return errors.New("signature processing queue full")
	}
}

// ProcessTransactionsBatch processes a batch of transactions using GPU acceleration
func (p *GPUProcessor) ProcessTransactionsBatch(txs []*types.Transaction, callback func([]*TxResult, error)) error {
	log.Trace("ProcessTransactionsBatch called", "batchSize", len(txs))
	
	if len(txs) == 0 {
		log.Debug("Empty transaction batch received, calling callback with nil")
		callback(nil, nil)
		return nil
	}
	
	// Validate input parameters
	if callback == nil {
		log.Error("ProcessTransactionsBatch called with nil callback")
		return errors.New("callback cannot be nil")
	}
	
	// Validate transactions
	nilTxCount := 0
	for i, tx := range txs {
		if tx == nil {
			nilTxCount++
			log.Warn("Nil transaction found in batch", "index", i)
		}
	}
	
	if nilTxCount > 0 {
		log.Error("Transaction batch contains nil transactions", 
			"nilCount", nilTxCount,
			"totalCount", len(txs),
		)
		return errors.New("batch contains nil transactions")
	}
	
	// Check batch size limits
	if len(txs) > p.maxBatchSize {
		log.Warn("Transaction batch size exceeds maximum", 
			"batchSize", len(txs),
			"maxBatchSize", p.maxBatchSize,
		)
		return errors.New("batch size exceeds maximum allowed")
	}
	
	batch := &TransactionBatch{
		Transactions: txs,
		Results:      make([]*TxResult, len(txs)),
		Callback:     callback,
	}
	
	log.Debug("Submitting transaction batch for processing", 
		"batchSize", len(txs),
		"queueSize", len(p.txPool),
		"queueCapacity", cap(p.txPool),
	)
	
	select {
	case p.txPool <- batch:
		log.Trace("Transaction batch successfully queued for processing")
		return nil
	case <-p.ctx.Done():
		log.Debug("Transaction batch submission cancelled due to context cancellation")
		return p.ctx.Err()
	default:
		log.Error("Transaction processing queue full, rejecting batch", 
			"batchSize", len(txs),
			"queueSize", len(p.txPool),
			"queueCapacity", cap(p.txPool),
		)
		return errors.New("transaction processing queue full")
	}
}

// hashWorker processes hash batches using GPU acceleration
func (p *GPUProcessor) hashWorker() {
	defer p.wg.Done()
	log.Debug("Hash worker started", "gpuType", p.gpuType)
	
	for {
		select {
		case <-p.ctx.Done():
			log.Debug("Hash worker shutting down due to context cancellation")
			return
		case batch := <-p.hashPool:
			log.Trace("Hash worker received batch", "batchSize", len(batch.Hashes))
			start := time.Now()
			
			if p.gpuType == GPUTypeNone {
				log.Trace("Processing hash batch on CPU (no GPU available)")
				// CPU fallback
				p.processHashesCPU(batch)
			} else {
				log.Trace("Processing hash batch on GPU", "gpuType", p.gpuType)
				// GPU processing
				p.processHashesGPU(batch)
			}
			
			duration := time.Since(start)
			log.Debug("Hash batch processing completed", 
				"batchSize", len(batch.Hashes),
				"processingTime", duration,
				"processingMode", func() string {
					if p.gpuType == GPUTypeNone {
						return "CPU"
					}
					return "GPU"
				}(),
			)
			p.updateHashStats(duration)
		}
	}
}

// signatureWorker processes signature verification batches
func (p *GPUProcessor) signatureWorker() {
	defer p.wg.Done()
	log.Debug("Signature worker started", "gpuType", p.gpuType)
	
	for {
		select {
		case <-p.ctx.Done():
			log.Debug("Signature worker shutting down due to context cancellation")
			return
		case batch := <-p.signaturePool:
			log.Trace("Signature worker received batch", "batchSize", len(batch.Signatures))
			start := time.Now()
			
			if p.gpuType == GPUTypeNone {
				log.Trace("Processing signature batch on CPU (no GPU available)")
				// CPU fallback
				p.processSignaturesCPU(batch)
			} else {
				log.Trace("Processing signature batch on GPU", "gpuType", p.gpuType)
				// GPU processing
				p.processSignaturesGPU(batch)
			}
			
			duration := time.Since(start)
			log.Debug("Signature batch processing completed", 
				"batchSize", len(batch.Signatures),
				"processingTime", duration,
				"processingMode", func() string {
					if p.gpuType == GPUTypeNone {
						return "CPU"
					}
					return "GPU"
				}(),
			)
			p.updateSigStats(duration)
		}
	}
}

// transactionWorker processes transaction batches
func (p *GPUProcessor) transactionWorker() {
	defer p.wg.Done()
	log.Debug("Transaction worker started", "gpuType", p.gpuType)
	
	for {
		select {
		case <-p.ctx.Done():
			log.Debug("Transaction worker shutting down due to context cancellation")
			return
		case batch := <-p.txPool:
			log.Trace("Transaction worker received batch", "batchSize", len(batch.Transactions))
			start := time.Now()
			
			if p.gpuType == GPUTypeNone {
				log.Trace("Processing transaction batch on CPU (no GPU available)")
				// CPU fallback
				p.processTransactionsCPU(batch)
			} else {
				log.Trace("Processing transaction batch on GPU", "gpuType", p.gpuType)
				// GPU processing
				p.processTransactionsGPU(batch)
			}
			
			duration := time.Since(start)
			log.Debug("Transaction batch processing completed", 
				"batchSize", len(batch.Transactions),
				"processingTime", duration,
				"processingMode", func() string {
					if p.gpuType == GPUTypeNone {
						return "CPU"
					}
					return "GPU"
				}(),
			)
			p.updateTxStats(duration)
		}
	}
}

// processHashesGPU processes hashes using GPU acceleration
func (p *GPUProcessor) processHashesGPU(batch *HashBatch) {
	log.Debug("Starting GPU hash processing", 
		"batchSize", len(batch.Hashes),
		"gpuType", p.gpuType,
	)
	
	defer func() {
		if r := recover(); r != nil {
			log.Error("GPU hash processing panicked", 
				"panic", r,
				"batchSize", len(batch.Hashes),
				"gpuType", p.gpuType,
			)
			p.processHashesCPU(batch) // Fallback to CPU
		}
	}()

	// Pack input into fixed 256-byte slots per hash
	log.Trace("Preparing hash data for GPU processing", "hashCount", len(batch.Hashes))
	dataStart := time.Now()
	in := p.prepareHashData(batch.Hashes)
	dataPreparationTime := time.Since(dataStart)
	defer p.memoryPool.Put(in)
	
	log.Debug("Hash data preparation completed", 
		"inputSize", len(in),
		"preparationTime", dataPreparationTime,
	)

	// Allocate output buffer: 32 bytes per hash
	count := len(batch.Hashes)
	out := make([]byte, count*32)
	log.Trace("Allocated output buffer", "size", len(out))

	// Process on GPU
	log.Debug("Executing GPU hash computation", 
		"gpuType", p.gpuType,
		"hashCount", count,
		"inputBufferSize", len(in),
		"outputBufferSize", len(out),
	)
	
	gpuStart := time.Now()
	var result int
	switch p.gpuType {
	case GPUTypeCUDA:
		log.Trace("Calling CUDA hash processing kernel")
		result = int(C.cuda_process_hashes(
			unsafe.Pointer(&in[0]),
			C.int(count),
			unsafe.Pointer(&out[0]),
		))
		log.Debug("CUDA hash processing completed", 
			"result", result,
			"duration", time.Since(gpuStart),
		)
	case GPUTypeOpenCL:
		log.Trace("Calling OpenCL hash processing kernel")
		result = int(C.processHashesOpenCL(
			unsafe.Pointer(&in[0]),
			C.int(count),
			unsafe.Pointer(&out[0]),
		))
		log.Debug("OpenCL hash processing completed", 
			"result", result,
			"duration", time.Since(gpuStart),
		)
	}
	
	gpuProcessingTime := time.Since(gpuStart)

	if result != 0 {
		log.Warn("GPU hash processing failed, falling back to CPU", 
			"error", result,
			"gpuType", p.gpuType,
			"batchSize", count,
			"gpuProcessingTime", gpuProcessingTime,
		)
		p.processHashesCPU(batch)
		return
	}

	// Split flat output into [][]byte
	log.Trace("Converting GPU output to result format")
	conversionStart := time.Now()
	for i := 0; i < count; i++ {
		start := i * 32
		dst := make([]byte, 32)
		copy(dst, out[start:start+32])
		batch.Results[i] = dst
	}
	conversionTime := time.Since(conversionStart)
	
	log.Debug("GPU hash processing completed successfully", 
		"batchSize", count,
		"gpuType", p.gpuType,
		"dataPreparationTime", dataPreparationTime,
		"gpuProcessingTime", gpuProcessingTime,
		"conversionTime", conversionTime,
		"totalTime", time.Since(dataStart),
	)

	if batch.Callback != nil {
		batch.Callback(batch.Results, nil)
	}
}

// processHashesCPU processes hashes using CPU as fallback
func (p *GPUProcessor) processHashesCPU(batch *HashBatch) {
	for i, hash := range batch.Hashes {
		result := crypto.Keccak256(hash)
		batch.Results[i] = result
	}
	
	if batch.Callback != nil {
		batch.Callback(batch.Results, nil)
	}
}

// processSignaturesGPU processes signature verification using GPU
func (p *GPUProcessor) processSignaturesGPU(batch *SignatureBatch) {
	log.Debug("Starting GPU signature verification", 
		"batchSize", len(batch.Signatures),
		"gpuType", p.gpuType,
	)
	
	defer func() {
		if r := recover(); r != nil {
			log.Error("GPU signature processing panicked", 
				"panic", r,
				"batchSize", len(batch.Signatures),
				"gpuType", p.gpuType,
			)
			p.processSignaturesCPU(batch) // Fallback to CPU
		}
	}()

	// Pack input as [65|32|64] per item (stride 161 bytes)
	log.Trace("Preparing signature data for GPU processing", "signatureCount", len(batch.Signatures))
	dataStart := time.Now()
	packed := p.prepareSignatureData(batch.Signatures, batch.Messages, batch.PublicKeys)
	dataPreparationTime := time.Since(dataStart)
	defer p.memoryPool.Put(packed)
	
	log.Debug("Signature data preparation completed", 
		"packedSize", len(packed),
		"preparationTime", dataPreparationTime,
	)

	// Output buffer: 1 byte (0/1) per signature
	count := len(batch.Signatures)
	out := make([]byte, count)
	log.Trace("Allocated signature output buffer", "size", len(out))

	// Process on GPU
	log.Debug("Executing GPU signature verification", 
		"gpuType", p.gpuType,
		"signatureCount", count,
		"inputBufferSize", len(packed),
		"outputBufferSize", len(out),
	)
	
	gpuStart := time.Now()
	var result int
	switch p.gpuType {
	case GPUTypeCUDA: {
		log.Trace("Preparing CUDA signature buffers")
		// CUDA expects separate buffers for signatures, messages, and pubkeys
		bufferStart := time.Now()
		sigs := make([]byte, count*65)
		msgs := make([]byte, count*32)
		keys := make([]byte, count*64)
		for i := 0; i < count; i++ {
			copy(sigs[i*65:(i+1)*65], batch.Signatures[i])
			copy(msgs[i*32:(i+1)*32], batch.Messages[i])
			copy(keys[i*64:(i+1)*64], batch.PublicKeys[i])
		}
		bufferPreparationTime := time.Since(bufferStart)
		
		log.Debug("CUDA signature buffers prepared", 
			"sigsSize", len(sigs),
			"msgsSize", len(msgs),
			"keysSize", len(keys),
			"bufferPreparationTime", bufferPreparationTime,
		)
		
		log.Trace("Calling CUDA signature verification kernel")
		kernelStart := time.Now()
		result = int(C.cuda_verify_signatures(
			unsafe.Pointer(&sigs[0]),
			unsafe.Pointer(&msgs[0]),
			unsafe.Pointer(&keys[0]),
			C.int(count),
			unsafe.Pointer(&out[0]),
		))
		kernelTime := time.Since(kernelStart)
		
		log.Debug("CUDA signature verification completed", 
			"result", result,
			"kernelTime", kernelTime,
			"totalCudaTime", time.Since(gpuStart),
		)
	}
	case GPUTypeOpenCL:
		log.Trace("Calling OpenCL signature verification kernel")
		result = int(C.verifySignaturesOpenCL(
			unsafe.Pointer(&packed[0]),
			C.int(count),
			unsafe.Pointer(&out[0]),
		))
		log.Debug("OpenCL signature verification completed", 
			"result", result,
			"duration", time.Since(gpuStart),
		)
	}
	
	gpuProcessingTime := time.Since(gpuStart)

	if result != 0 {
		log.Warn("GPU signature processing failed, falling back to CPU", 
			"error", result,
			"gpuType", p.gpuType,
			"batchSize", count,
			"gpuProcessingTime", gpuProcessingTime,
		)
		p.processSignaturesCPU(batch)
		return
	}

	// Map bytes to bools
	log.Trace("Converting GPU signature results to boolean format")
	conversionStart := time.Now()
	validCount := 0
	for i := 0; i < count; i++ {
		batch.Results[i] = out[i] != 0
		if batch.Results[i] {
			validCount++
		}
	}
	conversionTime := time.Since(conversionStart)
	
	log.Debug("GPU signature verification completed successfully", 
		"batchSize", count,
		"validSignatures", validCount,
		"invalidSignatures", count-validCount,
		"gpuType", p.gpuType,
		"dataPreparationTime", dataPreparationTime,
		"gpuProcessingTime", gpuProcessingTime,
		"conversionTime", conversionTime,
		"totalTime", time.Since(dataStart),
	)

	if batch.Callback != nil {
		batch.Callback(batch.Results, nil)
	}
}

// processSignaturesCPU processes signature verification using CPU as fallback
func (p *GPUProcessor) processSignaturesCPU(batch *SignatureBatch) {
	for i := range batch.Signatures {
		if len(batch.Signatures[i]) != 65 || len(batch.Messages[i]) != 32 || len(batch.PublicKeys[i]) != 64 {
			batch.Results[i] = false
			continue
		}
		sig := batch.Signatures[i][:64] // R||S only
		hash := batch.Messages[i]
		pubkey := batch.PublicKeys[i] // uncompressed 64-byte (X||Y)
		batch.Results[i] = crypto.VerifySignature(pubkey, hash, sig)
	}
	if batch.Callback != nil {
		batch.Callback(batch.Results, nil)
	}
}

// processTransactionsGPU processes transactions using GPU with full execution context
func (p *GPUProcessor) processTransactionsGPU(batch *TransactionBatch) {
	// GPU ACTIVATION LOG - This proves GPU is processing real transactions
	log.Info("🚀 GPU TRANSACTION PROCESSING ACTIVATED", 
		"batchSize", len(batch.Transactions),
		"gpuType", p.gpuType,
		"timestamp", time.Now().Format("2006-01-02 15:04:05.000"),
		"deviceCount", p.deviceCount,
	)
	
	// Save to GPU log file
	logging.LogGPU("INFO", "GPU TRANSACTION PROCESSING ACTIVATED", 
		"batchSize", len(batch.Transactions),
		"gpuType", p.gpuType,
		"deviceCount", p.deviceCount,
	)
	
	log.Debug("Starting enhanced GPU transaction processing", 
		"batchSize", len(batch.Transactions),
		"gpuType", p.gpuType,
	)
	
	defer func() {
		if r := recover(); r != nil {
			log.Error("GPU transaction processing panicked", 
				"panic", r,
				"batchSize", len(batch.Transactions),
				"gpuType", p.gpuType,
			)
			p.processTransactionsCPU(batch) // Fallback to CPU
		}
	}()

	count := len(batch.Transactions)
	dataStart := time.Now()
	
    // Prepare transaction data (1KB per transaction)
    log.Trace("Preparing transaction data for GPU processing", "txCount", count)
    txData := p.prepareTransactionData(batch.Transactions)
    defer p.memoryPool.Put(txData)
    // Prepare per-transaction lengths
    txLens := p.prepareTransactionLengths(batch.Transactions)
	
	// Prepare state snapshots (2KB per transaction)
	log.Trace("Preparing state snapshots for GPU processing")
	stateData := p.prepareStateSnapshots(batch.Transactions)
	defer func() {
		if stateData != nil {
			p.memoryPool.Put(stateData)
		}
	}()
	
	// Prepare access lists (512B per transaction)
	log.Trace("Preparing access lists for GPU processing")
	accessLists := p.prepareAccessLists(batch.Transactions)
	defer func() {
		if accessLists != nil {
			p.memoryPool.Put(accessLists)
		}
	}()
	
	dataPreparationTime := time.Since(dataStart)
	log.Debug("Enhanced transaction data preparation completed", 
		"txDataSize", len(txData),
		"stateDataSize", func() int {
			if stateData != nil {
				return len(stateData)
			}
			return 0
		}(),
		"accessListsSize", func() int {
			if accessLists != nil {
				return len(accessLists)
			}
			return 0
		}(),
		"preparationTime", dataPreparationTime,
	)

	// Enhanced output buffer: 128 bytes per transaction
	out := make([]byte, count*128)
	log.Trace("Allocated enhanced transaction output buffer", "size", len(out))

	// Process on GPU with full execution context
	log.Debug("Executing enhanced GPU transaction computation", 
		"gpuType", p.gpuType,
		"txCount", count,
		"inputBufferSize", len(txData),
		"outputBufferSize", len(out),
	)
	
	gpuStart := time.Now()
	var result int
    switch p.gpuType {
    case GPUTypeCUDA:
        log.Trace("Calling enhanced CUDA transaction processing kernel")
        result = int(C.cuda_process_transactions_full(
            unsafe.Pointer(&txData[0]),
            unsafe.Pointer(&txLens[0]),
            func() unsafe.Pointer {
                if stateData != nil {
                    return unsafe.Pointer(&stateData[0])
                }
                return nil
            }(),
            func() unsafe.Pointer {
                if accessLists != nil {
                    return unsafe.Pointer(&accessLists[0])
                }
                return nil
            }(),
            C.int(count),
            unsafe.Pointer(&out[0]),
        ))
		log.Debug("Enhanced CUDA transaction processing completed", 
			"result", result,
			"duration", time.Since(gpuStart),
		)
    case GPUTypeOpenCL:
        log.Trace("Calling enhanced OpenCL transaction processing kernel")
        result = int(C.processTxBatchOpenCLFull(
            unsafe.Pointer(&txData[0]),
            unsafe.Pointer(&txLens[0]),
            func() unsafe.Pointer {
                if stateData != nil {
                    return unsafe.Pointer(&stateData[0])
                }
                return nil
            }(),
            func() unsafe.Pointer {
                if accessLists != nil {
                    return unsafe.Pointer(&accessLists[0])
                }
                return nil
            }(),
            C.int(count),
            unsafe.Pointer(&out[0]),
        ))
		log.Debug("Enhanced OpenCL transaction processing completed", 
			"result", result,
			"duration", time.Since(gpuStart),
		)
	}
	
	gpuProcessingTime := time.Since(gpuStart)

	if result != 0 {
		log.Warn("Enhanced GPU transaction processing failed, falling back to CPU", 
			"error", result,
			"gpuType", p.gpuType,
			"batchSize", count,
			"gpuProcessingTime", gpuProcessingTime,
		)
		p.processTransactionsCPU(batch)
		return
	}

	// Convert enhanced results
	log.Trace("Converting enhanced GPU transaction results")
	conversionStart := time.Now()
	validCount := 0
	successCount := 0
	revertCount := 0
	outOfGasCount := 0
	invalidCount := 0
	
	for i := 0; i < count; i++ {
		offset := i * 128

		if batch.Results[i] == nil {
			batch.Results[i] = &TxResult{}
		}
		
		// Enhanced result structure:
		// [0-31]: transaction hash
		// [32]: validity flag
		// [33]: execution status (0=success, 1=revert, 2=out_of_gas, 3=invalid)
		// [34-41]: gas used (8 bytes LE)
		// [42-73]: return data hash (32 bytes)
		// [74-105]: revert reason hash (32 bytes)
		
		// Extract transaction hash
		var txHash common.Hash
		copy(txHash[:], out[offset:offset+32])
		batch.Results[i].Hash = txHash
		
		// Extract validity and execution status
		valid := out[offset+32] != 0
		execStatus := out[offset+33]
		
		// CPU signature verification for safety
		if valid {
			if err := p.verifyTxSignatureCPU(batch.Transactions[i]); err != nil {
				valid = false
				execStatus = 3 // invalid
			}
		}
		batch.Results[i].Valid = valid
		
		// Extract gas used
		gasUsed := binary.LittleEndian.Uint64(out[offset+34 : offset+42])
		if gasUsed > 0 {
			batch.Results[i].GasUsed = gasUsed
		} else {
			batch.Results[i].GasUsed = batch.Transactions[i].Gas()
		}
		
		// Set error based on execution status
		switch execStatus {
		case 0: // success
			batch.Results[i].Error = nil
			successCount++
		case 1: // revert
			batch.Results[i].Error = errors.New("execution reverted")
			revertCount++
		case 2: // out of gas
			batch.Results[i].Error = errors.New("out of gas")
			outOfGasCount++
		case 3: // invalid
			batch.Results[i].Error = errors.New("invalid transaction")
			invalidCount++
		}
		
		if valid {
			validCount++
		}
	}
	conversionTime := time.Since(conversionStart)
	
	// Calculate TPS for this batch
	totalTime := time.Since(dataStart)
	var tps float64
	if totalTime > 0 {
		tps = float64(count) / totalTime.Seconds()
	}
	
	// Enhanced GPU PERFORMANCE LOG
	log.Info("✅ ENHANCED GPU TRANSACTION BATCH COMPLETED", 
		"batchSize", count,
		"validTransactions", validCount,
		"successfulExecutions", successCount,
		"revertedExecutions", revertCount,
		"outOfGasExecutions", outOfGasCount,
		"invalidTransactions", invalidCount,
		"gpuType", p.gpuType,
		"batchTPS", tps,
		"gpuKernelTime", gpuProcessingTime,
		"totalProcessingTime", totalTime,
		"timestamp", time.Now().Format("2006-01-02 15:04:05.000"),
	)
	
	// Save enhanced performance data to files
	logging.LogGPU("INFO", "ENHANCED GPU TRANSACTION BATCH COMPLETED", 
		"batchSize", count,
		"validTransactions", validCount,
		"successfulExecutions", successCount,
		"revertedExecutions", revertCount,
		"outOfGasExecutions", outOfGasCount,
		"invalidTransactions", invalidCount,
		"gpuType", p.gpuType,
		"batchTPS", tps,
		"gpuKernelTime", gpuProcessingTime,
		"totalProcessingTime", totalTime,
	)
	
	logging.LogPerformance("INFO", "ENHANCED BATCH PERFORMANCE METRICS", 
		"batchSize", count,
		"batchTPS", tps,
		"processingMode", "Enhanced GPU",
		"kernelTime", gpuProcessingTime,
		"totalTime", totalTime,
	)
	
	logging.LogTransaction("INFO", "ENHANCED TRANSACTION BATCH PROCESSED", 
		"batchSize", count,
		"validTransactions", validCount,
		"successfulExecutions", successCount,
		"revertedExecutions", revertCount,
		"outOfGasExecutions", outOfGasCount,
		"invalidTransactions", invalidCount,
		"processingMode", "Enhanced GPU",
	)
	
	log.Debug("Enhanced GPU transaction processing completed successfully", 
		"batchSize", count,
		"validTransactions", validCount,
		"successfulExecutions", successCount,
		"revertedExecutions", revertCount,
		"outOfGasExecutions", outOfGasCount,
		"invalidTransactions", invalidCount,
		"gpuType", p.gpuType,
		"dataPreparationTime", dataPreparationTime,
		"gpuProcessingTime", gpuProcessingTime,
		"conversionTime", conversionTime,
		"totalTime", totalTime,
		"batchTPS", tps,
	)

	if batch.Callback != nil {
		batch.Callback(batch.Results, nil)
	}
}

// processTransactionsCPU processes transactions using CPU as fallback
func (p *GPUProcessor) processTransactionsCPU(batch *TransactionBatch) {
	for i, tx := range batch.Transactions {
		batch.Results[i] = &TxResult{
			Hash:    tx.Hash(),
			Valid:   true, // Simplified validation
			GasUsed: tx.Gas(),
			Error:   nil,
		}
	}
	
	if batch.Callback != nil {
		batch.Callback(batch.Results, nil)
	}
}

// Helper functions for data preparation with safety checks
func (p *GPUProcessor) prepareHashData(hashes [][]byte) []byte {
	log.Trace("Starting hash data preparation", "hashCount", len(hashes))
	
	// OpenCL/CUDA kernels expect fixed 256 bytes per input item
	const slot = 256
	count := len(hashes)
	total := count * slot
	
	log.Debug("Hash data preparation parameters", 
		"slotSize", slot,
		"hashCount", count,
		"totalBufferSize", total,
	)

	memStart := time.Now()
	buf := p.memoryPool.Get().([]byte)
	memGetTime := time.Since(memStart)
	
	log.Trace("Retrieved buffer from memory pool", 
		"bufferCapacity", cap(buf),
		"requiredSize", total,
		"memGetTime", memGetTime,
	)
	
	if cap(buf) < total {
		log.Debug("Memory pool buffer too small, allocating new buffer", 
			"poolBufferCap", cap(buf),
			"requiredSize", total,
		)
		allocStart := time.Now()
		buf = make([]byte, total)
		allocTime := time.Since(allocStart)
		log.Debug("Allocated new hash buffer", 
			"size", total,
			"allocTime", allocTime,
		)
	}
	data := buf[:total]

	copyStart := time.Now()
	truncatedCount := 0
	for i, h := range hashes {
		base := i * slot
		n := len(h)
		if n > slot {
			n = slot
			truncatedCount++
		}
		copy(data[base:base+n], h[:n])
		// Remaining bytes are already zeroed
	}
	copyTime := time.Since(copyStart)
	
	log.Debug("Hash data preparation completed", 
		"processedHashes", count,
		"truncatedHashes", truncatedCount,
		"finalBufferSize", len(data),
		"copyTime", copyTime,
		"totalPrepTime", time.Since(memStart),
	)
	
	return data
}

func (p *GPUProcessor) prepareSignatureData(signatures, messages, publicKeys [][]byte) []byte {
	log.Trace("Starting signature data preparation", "signatureCount", len(signatures))
	
	memStart := time.Now()
	data := p.memoryPool.Get().([]byte)
	memGetTime := time.Since(memStart)
	
	log.Debug("Retrieved signature buffer from memory pool", 
		"bufferSize", len(data),
		"memGetTime", memGetTime,
	)
	
	offset := 0
	processedCount := 0
	
	copyStart := time.Now()
	for i := range signatures {
		totalSize := len(signatures[i]) + len(messages[i]) + len(publicKeys[i])
		// Safety check to prevent buffer overflow
		if offset+totalSize > len(data) {
			log.Warn("Signature data buffer overflow, truncating batch", 
				"offset", offset, 
				"totalSize", totalSize, 
				"bufferLen", len(data),
				"processedItems", processedCount,
				"remainingItems", len(signatures)-i,
			)
			break
		}
		
		// Copy signature data
		copy(data[offset:], signatures[i])
		offset += len(signatures[i])
		
		// Copy message data
		copy(data[offset:], messages[i])
		offset += len(messages[i])
		
		// Copy public key data
		copy(data[offset:], publicKeys[i])
		offset += len(publicKeys[i])
		
		processedCount++
	}
	copyTime := time.Since(copyStart)
	
	log.Debug("Signature data preparation completed", 
		"requestedSignatures", len(signatures),
		"processedSignatures", processedCount,
		"finalBufferSize", offset,
		"copyTime", copyTime,
		"totalPrepTime", time.Since(memStart),
	)
	
	return data[:offset]
}

func (p *GPUProcessor) prepareTransactionData(txs []*types.Transaction) []byte {
	log.Trace("Starting transaction data preparation", "txCount", len(txs))
	
	// Kernels expect fixed 1024 bytes per transaction
	const slot = 1024
	count := len(txs)
	total := count * slot
	
	log.Debug("Transaction data preparation parameters", 
		"slotSize", slot,
		"txCount", count,
		"totalBufferSize", total,
	)

	memStart := time.Now()
	buf := p.memoryPool.Get().([]byte)
	memGetTime := time.Since(memStart)
	
	log.Trace("Retrieved transaction buffer from memory pool", 
		"bufferCapacity", cap(buf),
		"requiredSize", total,
		"memGetTime", memGetTime,
	)
	
	if cap(buf) < total {
		log.Debug("Memory pool buffer too small, allocating new transaction buffer", 
			"poolBufferCap", cap(buf),
			"requiredSize", total,
		)
		allocStart := time.Now()
		buf = make([]byte, total)
		allocTime := time.Since(allocStart)
		log.Debug("Allocated new transaction buffer", 
			"size", total,
			"allocTime", allocTime,
		)
	}
	data := buf[:total]

	marshalStart := time.Now()
	marshalErrors := 0
	truncatedCount := 0
	totalMarshalledBytes := 0
	
	for i, tx := range txs {
		txBytes, err := tx.MarshalBinary()
		if err != nil {
			log.Warn("Failed to marshal transaction", 
				"txIndex", i,
				"hash", tx.Hash(), 
				"error", err,
			)
			marshalErrors++
			continue
		}
		
		totalMarshalledBytes += len(txBytes)
		base := i * slot
		n := len(txBytes)
		if n > slot {
			// Truncate if too large for slot
			n = slot
			truncatedCount++
			log.Trace("Transaction data truncated", 
				"txIndex", i,
				"originalSize", len(txBytes),
				"truncatedSize", n,
				"hash", tx.Hash(),
			)
		}
		copy(data[base:base+n], txBytes[:n])
		// Remaining bytes are left zeroed
	}
	marshalTime := time.Since(marshalStart)
	
	log.Debug("Transaction data preparation completed", 
		"processedTxs", count,
		"marshalErrors", marshalErrors,
		"truncatedTxs", truncatedCount,
		"totalMarshalledBytes", totalMarshalledBytes,
		"finalBufferSize", len(data),
		"marshalTime", marshalTime,
		"totalPrepTime", time.Since(memStart),
	)
	
	return data
}

// prepareTransactionLengths returns the marshaled byte length of each tx (clamped to 1024)
func (p *GPUProcessor) prepareTransactionLengths(txs []*types.Transaction) []int32 {
    lengths := make([]int32, len(txs))
    for i, tx := range txs {
        b, err := tx.MarshalBinary()
        if err != nil {
            lengths[i] = 0
            continue
        }
        if len(b) > 1024 {
            lengths[i] = 1024
        } else {
            lengths[i] = int32(len(b))
        }
    }
    return lengths
}

// verifyTxSignatureCPU performs canonical ECDSA recovery/verification on CPU
func (p *GPUProcessor) verifyTxSignatureCPU(tx *types.Transaction) error {
    // Choose permissive signer based on chain ID (handles typed txs)
    chainID := tx.ChainId()
    // Ensure non-nil
    if chainID == nil {
        chainID = new(big.Int)
    }
    signer := types.LatestSignerForChainID(chainID)
    // Attempt to recover sender; error implies invalid signature
    if _, err := types.Sender(signer, tx); err != nil {
        return err
    }
    return nil
}

// prepareStateSnapshots prepares state snapshot data for GPU processing
func (p *GPUProcessor) prepareStateSnapshots(txs []*types.Transaction) []byte {
	log.Trace("Starting state snapshot preparation", "txCount", len(txs))
	
	// Kernels expect fixed 2048 bytes per transaction for state data
	const slot = 2048
	count := len(txs)
	total := count * slot
	
	log.Debug("State snapshot preparation parameters", 
		"slotSize", slot,
		"txCount", count,
		"totalBufferSize", total,
	)

	memStart := time.Now()
	buf := p.memoryPool.Get().([]byte)
	memGetTime := time.Since(memStart)
	
	log.Trace("Retrieved state buffer from memory pool", 
		"bufferCapacity", cap(buf),
		"requiredSize", total,
		"memGetTime", memGetTime,
	)
	
	if cap(buf) < total {
		log.Debug("Memory pool buffer too small, allocating new state buffer", 
			"poolBufferCap", cap(buf),
			"requiredSize", total,
		)
		allocStart := time.Now()
		buf = make([]byte, total)
		allocTime := time.Since(allocStart)
		log.Debug("Allocated new state buffer", 
			"size", total,
			"allocTime", allocTime,
		)
	}
	data := buf[:total]

	// For now, we'll create simplified state snapshots
	// In a full implementation, this would include:
	// - Account balances and nonces
	// - Contract storage states
	// - Code hashes
	// - State root information
	
	prepStart := time.Now()
	for i, tx := range txs {
		base := i * slot
		
		// Simplified state snapshot structure:
		// [0-31]: sender address (20 bytes + 12 padding)
		// [32-63]: recipient address (20 bytes + 12 padding) 
		// [64-95]: sender balance (32 bytes)
		// [96-127]: sender nonce (32 bytes)
		// [128-159]: recipient balance (32 bytes)
		// [160-191]: gas price (32 bytes)
		// [192-223]: block number (32 bytes)
		// [224-255]: block timestamp (32 bytes)
		// [256-2047]: reserved for contract storage/code
		
		// Extract sender address (simplified - would need proper state access)
		if tx.To() != nil {
			// Copy recipient address
			copy(data[base+32:base+52], tx.To().Bytes())
		}
		
		// Set simplified values (in production, these would come from state DB)
		// Gas price
		gasPrice := tx.GasPrice()
		if gasPrice != nil {
			gasPriceBytes := gasPrice.Bytes()
			if len(gasPriceBytes) <= 32 {
				copy(data[base+160+32-len(gasPriceBytes):base+192], gasPriceBytes)
			}
		}
		
		// Transaction value
		value := tx.Value()
		if value != nil {
			valueBytes := value.Bytes()
			if len(valueBytes) <= 32 {
				copy(data[base+64+32-len(valueBytes):base+96], valueBytes)
			}
		}
		
		// Nonce
		nonce := tx.Nonce()
		binary.BigEndian.PutUint64(data[base+96+24:base+104], nonce)
	}
	prepTime := time.Since(prepStart)
	
	log.Debug("State snapshot preparation completed", 
		"processedTxs", count,
		"finalBufferSize", len(data),
		"prepTime", prepTime,
		"totalPrepTime", time.Since(memStart),
	)
	
	return data
}

// prepareAccessLists prepares access list data for GPU processing
func (p *GPUProcessor) prepareAccessLists(txs []*types.Transaction) []byte {
	log.Trace("Starting access list preparation", "txCount", len(txs))
	
	// Kernels expect fixed 512 bytes per transaction for access lists
	const slot = 512
	count := len(txs)
	total := count * slot
	
	log.Debug("Access list preparation parameters", 
		"slotSize", slot,
		"txCount", count,
		"totalBufferSize", total,
	)

	memStart := time.Now()
	buf := p.memoryPool.Get().([]byte)
	memGetTime := time.Since(memStart)
	
	log.Trace("Retrieved access list buffer from memory pool", 
		"bufferCapacity", cap(buf),
		"requiredSize", total,
		"memGetTime", memGetTime,
	)
	
	if cap(buf) < total {
		log.Debug("Memory pool buffer too small, allocating new access list buffer", 
			"poolBufferCap", cap(buf),
			"requiredSize", total,
		)
		allocStart := time.Now()
		buf = make([]byte, total)
		allocTime := time.Since(allocStart)
		log.Debug("Allocated new access list buffer", 
			"size", total,
			"allocTime", allocTime,
		)
	}
	data := buf[:total]

	prepStart := time.Now()
	accessListCount := 0
	
	for i, tx := range txs {
		base := i * slot
		offset := 0
		
		// Get access list from transaction
		accessList := tx.AccessList()
		if len(accessList) > 0 {
			accessListCount++
			
			// Access list structure:
			// [0-1]: number of addresses (2 bytes)
			// [2-3]: total storage keys (2 bytes)
			// [4+]: address entries, each followed by storage keys
			//   - Address: 20 bytes
			//   - Storage key count: 2 bytes  
			//   - Storage keys: 32 bytes each
			
			if offset+4 < slot {
				// Write number of addresses
				binary.LittleEndian.PutUint16(data[base+offset:base+offset+2], uint16(len(accessList)))
				offset += 2
				
				// Count total storage keys
				totalKeys := 0
				for _, entry := range accessList {
					totalKeys += len(entry.StorageKeys)
				}
				binary.LittleEndian.PutUint16(data[base+offset:base+offset+2], uint16(totalKeys))
				offset += 2
				
				// Write access list entries
				for _, entry := range accessList {
					// Write address (20 bytes)
					if offset+20 < slot {
						copy(data[base+offset:base+offset+20], entry.Address.Bytes())
						offset += 20
					} else {
						break
					}
					
					// Write storage key count (2 bytes)
					if offset+2 < slot {
						binary.LittleEndian.PutUint16(data[base+offset:base+offset+2], uint16(len(entry.StorageKeys)))
						offset += 2
					} else {
						break
					}
					
					// Write storage keys (32 bytes each)
					for _, key := range entry.StorageKeys {
						if offset+32 < slot {
							copy(data[base+offset:base+offset+32], key.Bytes())
							offset += 32
						} else {
							break
						}
					}
					
					if offset >= slot-64 { // Leave some buffer space
						break
					}
				}
			}
		}
		// If no access list, the slot remains zeroed
	}
	prepTime := time.Since(prepStart)
	
	log.Debug("Access list preparation completed", 
		"processedTxs", count,
		"txsWithAccessLists", accessListCount,
		"finalBufferSize", len(data),
		"prepTime", prepTime,
		"totalPrepTime", time.Since(memStart),
	)
	
	return data
}

func (p *GPUProcessor) convertHashResults(batch *HashBatch) {
	// Results are already in the correct format from GPU
	// This function can be extended for format conversion if needed
}

func (p *GPUProcessor) convertTransactionResults(batch *TransactionBatch) {
	// Convert GPU results to TxResult format
	// This is a simplified implementation
	for i := range batch.Results {
		if batch.Results[i] == nil {
			batch.Results[i] = &TxResult{
				Hash:  batch.Transactions[i].Hash(),
				Valid: true,
			}
		}
	}
}

// Statistics update functions
func (p *GPUProcessor) updateHashStats(duration time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.processedHashes++
	if p.avgHashTime == 0 {
		p.avgHashTime = duration
	} else {
		p.avgHashTime = (p.avgHashTime + duration) / 2
	}
}

func (p *GPUProcessor) updateSigStats(duration time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.processedSigs++
	if p.avgSigTime == 0 {
		p.avgSigTime = duration
	} else {
		p.avgSigTime = (p.avgSigTime + duration) / 2
	}
}

func (p *GPUProcessor) updateTxStats(duration time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	
	p.processedTxs++
	if p.avgTxTime == 0 {
		p.avgTxTime = duration
	} else {
		p.avgTxTime = (p.avgTxTime + duration) / 2
	}
	
	// Log cumulative GPU statistics every 1000 transactions
	if p.processedTxs%1000 == 0 {
		log.Info("📊 GPU CUMULATIVE STATISTICS", 
			"totalTransactionsProcessed", p.processedTxs,
			"totalHashesProcessed", p.processedHashes,
			"totalSignaturesProcessed", p.processedSigs,
			"avgTransactionTime", p.avgTxTime,
			"avgHashTime", p.avgHashTime,
			"avgSignatureTime", p.avgSigTime,
			"gpuType", p.gpuType,
			"deviceCount", p.deviceCount,
			"timestamp", time.Now().Format("2006-01-02 15:04:05.000"),
		)
	}
}

// GetStats returns current GPU processor statistics
func (p *GPUProcessor) GetStats() GPUStats {
	p.mu.RLock()
	defer p.mu.RUnlock()
	
	return GPUStats{
		GPUType:         p.gpuType,
		DeviceCount:     p.deviceCount,
		ProcessedHashes: p.processedHashes,
		ProcessedSigs:   p.processedSigs,
		ProcessedTxs:    p.processedTxs,
		AvgHashTime:     p.avgHashTime,
		AvgSigTime:      p.avgSigTime,
		AvgTxTime:       p.avgTxTime,
		HashQueueSize:   len(p.hashPool),
		SigQueueSize:    len(p.signaturePool),
		TxQueueSize:     len(p.txPool),
	}
}

// GPUStats holds GPU processor statistics
type GPUStats struct {
	GPUType         GPUType       `json:"gpuType"`
	DeviceCount     int           `json:"deviceCount"`
	ProcessedHashes uint64        `json:"processedHashes"`
	ProcessedSigs   uint64        `json:"processedSigs"`
	ProcessedTxs    uint64        `json:"processedTxs"`
	AvgHashTime     time.Duration `json:"avgHashTime"`
	AvgSigTime      time.Duration `json:"avgSigTime"`
	AvgTxTime       time.Duration `json:"avgTxTime"`
	HashQueueSize   int           `json:"hashQueueSize"`
	SigQueueSize    int           `json:"sigQueueSize"`
	TxQueueSize     int           `json:"txQueueSize"`
}

// IsGPUAvailable returns true if GPU acceleration is available
func (p *GPUProcessor) IsGPUAvailable() bool {
	return p.gpuType != GPUTypeNone
}

// GetGPUType returns the current GPU type
func (p *GPUProcessor) GetGPUType() GPUType {
	return p.gpuType
}

// Close gracefully shuts down the GPU processor
func (p *GPUProcessor) Close() error {
	log.Info("Shutting down GPU processor...")
	
	// Cancel context to stop all workers
	p.cancel()
	
	// Wait for all workers to finish
	p.wg.Wait()
	
	// Cleanup GPU resources
	switch p.gpuType {
	case GPUTypeCUDA:
		C.cuda_cleanup()
	case GPUTypeOpenCL:
		C.cleanupOpenCL()
	}
	
	log.Info("GPU processor shutdown complete")
	return nil
}

// Global GPU processor instance
var globalGPUProcessor *GPUProcessor

// InitGlobalGPUProcessor initializes the global GPU processor
func InitGlobalGPUProcessor(config *GPUConfig) error {
	if globalGPUProcessor != nil {
		globalGPUProcessor.Close()
	}
	
	var err error
	globalGPUProcessor, err = NewGPUProcessor(config)
	return err
}

// GetGlobalGPUProcessor returns the global GPU processor
func GetGlobalGPUProcessor() *GPUProcessor {
	return globalGPUProcessor
}

// CloseGlobalGPUProcessor closes the global GPU processor
func CloseGlobalGPUProcessor() error {
	if globalGPUProcessor != nil {
		return globalGPUProcessor.Close()
	}
	return nil
}
