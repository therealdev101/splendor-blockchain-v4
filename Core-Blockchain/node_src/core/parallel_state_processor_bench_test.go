package core

import (
	"fmt"
	"math/big"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/gopool"
	"github.com/ethereum/go-ethereum/core/rawdb"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/core/vm"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/params"
)

func BenchmarkParallelStateProcessorPipeline(b *testing.B) {
	txCount := 120000
	chainCfg := params.TestChainConfig

	key, err := crypto.GenerateKey()
	if err != nil {
		b.Fatalf("failed to generate key: %v", err)
	}
	from := crypto.PubkeyToAddress(key.PublicKey)
	to := common.HexToAddress("0x000000000000000000000000000000000000dead")

	db := rawdb.NewMemoryDatabase()
	stateDB, err := state.New(common.Hash{}, state.NewDatabase(db), nil)
	if err != nil {
		b.Fatalf("failed to create state: %v", err)
	}
	stateDB.CreateAccount(from)
	stateDB.AddBalance(from, big.NewInt(0).Mul(big.NewInt(1e18), big.NewInt(int64(txCount))))

	txs := make([]*types.Transaction, txCount)
	signer := types.MakeSigner(chainCfg, big.NewInt(1))
	gasLimit := uint64(50000)
	gasUsedPerTx := uint64(21000)

	for i := 0; i < txCount; i++ {
		unsigned := types.NewTransaction(uint64(i), to, big.NewInt(1), gasLimit, big.NewInt(1), nil)
		signed, signErr := types.SignTx(unsigned, signer, key)
		if signErr != nil {
			b.Fatalf("failed to sign tx: %v", signErr)
		}
		txs[i] = signed
	}

	baseState := stateDB
	blockHash := common.HexToHash("0x1234")
	blockNumber := big.NewInt(1)
	blockGasLimit := uint64(txCount) * gasLimit

	createProcessor := func(workers int) *ParallelStateProcessor {
		parallelCfg := DefaultParallelProcessorConfig()
		parallelCfg.EnableTxBatching = false
		parallelCfg.EnableBloomParallel = false
		parallelCfg.EnablePipelining = true
		parallelCfg.MaxTxConcurrency = workers
		parallelCfg.TxBatchSize = 1024
		parallelCfg.MaxValidationWorkers = workers
		parallelCfg.MaxGoroutines = runtime.NumCPU() * 2
		parallelCfg.StateWorkers = runtime.NumCPU()
		processorCfg := &gopool.ProcessorConfig{
			MaxWorkers:        parallelCfg.MaxGoroutines,
			QueueSize:         1 << 14,
			Timeout:           time.Second,
			TxWorkers:         parallelCfg.MaxTxConcurrency,
			ValidationWorkers: parallelCfg.MaxValidationWorkers,
			StateWorkers:      parallelCfg.StateWorkers,
		}
		processor, err := gopool.NewParallelProcessor(processorCfg)
		if err != nil {
			b.Fatalf("failed to construct processor: %v", err)
		}
		psp := &ParallelStateProcessor{
			StateProcessor: &StateProcessor{config: chainCfg},
			processor:      processor,
			config:         parallelCfg,
			maxConcurrency: int32(workers),
		}
		psp.txExecutor = func(_ types.Message, _ *params.ChainConfig, _ ChainContext, _ *common.Address, _ *GasPool, _ *state.StateDB, _ *big.Int, _ common.Hash, tx *types.Transaction, usedGas *uint64, _ *vm.EVM, _ *sync.WaitGroup) (*types.Receipt, error) {
			time.Sleep(50 * time.Microsecond)
			*usedGas += gasUsedPerTx
			receipt := &types.Receipt{
				Type:              tx.Type(),
				CumulativeGasUsed: *usedGas,
				GasUsed:           gasUsedPerTx,
				TxHash:            tx.Hash(),
				BlockHash:         blockHash,
				BlockNumber:       blockNumber,
			}
			return receipt, nil
		}
		return psp
	}

	workers := []int{1, runtime.NumCPU()}
	for _, w := range workers {
		b.Run(fmt.Sprintf("workers_%d", w), func(b *testing.B) {
			psp := createProcessor(w)
			b.Cleanup(func() {
				_ = psp.processor.Close()
			})

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				stateCopy := baseState.Copy()
				gp := new(GasPool).AddGas(blockGasLimit)
				used := new(uint64)
				_, _, err := psp.processPipelinedTransactions(txs, stateCopy, nil, gp, used, blockNumber, blockHash)
				if err != nil {
					b.Fatalf("pipeline execution failed: %v", err)
				}
			}
		})
	}
}
