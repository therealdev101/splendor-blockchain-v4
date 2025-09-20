package eth

import (
	"context"
	"math/big"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/core/rawdb"
	"github.com/ethereum/go-ethereum/core/state"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/eth/ethconfig"
	"github.com/ethereum/go-ethereum/event"
	"github.com/ethereum/go-ethereum/params"
	"github.com/ethereum/go-ethereum/trie"
)

type testAPITxPoolChain struct {
	gasLimit uint64
	state    *state.StateDB
	headFeed event.Feed
}

func (bc *testAPITxPoolChain) CurrentBlock() *types.Block {
	header := &types.Header{GasLimit: bc.gasLimit}
	return types.NewBlock(header, nil, nil, nil, trie.NewStackTrie(nil))
}

func (bc *testAPITxPoolChain) GetBlock(hash common.Hash, number uint64) *types.Block {
	return bc.CurrentBlock()
}

func (bc *testAPITxPoolChain) StateAt(root common.Hash) (*state.StateDB, error) {
	return bc.state, nil
}

func (bc *testAPITxPoolChain) SubscribeChainHeadEvent(ch chan<- core.ChainHeadEvent) event.Subscription {
	return bc.headFeed.Subscribe(ch)
}

func newTestAPIBackend(t *testing.T, async bool) (*EthAPIBackend, *core.TxPool, *types.Transaction, func()) {
	t.Helper()

	key, err := crypto.GenerateKey()
	if err != nil {
		t.Fatalf("failed to generate key: %v", err)
	}
	addr := crypto.PubkeyToAddress(key.PublicKey)

	db := state.NewDatabase(rawdb.NewMemoryDatabase())
	statedb, err := state.New(common.Hash{}, db, nil)
	if err != nil {
		t.Fatalf("failed to create state: %v", err)
	}
	statedb.SetBalance(addr, big.NewInt(1e18))

	bc := &testAPITxPoolChain{
		gasLimit: params.GenesisGasLimit,
		state:    statedb,
	}

	cfg := ethconfig.Defaults
	cfg.TxPool = core.DefaultTxPoolConfig
	cfg.TxPool.Journal = ""
	cfg.TxPool.AsyncLocals = async

	eth := &Ethereum{config: &cfg}
	pool := core.NewTxPool(cfg.TxPool, params.TestChainConfig, bc)
	eth.txPool = pool

	tx := types.MustSignNewTx(key, types.LatestSigner(params.TestChainConfig), &types.LegacyTx{
		Nonce:    0,
		To:       new(common.Address),
		Value:    big.NewInt(1),
		Gas:      params.TxGas,
		GasPrice: big.NewInt(params.InitialBaseFee),
	})

	backend := &EthAPIBackend{eth: eth}
	cleanup := func() {
		pool.Stop()
	}
	return backend, pool, tx, cleanup
}

func waitForTxStatus(t *testing.T, pool *core.TxPool, hash common.Hash, want core.TxStatus) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		status := pool.Status([]common.Hash{hash})[0]
		if status == want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	status := pool.Status([]common.Hash{hash})[0]
	t.Fatalf("transaction status %v after waiting, want %v", status, want)
}

func TestSendTxAsyncLocalImmediateQuery(t *testing.T) {
	backend, pool, tx, cleanup := newTestAPIBackend(t, true)
	defer cleanup()

	if err := backend.SendTx(context.Background(), tx); err != nil {
		t.Fatalf("SendTx async failed: %v", err)
	}
	if got := backend.GetPoolTransaction(tx.Hash()); got == nil {
		t.Fatalf("transaction not immediately queryable")
	}
	waitForTxStatus(t, pool, tx.Hash(), core.TxStatusPending)
}

func TestSendTxSyncLocalImmediatePromotion(t *testing.T) {
	backend, pool, tx, cleanup := newTestAPIBackend(t, false)
	defer cleanup()

	if err := backend.SendTx(context.Background(), tx); err != nil {
		t.Fatalf("SendTx sync failed: %v", err)
	}
	if got := backend.GetPoolTransaction(tx.Hash()); got == nil {
		t.Fatalf("transaction not immediately queryable")
	}
	status := pool.Status([]common.Hash{tx.Hash()})[0]
	if status != core.TxStatusPending {
		t.Fatalf("expected pending status immediately, got %v", status)
	}
}
