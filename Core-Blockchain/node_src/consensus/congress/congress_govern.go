package congress

import (
    "bytes"
    "errors"
    "fmt"
    "github.com/ethereum/go-ethereum/accounts"
    "github.com/ethereum/go-ethereum/common"
    "github.com/ethereum/go-ethereum/common/hexutil"
    "github.com/ethereum/go-ethereum/consensus"
    "github.com/ethereum/go-ethereum/consensus/congress/systemcontract"
    "github.com/ethereum/go-ethereum/consensus/congress/vmcaller"
    "github.com/ethereum/go-ethereum/core/state"
    "github.com/ethereum/go-ethereum/core/types"
    "github.com/ethereum/go-ethereum/core/vm"
    "github.com/ethereum/go-ethereum/crypto"
    "github.com/ethereum/go-ethereum/log"
    "github.com/ethereum/go-ethereum/rlp"
    "math"
    "math/big"
)

// Proposal is the system governance proposal info.
type Proposal struct {
	Id     *big.Int
	Action *big.Int
	From   common.Address
	To     common.Address
	Value  *big.Int
	Data   []byte
}

func (c *Congress) getPassedProposalCount(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB) (uint32, error) {

	method := "getPassedProposalCount"
	data, err := c.abi[systemcontract.SysGovContractName].Pack(method)
	if err != nil {
		log.Error("Can't pack data for getPassedProposalCount", "error", err)
		return 0, err
	}

	msg := vmcaller.NewLegacyMessage(header.Coinbase, &systemcontract.SysGovContractAddr, 0, new(big.Int), math.MaxUint64, new(big.Int), data, false)

	// use parent
	result, err := vmcaller.ExecuteMsg(msg, state, header, newChainContext(chain, c), c.chainConfig)
	if err != nil {
		return 0, err
	}

	// unpack data
	ret, err := c.abi[systemcontract.SysGovContractName].Unpack(method, result)
	if err != nil {
		return 0, err
	}
	if len(ret) != 1 {
		return 0, errors.New("invalid output length")
	}
	count, ok := ret[0].(uint32)
	if !ok {
		return 0, errors.New("invalid count format")
	}

	return count, nil
}

func (c *Congress) getPassedProposalByIndex(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB, idx uint32) (*Proposal, error) {

	method := "getPassedProposalByIndex"
	data, err := c.abi[systemcontract.SysGovContractName].Pack(method, idx)
	if err != nil {
		log.Error("Can't pack data for getPassedProposalByIndex", "error", err)
		return nil, err
	}

	msg := vmcaller.NewLegacyMessage(header.Coinbase, &systemcontract.SysGovContractAddr, 0, new(big.Int), math.MaxUint64, new(big.Int), data, false)

	// use parent
	result, err := vmcaller.ExecuteMsg(msg, state, header, newChainContext(chain, c), c.chainConfig)
	if err != nil {
		return nil, err
	}

	// unpack data
	prop := &Proposal{}
	err = c.abi[systemcontract.SysGovContractName].UnpackIntoInterface(prop, method, result)
	if err != nil {
		return nil, err
	}

	return prop, nil
}

//finishProposalById
func (c *Congress) finishProposalById(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB, id *big.Int) error {
	method := "finishProposalById"
	data, err := c.abi[systemcontract.SysGovContractName].Pack(method, id)
	if err != nil {
		log.Error("Can't pack data for getPassedProposalByIndex", "error", err)
		return err
	}

	msg := vmcaller.NewLegacyMessage(header.Coinbase, &systemcontract.SysGovContractAddr, 0, new(big.Int), math.MaxUint64, new(big.Int), data, false)

	// execute message without a transaction
	state.Prepare(common.Hash{}, 0)
	_, err = vmcaller.ExecuteMsg(msg, state, header, newChainContext(chain, c), c.chainConfig)
	if err != nil {
		return err
	}

	return nil
}

func (c *Congress) executeProposal(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB, prop *Proposal, totalTxIndex int) (*types.Transaction, *types.Receipt, error) {
	// Even if the miner is not `running`, it's still working,
	// the 'miner.worker' will try to FinalizeAndAssemble a block,
	// in this case, the signTxFn is not set. A `non-miner node` can't execute system governance proposal.
	if c.signTxFn == nil {
		return nil, nil, errors.New("signTxFn not set")
	}

	propRLP, err := rlp.EncodeToBytes(prop)
	if err != nil {
		return nil, nil, err
	}
	//make system governance transaction
	nonce := state.GetNonce(c.validator)
	amout := prop.Value
	if c.chainConfig.IsSophon(header.Number) {
		// fix bug
		amout = new(big.Int)
	}
	tx := types.NewTransaction(nonce, systemcontract.SysGovToAddr, amout, header.GasLimit, new(big.Int), propRLP)
	tx, err = c.signTxFn(accounts.Account{Address: c.validator}, tx, chain.Config().ChainID)
	if err != nil {
		return nil, nil, err
	}
	//add nonce for validator
	state.SetNonce(c.validator, nonce+1)
	receipt := c.executeProposalMsg(chain, header, state, prop, totalTxIndex, tx.Hash(), common.Hash{})

	return tx, receipt, nil
}

func (c *Congress) replayProposal(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB, prop *Proposal, totalTxIndex int, tx *types.Transaction) (*types.Receipt, error) {
	sender, err := types.Sender(c.signer, tx)
	if err != nil {
		return nil, err
	}
	if sender != header.Coinbase {
		return nil, errors.New("invalid sender for system governance transaction")
	}
	propRLP, err := rlp.EncodeToBytes(prop)
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(propRLP, tx.Data()) {
		return nil, fmt.Errorf("data missmatch, proposalID: %s, rlp: %s, txHash:%s, txData:%s", prop.Id.String(), hexutil.Encode(propRLP), tx.Hash().String(), hexutil.Encode(tx.Data()))
	}
	//make system governance transaction
	nonce := state.GetNonce(sender)
	//add nonce for validator
	state.SetNonce(sender, nonce+1)
	receipt := c.executeProposalMsg(chain, header, state, prop, totalTxIndex, tx.Hash(), header.Hash())

	return receipt, nil
}

func (c *Congress) executeProposalMsg(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB, prop *Proposal, totalTxIndex int, txHash, bHash common.Hash) *types.Receipt {
	var receipt *types.Receipt
	action := prop.Action.Uint64()
	switch action {
	case 0:
		// evm action.
		receipt = c.executeEvmCallProposal(chain, header, state, prop, totalTxIndex, txHash, bHash)
	case 1:
		// delete code action
		ok := state.Erase(prop.To)
		receipt = &types.Receipt{
			Type:              types.LegacyTxType,
			PostState:         []byte{},
			Status:            types.ReceiptStatusFailed,
			CumulativeGasUsed: header.GasUsed,
		}
		if ok {
			receipt.Status = types.ReceiptStatusSuccessful
		}
		log.Info("executeProposalMsg", "action", "erase", "id", prop.Id.String(), "to", prop.To, "txHash", txHash.String(), "success", ok)
	default:
		receipt = &types.Receipt{
			Type:              types.LegacyTxType,
			PostState:         []byte{},
			Status:            types.ReceiptStatusFailed,
			CumulativeGasUsed: header.GasUsed,
		}
		log.Warn("executeProposalMsg failed, unsupported action", "action", action, "id", prop.Id.String(), "from", prop.From, "to", prop.To, "value", prop.Value.String(), "data", hexutil.Encode(prop.Data), "txHash", txHash.String())
	}

	receipt.TxHash = txHash
	receipt.BlockHash = bHash
	receipt.BlockNumber = header.Number
	receipt.TransactionIndex = uint(state.TxIndex())

	return receipt
}

// the returned value should not nil.
func (c *Congress) executeEvmCallProposal(chain consensus.ChainHeaderReader, header *types.Header, state *state.StateDB, prop *Proposal, totalTxIndex int, txHash, bHash common.Hash) *types.Receipt {
	// actually run the governance message
	msg := vmcaller.NewLegacyMessage(prop.From, &prop.To, 0, prop.Value, header.GasLimit, new(big.Int), prop.Data, false)
	state.Prepare(txHash, totalTxIndex)
	_, err := vmcaller.ExecuteMsg(msg, state, header, newChainContext(chain, c), c.chainConfig)

	// governance message will not actually consumes gas
	receipt := &types.Receipt{
		Type:              types.LegacyTxType,
		PostState:         []byte{},
		Status:            types.ReceiptStatusSuccessful,
		CumulativeGasUsed: header.GasUsed,
	}
	if err != nil {
		receipt.Status = types.ReceiptStatusFailed
	}
	// Set the receipt logs and create a bloom for filtering
	receipt.Logs = state.GetLogs(txHash, bHash)
	receipt.Bloom = types.CreateBloom(types.Receipts{receipt})

	log.Info("executeProposalMsg", "action", "evmCall", "id", prop.Id.String(), "from", prop.From, "to", prop.To, "value", prop.Value.String(), "data", hexutil.Encode(prop.Data), "txHash", txHash.String(), "err", err)

	return receipt
}

// Methods for debug trace

// ApplySysTx applies a system-transaction using a given evm,
// the main purpose of this method is for tracing a system-transaction.
func (c *Congress) ApplySysTx(evm *vm.EVM, state *state.StateDB, txIndex int, sender common.Address, tx *types.Transaction) (ret []byte, vmerr error, err error) {
    // Handle native x402 typed settlement envelopes
    if tx.Type() == types.X402TxType {
        type x402Payload struct {
            From        common.Address
            To          common.Address
            Value       *big.Int
            ValidAfter  uint64
            ValidBefore uint64
            Nonce       common.Hash
            Signature   []byte
        }
        var p x402Payload
        if err = rlp.DecodeBytes(tx.Data(), &p); err != nil {
            return nil, nil, fmt.Errorf("x402: decode payload failed: %v", err)
        }
        now := evm.Context.Time.Uint64()
        if now < p.ValidAfter || now > p.ValidBefore {
            return nil, nil, errors.New("x402: outside validity window")
        }
        // Canonical message per production guide
        msg := fmt.Sprintf("x402-payment:%s:%s:%s:%d:%d:%s:%d",
            p.From.Hex(), p.To.Hex(), (*hexutil.Big)(p.Value).String(), p.ValidAfter, p.ValidBefore, p.Nonce.Hex(), c.chainConfig.ChainID.Uint64())
        sig := make([]byte, len(p.Signature))
        copy(sig, p.Signature)
        if len(sig) != 65 {
            return nil, nil, errors.New("x402: bad signature length")
        }
        if sig[64] >= 27 { sig[64] -= 27 }
        h := accounts.TextHash([]byte(msg))
        pub, rerr := crypto.SigToPub(h, sig)
        if rerr != nil { return nil, nil, fmt.Errorf("x402: recover failed: %v", rerr) }
        if crypto.PubkeyToAddress(*pub) != p.From {
            return nil, nil, errors.New("x402: recovered address mismatch")
        }
        // Anti-replay: keccak(from||nonce) stored under registry address
        registry := common.HexToAddress("0x0000000000000000000000000000000000000402")
        key := crypto.Keccak256Hash(append(p.From.Bytes(), p.Nonce.Bytes()...))
        if state.GetState(registry, key) != (common.Hash{}) {
            return nil, nil, errors.New("x402: nonce already used")
        }
        state.SetState(registry, key, common.BigToHash(big.NewInt(1)))
        // Apply balance transfer
        if state.GetBalance(p.From).Cmp(p.Value) < 0 {
            return nil, nil, errors.New("x402: insufficient balance")
        }
        state.SubBalance(p.From, p.Value)
        state.AddBalance(p.To, p.Value)
        // Increment validator (sender) nonce to consume the sys-tx
        evm.StateDB.SetNonce(sender, evm.StateDB.GetNonce(sender)+1)
        return nil, nil, nil
    }
    var prop = &Proposal{}
    if err = rlp.DecodeBytes(tx.Data(), prop); err != nil {
        return
    }
	evm.Context.ExtraValidator = nil
	nonce := evm.StateDB.GetNonce(sender)
	//add nonce for validator
	evm.StateDB.SetNonce(sender, nonce+1)

	action := prop.Action.Uint64()
	switch action {
	case 0:
		// evm action.
		// actually run the governance message
		msg := vmcaller.NewLegacyMessage(prop.From, &prop.To, 0, prop.Value, tx.Gas(), new(big.Int), prop.Data, false)
		state.Prepare(tx.Hash(), txIndex)
		evm.TxContext = vm.TxContext{
			Origin:   msg.From(),
			GasPrice: new(big.Int).Set(msg.GasPrice()),
		}
		ret, _, vmerr = evm.Call(vm.AccountRef(msg.From()), *msg.To(), msg.Data(), msg.Gas(), msg.Value())
		state.Finalise(true)
	case 1:
		// delete code action
		_ = state.Erase(prop.To)
	default:
		vmerr = errors.New("unsupported action")
	}
	return
}
