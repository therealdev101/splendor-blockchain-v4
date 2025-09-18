package eth

import (
    "context"
    "fmt"
    "math/big"
    "os"
    "time"

    "github.com/ethereum/go-ethereum/accounts"
    "github.com/ethereum/go-ethereum/common"
    "github.com/ethereum/go-ethereum/common/hexutil"
    "github.com/ethereum/go-ethereum/core/types"
    "github.com/ethereum/go-ethereum/crypto"
    ethapi "github.com/ethereum/go-ethereum/internal/ethapi"
    "github.com/ethereum/go-ethereum/rlp"
)

// X402API exposes x402 verification/settlement RPCs
type X402API struct{ eth *Ethereum }

func NewX402API(eth *Ethereum) *X402API { return &X402API{eth: eth} }

type PaymentRequirements struct {
    Scheme            string         `json:"scheme"`
    Network           string         `json:"network"`
    MaxAmountRequired *hexutil.Big   `json:"maxAmountRequired"`
    Resource          string         `json:"resource"`
    Description       string         `json:"description"`
    MimeType          string         `json:"mimeType"`
    PayTo             common.Address `json:"payTo"`
    MaxTimeoutSeconds uint64         `json:"maxTimeoutSeconds"`
    Asset             common.Address `json:"asset"`
}

type PaymentPayload struct {
    X402Version int                `json:"x402Version"`
    Scheme      string             `json:"scheme"`
    Network     string             `json:"network"`
    Payload     PaymentPayloadData `json:"payload"`
}

type PaymentPayloadData struct {
    From        common.Address `json:"from"`
    To          common.Address `json:"to"`
    Value       *hexutil.Big   `json:"value"`
    ValidAfter  uint64         `json:"validAfter"`
    ValidBefore uint64         `json:"validBefore"`
    Nonce       common.Hash    `json:"nonce"`
    Signature   hexutil.Bytes  `json:"signature"`
}

type VerificationResponse struct {
    IsValid       bool   `json:"isValid"`
    InvalidReason string `json:"invalidReason,omitempty"`
    PayerAddress  string `json:"payerAddress,omitempty"`
}

type SettlementResponse struct {
    Success   bool        `json:"success"`
    Error     string      `json:"error,omitempty"`
    TxHash    common.Hash `json:"txHash,omitempty"`
    NetworkId string      `json:"networkId,omitempty"`
}

func (api *X402API) Verify(ctx context.Context, req PaymentRequirements, pl PaymentPayload) (*VerificationResponse, error) {
    // Basic checks
    if pl.Scheme != "exact" || pl.Network == "" {
        return &VerificationResponse{IsValid: false, InvalidReason: "unsupported scheme/network"}, nil
    }
    now := uint64(time.Now().Unix())
    if now < pl.Payload.ValidAfter { return &VerificationResponse{IsValid:false, InvalidReason:"not yet valid"}, nil }
    if now > pl.Payload.ValidBefore { return &VerificationResponse{IsValid:false, InvalidReason:"expired"}, nil }
    if pl.Payload.To != req.PayTo { return &VerificationResponse{IsValid:false, InvalidReason:"recipient mismatch"}, nil }
    if (*big.Int)(pl.Payload.Value).Cmp((*big.Int)(req.MaxAmountRequired)) != 0 { return &VerificationResponse{IsValid:false, InvalidReason:"amount mismatch"}, nil }

    // Strict canonical message per production guide
    valHex := pl.Payload.Value.String()
    msg := fmt.Sprintf("x402-payment:%s:%s:%s:%d:%d:%s:%d",
        pl.Payload.From.Hex(), pl.Payload.To.Hex(), valHex,
        pl.Payload.ValidAfter, pl.Payload.ValidBefore, pl.Payload.Nonce.Hex(), api.eth.networkID,
    )
    sig := make([]byte, len(pl.Payload.Signature))
    copy(sig, pl.Payload.Signature)
    if len(sig) != 65 { return &VerificationResponse{IsValid:false, InvalidReason:"bad signature length"}, nil }
    if sig[64] >= 27 { sig[64] -= 27 }
    hash := accounts.TextHash([]byte(msg))
    pub, err := crypto.SigToPub(hash, sig)
    if err != nil { return &VerificationResponse{IsValid:false, InvalidReason:"signature recover failed"}, nil }
    addr := crypto.PubkeyToAddress(*pub)
    if addr != pl.Payload.From { return &VerificationResponse{IsValid:false, InvalidReason:"recovered address mismatch"}, nil }

    // Balance check
    st, err := api.eth.blockchain.State()
    if err != nil { return &VerificationResponse{IsValid:false, InvalidReason:"state unavailable"}, nil }
    if st.GetBalance(pl.Payload.From).Cmp((*big.Int)(pl.Payload.Value)) < 0 {
        return &VerificationResponse{IsValid:false, InvalidReason:"insufficient balance"}, nil
    }
    return &VerificationResponse{IsValid:true, PayerAddress: pl.Payload.From.Hex()}, nil
}

func (api *X402API) Settle(ctx context.Context, req PaymentRequirements, pl PaymentPayload) (*SettlementResponse, error) {
    v, err := api.Verify(ctx, req, pl)
    if err != nil { return &SettlementResponse{Success:false, Error: err.Error()}, nil }
    if !v.IsValid { return &SettlementResponse{Success:false, Error: v.InvalidReason}, nil }

    // Encode payload
    type x402Payload struct {
        From        common.Address
        To          common.Address
        Value       *big.Int
        ValidAfter  uint64
        ValidBefore uint64
        Nonce       common.Hash
        Signature   []byte
    }
    enc, err := rlpEncode(x402Payload{
        From: pl.Payload.From,
        To: pl.Payload.To,
        Value: (*big.Int)(pl.Payload.Value),
        ValidAfter: pl.Payload.ValidAfter,
        ValidBefore: pl.Payload.ValidBefore,
        Nonce: pl.Payload.Nonce,
        Signature: append([]byte(nil), pl.Payload.Signature...),
    })
    if err != nil { return &SettlementResponse{Success:false, Error: fmt.Sprintf("encode failed: %v", err)}, nil }

    chainID := api.eth.blockchain.Config().ChainID
    tx := types.NewX402Tx(chainID, 0, nil, enc)

    // Optional envelope signing with etherbase (if wallet available)
    if eb, e := api.eth.Etherbase(); e == nil {
        if w, werr := api.eth.AccountManager().Find(accounts.Account{Address: eb}); w != nil && werr == nil {
            if signed, serr := w.SignTx(accounts.Account{Address: eb}, tx, chainID); serr == nil { tx = signed }
        }
    }
    hash, subErr := ethapi.SubmitTransaction(ctx, api.eth.APIBackend, tx)
    if subErr != nil { return &SettlementResponse{Success:false, Error: fmt.Sprintf("submit failed: %v", subErr)}, nil }
    return &SettlementResponse{Success:true, TxHash: hash, NetworkId: os.Getenv("NETWORK_ID")}, nil
}

// small helper to avoid pulling rlp directly into this file
func rlpEncode(v interface{}) ([]byte, error) { return rlp.EncodeToBytes(v) }
