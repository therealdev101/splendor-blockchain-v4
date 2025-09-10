// Copyright 2021 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package misc

import (
	"fmt"
	"math/big"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/params"
)

// VerifyEip1559Header verifies some header attributes which were changed in EIP-1559,
// - gas limit check
// - basefee check
func VerifyEip1559Header(config *params.ChainConfig, parent, header *types.Header) error {
	// Verify that the gas limit remains within allowed bounds
	parentGasLimit := parent.GasLimit

	if err := VerifyGaslimit(parentGasLimit, header.GasLimit); err != nil {
		return err
	}
	// Verify the header is not malformed
	if header.BaseFee == nil {
		return fmt.Errorf("header is missing baseFee")
	}
	// Verify the baseFee is correct based on the parent header.
	expectedBaseFee := CalcBaseFee(config, parent)
	if header.BaseFee.Cmp(expectedBaseFee) != 0 {
		return fmt.Errorf("invalid baseFee: have %s, want %s, parentBaseFee %s, parentGasUsed %d",
			expectedBaseFee, header.BaseFee, parent.BaseFee, parent.GasUsed)
	}
	return nil
}

// CalcBaseFee calculates the basefee of the header.
func CalcBaseFee(config *params.ChainConfig, parent *types.Header) *big.Int {
	// If not London fork, return 0
	if !config.IsLondon(parent.Number) {
		return common.Big0
	}
	
	// EIP-1559 base fee calculation
	parentBaseFee := parent.BaseFee
	if parentBaseFee == nil {
		return new(big.Int).SetUint64(params.InitialBaseFee)
	}
	
	// Calculate based on parent gas usage
	parentGasTarget := parent.GasLimit / params.ElasticityMultiplier
	
	if parent.GasUsed == parentGasTarget {
		// If parent block used exactly the target gas, keep base fee the same
		return new(big.Int).Set(parentBaseFee)
	} else if parent.GasUsed > parentGasTarget {
		// If parent block used more gas than target, increase base fee
		gasUsedDelta := new(big.Int).SetUint64(parent.GasUsed - parentGasTarget)
		x := new(big.Int).Mul(parentBaseFee, gasUsedDelta)
		y := x.Div(x, new(big.Int).SetUint64(parentGasTarget))
		baseFeeDelta := x.Div(y, new(big.Int).SetUint64(params.BaseFeeChangeDenominator))
		
		// Ensure minimum increase of 1 wei
		if baseFeeDelta.Cmp(common.Big0) == 0 {
			baseFeeDelta = common.Big1
		}
		
		return x.Add(parentBaseFee, baseFeeDelta)
	} else {
		// If parent block used less gas than target, decrease base fee
		gasUsedDelta := new(big.Int).SetUint64(parentGasTarget - parent.GasUsed)
		x := new(big.Int).Mul(parentBaseFee, gasUsedDelta)
		y := x.Div(x, new(big.Int).SetUint64(parentGasTarget))
		baseFeeDelta := x.Div(y, new(big.Int).SetUint64(params.BaseFeeChangeDenominator))
		
		result := x.Sub(parentBaseFee, baseFeeDelta)
		
		// Ensure base fee never goes below minimum floor (1 Gwei = $0.001 per tx)
		minimumBaseFee := new(big.Int).SetUint64(params.MinimumBaseFee)
		if result.Cmp(minimumBaseFee) < 0 {
			return minimumBaseFee
		}
		
		return result
	}
}
