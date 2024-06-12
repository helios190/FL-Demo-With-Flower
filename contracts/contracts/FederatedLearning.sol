// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/// @title Federated Learning Contract
/// @notice This contract handles client registration and token management for federated learning.
/// @custom:dev-run-script scripts/deploy.js
contract FederatedLearning {
    mapping(address => uint256) public balances;
    uint256 public constant TOKEN_REWARD = 100;

    function registerClient() public {
        balances[msg.sender] += TOKEN_REWARD;
    }

    function getBalance(address client) public view returns (uint256) {
        return balances[client];
    }

    function useGas(address client, uint256 amount) public {
        require(balances[client] >= amount, "Insufficient balance");
        balances[client] -= amount;
    }
}
