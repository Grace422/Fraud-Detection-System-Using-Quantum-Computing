import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing import prepare_quantum_data

from qiskit import QuantumCircuit


def encode_transaction(features):
    """
    Encode one transaction into a 5-qubit quantum circuit.

    Features:
        V1
        V2
        V3
        V4
        Amount
    """

    n_qubits = len(features)

    qc = QuantumCircuit(n_qubits)

    # -----------------------------------------
    # 1. Superposition
    # -----------------------------------------

    for qubit in range(n_qubits):
        qc.h(qubit)

    # -----------------------------------------
    # 2. Encode transaction features
    # -----------------------------------------

    for qubit, value in enumerate(features):
        qc.ry(value, qubit)

    # -----------------------------------------
    # 3. Entanglement
    # -----------------------------------------

    for qubit in range(n_qubits - 1):
        qc.cx(qubit, qubit + 1)

    return qc


def main():

    print("Preparing quantum dataset...")

    X_train, X_test, y_train, y_test = prepare_quantum_data()

    print("Quantum dataset ready.")

    print("\nTraining samples:", X_train.shape)
    print("Testing samples :", X_test.shape)

    # Take the first transaction
    sample = X_train[0]

    print("\nFirst transaction features:")
    print(sample)

    print("\nCreating quantum circuit...")

    circuit = encode_transaction(sample)

    print("\nEncoded quantum circuit:")
    print(circuit)

    print("\nNumber of qubits:", circuit.num_qubits)

    print("Circuit depth:", circuit.depth())


if __name__ == "__main__":
    main()