import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing import prepare_quantum_data
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def encode_transaction(features):
    """
    Encode transaction features into a quantum circuit.
    """

    n_qubits = len(features)

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Create superposition
    for qubit in range(n_qubits):
        qc.h(qubit)

    # Encode transaction features using Ry rotations
    for qubit, value in enumerate(features):
        qc.ry(value, qubit)

    # Create entanglement
    for qubit in range(n_qubits - 1):
        qc.cx(qubit, qubit + 1)

    # Measure all qubits
    qc.measure(range(n_qubits), range(n_qubits))

    return qc


def simulate_circuit(circuit):
    """
    Execute the quantum circuit using AerSimulator.
    """

    simulator = AerSimulator()

    # Run circuit with 1024 measurements
    job = simulator.run(circuit, shots=1024)

    result = job.result()

    counts = result.get_counts()

    return counts


def main():

    print("Preparing quantum dataset...")

    X_train, X_test, y_train, y_test = prepare_quantum_data()

    print("Quantum dataset ready.")

    print("\nTraining samples:", X_train.shape)
    print("Testing samples :", X_test.shape)

    # Select one transaction
    sample = X_train[0]

    print("\nFirst transaction features:")
    print(sample)

    print("\nCreating quantum circuit...")

    circuit = encode_transaction(sample)

    print("\nQuantum circuit:")
    print(circuit)

    print("\nNumber of qubits:", circuit.num_qubits)
    print("Circuit depth:", circuit.depth())

    print("\nRunning quantum circuit on Aer simulator...")

    counts = simulate_circuit(circuit)

    print("\n========== QUANTUM MEASUREMENT RESULTS ==========")

    for state, count in sorted(counts.items()):
        probability = count / 1024

        print(
            f"|{state}> : {count} shots "
            f"({probability:.4f})"
        )


if __name__ == "__main__":
    main()