package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.operators.activations.Sigmoid
import org.jetbrains.research.kotlin.mpp.inference.operators.activations.Tanh
import org.jetbrains.research.kotlin.mpp.inference.tensors.*
import scientifik.kmath.structures.*

open class LSTMLayer<T : Number> {
    open fun apply(inputs: Collection<Tensor>): Collection<Tensor> {
        require(inputs.size in 3..4) { "Applicable only for three or four arguments" }

        val inputList = inputs.toList()

        val inputTensor = inputList[0]
        val weights = inputList[1].squeeze(0)
        val recWeights = inputList[2].squeeze(0)
        val bias = inputList.getOrNull(3)

        val batchSize = inputTensor.data.shape[1]
        val hiddenSize = recWeights.data.shape[1]

        val (mainOutput, currentState) = activate(inputTensor.as2DList(), weights, recWeights, bias)
        val shapeForOutput = intArrayOf(1, batchSize, hiddenSize)
        return listOf(mainOutput.toOutput(), currentState.output.reshape(shapeForOutput), currentState.cellGate.reshape(shapeForOutput))
    }

    protected fun activate(inputMatrices: Collection<Tensor>, weights: Tensor, recWeights: Tensor, bias: Tensor?): Pair<List<Tensor>, State> {
        val hiddenSize = recWeights.data.shape[1]
        val batchSize = inputMatrices.first().data.shape[0]

        var currentState = State.initialize<T>(batchSize, hiddenSize, inputMatrices.first().type)
        val biasesData = if (bias != null) BiasesData.create(bias, hiddenSize, batchSize) else null

        val mainOutput = inputMatrices.map { inputMatrix ->
            val gatesData = GatesData.create(inputMatrix, weights, recWeights, currentState)

            val gatesDataWithBiases = if (biasesData != null) gatesData.addBiases(biasesData.first, biasesData.second) else gatesData
            val activatedGatesData = gatesDataWithBiases.activate()

            currentState = State.create(activatedGatesData, currentState)

            currentState.output
        }

        return Pair(mainOutput, currentState)
    }

    data class GatesData(
        val inputGate: Tensor,
        val outputGate: Tensor,
        val forgetGate: Tensor,
        val cellGate: Tensor
    ) {
        fun activate(): GatesData {
            val activatedInputGate = Sigmoid().apply(inputGate).first()
            val activatedOutputGate = Sigmoid().apply(outputGate).first()
            val activatedForgetGate = Sigmoid().apply(forgetGate).first()
            val activatedCellGate = Tanh().apply(cellGate).first()
            return GatesData(activatedInputGate, activatedOutputGate, activatedForgetGate, activatedCellGate)
        }

        fun addBiases(weightsBiasesData: BiasesData, recursiveWeightsBiasesData: BiasesData): GatesData {
            val inputGateWithBiases = inputGate + weightsBiasesData.inputGateBiases + recursiveWeightsBiasesData.inputGateBiases
            val outputGateWithBiases = outputGate + weightsBiasesData.outputGateBiases + recursiveWeightsBiasesData.outputGateBiases
            val forgetGateWithBiases = forgetGate + weightsBiasesData.forgetGateBiases + recursiveWeightsBiasesData.forgetGateBiases
            val cellGateWithBiases = cellGate + weightsBiasesData.cellGateBiases + recursiveWeightsBiasesData.forgetGateBiases

            return GatesData(inputGateWithBiases, outputGateWithBiases, forgetGateWithBiases, cellGateWithBiases)
        }

        companion object {
            fun create(inputMatrix: Tensor, weights: Tensor, recWeights: Tensor, prevState: State): GatesData {
                val gates = inputMatrix.matmul(weights.transpose()) + prevState.output.matmul(recWeights.transpose())
                val gatesList = gates.splitWithAxis(4, 1)
                return GatesData(gatesList[0], gatesList[1], gatesList[2], gatesList[3])
            }
        }
    }

    data class State(val output: Tensor, val cellGate: Tensor) {
        companion object {
            @Suppress("UNCHECKED_CAST")
            fun <T : Number> initialize(batchSize: Int, hiddenSize: Int, type: TensorProto.DataType): State {
                val newShape = intArrayOf(batchSize, hiddenSize)
                val zeros = BufferNDStructure(TensorStrides(newShape), VirtualBuffer(batchSize * hiddenSize) { 0.0 as T }) as BufferNDStructure<Any>
                return State(Tensor(null, zeros, type), Tensor(null, zeros, type))
            }

            fun create(gatesData: GatesData, prevState: State): State {
                val newCellGate = gatesData.forgetGate * prevState.cellGate + gatesData.inputGate * gatesData.cellGate
                val newOutput = gatesData.outputGate * Tanh().apply(newCellGate).first()
                return State(newOutput, newCellGate)
            }
        }
    }

    data class BiasesData(
        val inputGateBiases: Tensor,
        val outputGateBiases: Tensor,
        val forgetGateBiases: Tensor,
        val cellGateBiases: Tensor
    ) {
        companion object {
            fun create(biases: Tensor, hiddenSize: Int, batchSize: Int): Pair<BiasesData, BiasesData> {
                val shape = intArrayOf(batchSize, hiddenSize)
                val blockSize = hiddenSize * batchSize
                val newStrides = TensorStrides(shape)

                @Suppress("UNCHECKED_CAST")
                val parsedBiases = List(8) { index ->
                    val newBuffer = VirtualBuffer(blockSize) { i ->
                        val indices = newStrides.index(i)
                        val rowNum = indices[0]
                        biases.data.buffer[hiddenSize * index + rowNum]
                    }
                    val newStructure = BufferNDStructure(newStrides, newBuffer)
                    Tensor(null, newStructure, biases.type)
                }
                val weightsBiasesData = BiasesData(parsedBiases[0], parsedBiases[1], parsedBiases[2], parsedBiases[3])
                val recursiveWeightsBiasesData = BiasesData(parsedBiases[4], parsedBiases[5], parsedBiases[6], parsedBiases[7])
                return Pair(weightsBiasesData, recursiveWeightsBiasesData)
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun List<Tensor>.toOutput(): Tensor {
        val newShape = intArrayOf(this.size, 1, this.first().data.shape[0], this.first().data.shape[1])
        val newStrides = TensorStrides(newShape)
        val newData = VirtualBuffer(newStrides.linearSize) { i ->
            val indices = newStrides.index(i)
            val (inputNum, _, rowNum, colNum) = indices
            this[inputNum].data[rowNum, colNum]
        }
        val newBuffer = BufferNDStructure(newStrides, newData)
        return Tensor(null, newBuffer, this.first().type)
    }
}
