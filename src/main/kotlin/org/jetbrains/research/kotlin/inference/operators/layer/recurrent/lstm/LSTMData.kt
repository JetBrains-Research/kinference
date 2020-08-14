package org.jetbrains.research.kotlin.inference.operators.layer.recurrent.lstm

import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*
import org.jetbrains.research.kotlin.inference.onnx.TensorProto.DataType

class LSTMData(val weights: GatesData,
               val recurrentWeights: GatesData,
               val bias: GatesData?,
               val initialOutput: List<TypedNDArray<Any>>?,
               val initialCellState: List<TypedNDArray<Any>>?,
               val peepholes: GatesData?,
               val type: DataType) {

    fun updateWeights(weights: GatesData) = LSTMData(weights, recurrentWeights, bias, initialOutput, initialCellState, peepholes, type)
    fun updateRecurrentWeights(recurrentWeights: GatesData) = LSTMData(weights, recurrentWeights, bias, initialOutput, initialCellState, peepholes, type)
    fun updateBias(bias: GatesData) = LSTMData(weights, recurrentWeights, bias, initialOutput, initialCellState, peepholes, type)
    fun updateInitialOutput(initialOutput: List<TypedNDArray<Any>>) = LSTMData(weights, recurrentWeights, bias, initialOutput, initialCellState, peepholes, type)
    fun updateInitialCellGate(initialCellSate: List<TypedNDArray<Any>>) = LSTMData(weights, recurrentWeights, bias, initialOutput, initialCellSate, peepholes, type)
    fun updatePeepholes(peepholes: GatesData) = LSTMData(weights, recurrentWeights, bias, initialOutput, initialCellState, peepholes, type)
}


data class GatesData(val input: MutableTypedNDArray<Any>,
                     val output: MutableTypedNDArray<Any>,
                     val forget: MutableTypedNDArray<Any>,
                     val cellGate: MutableTypedNDArray<Any>
) {
    companion object {
        fun createWeights(weights: MutableTypedNDArray<Any>): GatesData {
            require(weights.shape[0] == 1)

            val matrix = weights.squeeze(0)

            val weightsList = matrix.splitWithAxis(4)
            return GatesData(weightsList[0].matrixTranspose(), weightsList[1].matrixTranspose(),
                weightsList[2].matrixTranspose(), weightsList[3].matrixTranspose())
        }

        fun createBias(bias: MutableTypedNDArray<Any>): GatesData {
            require(bias.shape[0] == 1)

            val linear = bias.squeeze(0)

            val biasList = linear.splitWithAxis(8)

            return GatesData(
                biasList[0].apply { plusAssign(biasList[4]) }.wrapOneDim(),
                biasList[1].apply { plusAssign(biasList[5]) }.wrapOneDim(),
                biasList[2].apply { plusAssign(biasList[6]) }.wrapOneDim(),
                biasList[3].apply { plusAssign(biasList[7]) }.wrapOneDim()
            )
        }

        fun createPeepholes(peepholes: MutableTypedNDArray<Any>): GatesData {
            require(peepholes.shape[0] == 1)

            val linear = peepholes.squeeze(0)

            val peepholesList = linear.splitWithAxis(3)
            return GatesData(peepholesList[0], peepholesList[1], peepholesList[2], allocateNDArray(DataType.INT16, Strides(intArrayOf(1))))
        }

        fun allocateGates(hiddenSize: Int, type: DataType): GatesData {
            val newStrides = Strides(intArrayOf(1, hiddenSize))

            val allocArrays = Array(4) {
                allocateNDArray<Any>(type, newStrides)
            }
            return GatesData(allocArrays[0], allocArrays[1], allocArrays[2], allocArrays[3])
        }
    }
}

data class State(var output: MutableTypedNDArray<Any>, val cellState: MutableTypedNDArray<Any>, var isOutputZero: Boolean, var isCellStateZero: Boolean) {
    companion object {
        fun allocateState(batchSize: Int, hiddenSize: Int, type: DataType): Array<State> {
            val newStrides = Strides(intArrayOf(1, hiddenSize))

            return Array(batchSize) {
                val out = allocateNDArray<Any>(type, newStrides)
                val cell = allocateNDArray<Any>(type, newStrides)
                State(out, cell, true, true)
            }
        }

        fun create(initialOutput: List<TypedNDArray<Any>>?, initialCellState: List<TypedNDArray<Any>>?, batchSize: Int, hiddenSize: Int, type: DataType): Array<State> {
            val allocatedStates = allocateState(batchSize, hiddenSize, type)

            if (initialOutput != null) {
                for (i in allocatedStates.indices) {
                    allocatedStates[i].output.placeAll(0, initialOutput[i])
                    allocatedStates[i].isOutputZero = false
                }
            }

            if (initialCellState != null) {
                for (i in allocatedStates.indices) {
                    allocatedStates[i].cellState.placeAll(0, initialCellState[i])
                    allocatedStates[i].isCellStateZero = false
                }
            }

            return allocatedStates
        }
    }
}
