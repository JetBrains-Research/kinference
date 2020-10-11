package io.kinference.operators.layer.recurrent.lstm

import io.kinference.ndarray.MutableNDArray
import io.kinference.ndarray.MutableNumberNDArray
import io.kinference.ndarray.NDArray
import io.kinference.ndarray.Strides
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.ndarray.extensions.splitWithAxis
import io.kinference.ndarray.extensions.squeeze
import io.kinference.ndarray.extensions.wrapOneDim
import io.kinference.primitives.types.DataType
import io.kinference.ndarray.*
import io.kinference.ndarray.extensions.*

@ExperimentalUnsignedTypes
class LSTMData(val type: DataType,
               val weights: GatesData,
               val recurrentWeights: GatesData,
               val bias: GatesData? = null,
               val initialOutput: List<NDArray>? = null,
               val initialCellState: List<NDArray>? = null,
               val peepholes: GatesData? = null
) {

    fun updateWeights(weights: GatesData) = LSTMData(type, weights, recurrentWeights, bias, initialOutput, initialCellState, peepholes)
    fun updateRecurrentWeights(recurrentWeights: GatesData) = LSTMData(type, weights, recurrentWeights, bias, initialOutput, initialCellState, peepholes)
    fun updateBias(bias: GatesData) = LSTMData(type, weights, recurrentWeights, bias, initialOutput, initialCellState, peepholes)
    fun updateInitialOutput(initialOutput: List<NDArray>) = LSTMData(type, weights, recurrentWeights, bias, initialOutput, initialCellState, peepholes)
    fun updateInitialCellGate(initialCellSate: List<NDArray>) = LSTMData(type, weights, recurrentWeights, bias, initialOutput, initialCellSate, peepholes)
    fun updatePeepholes(peepholes: GatesData) = LSTMData(type, weights, recurrentWeights, bias, initialOutput, initialCellState, peepholes)
}

@ExperimentalUnsignedTypes
data class GatesData(val input: MutableNDArray,
                     val output: MutableNDArray,
                     val forget: MutableNDArray,
                     val cellGate: MutableNDArray
) {
    fun cleanup() {
        input.clean()
        output.clean()
        forget.clean()
        cellGate.clean()
    }

    companion object {
        fun createWeights(weights: MutableNDArray): GatesData {
            require(weights.shape[0] == 1)

            val matrix = weights.squeeze(0)

            val weightsList = matrix.splitWithAxis(4)
            return GatesData(weightsList[0].transpose2D(), weightsList[1].transpose2D(),
                weightsList[2].transpose2D(), weightsList[3].transpose2D())
        }

        fun createBias(bias: MutableNDArray): GatesData {
            require(bias.shape[0] == 1)

            val linear = bias.squeeze(0)

            val biasList = linear.splitWithAxis(8)

            return GatesData(
                (biasList[0] as MutableNumberNDArray).apply { plusAssign(biasList[4]) }.wrapOneDim(),
                (biasList[1] as MutableNumberNDArray).apply { plusAssign(biasList[5]) }.wrapOneDim(),
                (biasList[2] as MutableNumberNDArray).apply { plusAssign(biasList[6]) }.wrapOneDim(),
                (biasList[3] as MutableNumberNDArray).apply { plusAssign(biasList[7]) }.wrapOneDim()
            )
        }

        fun createPeepholes(peepholes: MutableNDArray): GatesData {
            require(peepholes.shape[0] == 1)

            val linear = peepholes.squeeze(0)

            val peepholesList = linear.splitWithAxis(3)
            return GatesData(peepholesList[0], peepholesList[1], peepholesList[2], allocateNDArray(DataType.SHORT, Strides(intArrayOf(1))))
        }

        fun allocateGates(hiddenSize: Int, type: DataType): GatesData {
            val newStrides = Strides(intArrayOf(1, hiddenSize))

            val allocArrays = Array(4) { allocateNDArray(type, newStrides) }
            return GatesData(allocArrays[0], allocArrays[1], allocArrays[2], allocArrays[3])
        }
    }
}

@ExperimentalUnsignedTypes
data class State(var output: MutableNDArray, val cellState: MutableNDArray, var isOutputZero: Boolean, var isCellStateZero: Boolean) {
    companion object {
        fun allocateState(batchSize: Int, hiddenSize: Int, type: DataType): Array<State> {
            val newStrides = Strides(intArrayOf(1, hiddenSize))

            return Array(batchSize) {
                val out = allocateNDArray(type, newStrides)
                val cell = allocateNDArray(type, newStrides)
                State(out, cell, true, true)
            }
        }

        fun create(initialOutput: List<NDArray>?, initialCellState: List<NDArray>?, batchSize: Int, hiddenSize: Int, type: DataType): Array<State> {
            val allocatedStates = allocateState(batchSize, hiddenSize, type)

            if (initialOutput != null) {
                for (i in allocatedStates.indices) {
                    allocatedStates[i].output.copyFrom(0, initialOutput[i])
                    allocatedStates[i].isOutputZero = false
                }
            }

            if (initialCellState != null) {
                for (i in allocatedStates.indices) {
                    allocatedStates[i].cellState.copyFrom(0, initialCellState[i])
                    allocatedStates[i].isCellStateZero = false
                }
            }

            return allocatedStates
        }
    }
}
