/*
package org.jetbrains.research.kotlin.inference.operators.layer.recurrent.lstm

import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*

class BiLSTMLayer<T : Number> : LSTMLayer<T>() {
    override fun apply(inputs: List<Tensor>): List<Tensor> {
        require(inputs.size in 3..4) { "Applicable only for three or four arguments" }

        val inputList = inputs.toList()

        val inputTensor = inputList[0]
        val (forwardWeights, backwardWeights) = inputList[1].data.splitWithAxis(2)
        val (forwardRecWeights, backwardRecWeights) = inputList[2].data.splitWithAxis(2)
        val (forwardBias, backwardBias) = inputList.getOrNull(3)?.data?.splitWithAxis(2) ?: listOf(null, null)

        val inputMatrices = inputTensor.data.as2DList()

        val (mainForwardOutput, lastForwardState) =
            activate(inputMatrices, forwardWeights.squeeze(0), forwardRecWeights.squeeze(0), forwardBias)
        val (mainBackwardOutput, lastBackwardState) =
            activate(inputMatrices.asReversed(), backwardWeights.squeeze(0), backwardRecWeights.squeeze(0), backwardBias)

        val mainTensor = mainOutputHelper(mainForwardOutput, mainBackwardOutput)
        val (outputTensor, cellGateTensor) = stateOutputHelper(lastForwardState, lastBackwardState)

        return listOf(mainTensor, outputTensor, cellGateTensor)
    }

    @Suppress("UNCHECKED_CAST")
    private fun mainOutputHelper(mainForwardOutput: List<NDArray<Any>>, mainBackwardOutput: List<NDArray<Any>>): Tensor {
        val (batchSize, hiddenSize) = mainBackwardOutput.first().shape
        val mainOutputs = listOf(mainForwardOutput, mainBackwardOutput.reversed())

        val newShape = intArrayOf(mainForwardOutput.size, 2, batchSize, hiddenSize)
        val newStrides = Strides(newShape)

        val newData = createArray(mainForwardOutput.first().type, newStrides.linearSize) { i ->
            val indices = newStrides.index(i)
            val (inputNum, numDirection, rowNum, colNum) = indices
            mainOutputs[numDirection][inputNum].get(intArrayOf(rowNum, colNum))
        }
        return NDArray(newData, mainForwardOutput.first().type, newStrides).asTensor()
    }

    @Suppress("UNCHECKED_CAST")
    private fun stateOutputHelper(lastForwardState: State, lastBackwardState: State): List<Tensor> {
        val (batchSize, hiddenSize) = lastForwardState.output.shape

        val newShape = intArrayOf(2, batchSize, hiddenSize)
        val newStrides = Strides(newShape)

        val lastOutputs = listOf(lastForwardState.output, lastBackwardState.output)
        val newOutputBuffer = extractActualStates(lastOutputs, newStrides)

        val lastCellGates = listOf(lastForwardState.cellGate, lastBackwardState.cellGate)
        val newCellGateBuffer = extractActualStates(lastCellGates, newStrides)

        return listOf(newOutputBuffer.asTensor(), newCellGateBuffer.asTensor())
    }

    private fun extractActualStates(states: List<NDArray<Any>>, strides: Strides): NDArray<Any> {
        val array = createArray(states.first().type, strides.linearSize) { i ->
            val indices = strides.index(i)
            val (numDirection, rowNum, colNum) = indices
            states[numDirection].get(intArrayOf(rowNum, colNum))
        }
        return NDArray(array, states.first().type, strides)
    }
}
*/
