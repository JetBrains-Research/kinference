package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.TensorStrides
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.as2DList
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.splitWithAxis
import scientifik.kmath.structures.BufferNDStructure
import scientifik.kmath.structures.VirtualBuffer
import scientifik.kmath.structures.get

class BiLSTMLayer<T : Number> : LSTMLayer<T>() {
    override fun apply(inputs: List<Tensor>): List<Tensor> {
        require(inputs.size in 3..4) { "Applicable only for three or four arguments" }

        val inputList = inputs.toList()

        val inputTensor = inputList[0]
        val (forwardWeights, backwardWeights) = inputList[1].splitWithAxis(2)
        val (forwardRecWeights, backwardRecWeights) = inputList[2].splitWithAxis(2)
        val (forwardBias, backwardBias) = inputList.getOrNull(3)?.splitWithAxis(2) ?: listOf(null, null)

        val inputMatrices = inputTensor.as2DList()

        val (mainForwardOutput, lastForwardState) =
            activate(inputMatrices, forwardWeights.squeeze(0), forwardRecWeights.squeeze(0), forwardBias)
        val (mainBackwardOutput, lastBackwardState) =
            activate(inputMatrices.asReversed(), backwardWeights.squeeze(0), backwardRecWeights.squeeze(0), backwardBias)

        val mainTensor = mainOutputHelper(mainForwardOutput, mainBackwardOutput)
        val (outputTensor, cellGateTensor) = stateOutputHelper(lastForwardState, lastBackwardState)

        return listOf(mainTensor, outputTensor, cellGateTensor)
    }

    @Suppress("UNCHECKED_CAST")
    private fun mainOutputHelper(mainForwardOutput: List<Tensor>, mainBackwardOutput: List<Tensor>): Tensor {
        val (batchSize, hiddenSize) = mainBackwardOutput.first().data.shape
        val mainOutputs = listOf(mainForwardOutput, mainBackwardOutput)

        val newShape = intArrayOf(mainForwardOutput.size, 2, batchSize, hiddenSize)
        val newStrides = TensorStrides(newShape)

        val newData = VirtualBuffer(newStrides.linearSize) { i ->
            val indices = newStrides.index(i)
            val (inputNum, numDirection, rowNum, colNum) = indices
            mainOutputs[numDirection][inputNum].data[rowNum, colNum]
        }
        val newBuffer = BufferNDStructure(newStrides, newData)
        return Tensor(null, newBuffer, mainForwardOutput.first().info.type)
    }

    @Suppress("UNCHECKED_CAST")
    private fun stateOutputHelper(lastForwardState: State, lastBackwardState: State): List<Tensor> {
        val (batchSize, hiddenSize) = lastForwardState.output.data.shape
        val type = lastForwardState.output.info.type

        val newShape = intArrayOf(2, batchSize, hiddenSize)
        val newStrides = TensorStrides(newShape)

        val lastOutputs = listOf(lastForwardState.output, lastBackwardState.output)
        val newOutputBuffer = extractActualStates(lastOutputs, newStrides)

        val lastCellGates = listOf(lastForwardState.cellGate, lastBackwardState.cellGate)
        val newCellGateBuffer = extractActualStates(lastCellGates, newStrides)

        val outputTensor = Tensor(null, newOutputBuffer, type)
        val cellGateTensor = Tensor(null, newCellGateBuffer, type)
        return listOf(outputTensor, cellGateTensor)
    }

    private fun extractActualStates(states: List<Tensor>, strides: TensorStrides): BufferNDStructure<Any> {
        val newOutputData = VirtualBuffer(strides.linearSize) { i ->
            val indices = strides.index(i)
            val (numDirection, rowNum, colNum) = indices
            states[numDirection].data[rowNum, colNum]
        }
        return BufferNDStructure(strides, newOutputData)
    }
}
