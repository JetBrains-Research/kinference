package org.jetbrains.research.kotlin.mpp.inference.operators.layer.reccurent.lstm

import org.jetbrains.research.kotlin.mpp.inference.tensors.*
import scientifik.kmath.structures.BufferNDStructure
import scientifik.kmath.structures.VirtualBuffer
import scientifik.kmath.structures.get

class BiLSTMLayer<T : Number> : LSTMLayer<T>() {
    override fun apply(inputs: Collection<Tensor>): Collection<Tensor> {
        require(inputs.size in 3..4) { "Applicable only for three or four arguments" }

        val inputList = inputs.toList()

        val inputTensor = inputList[0]
        val (forwardWeights, backwardWeights) = inputList[1].splitWithAxis(2)
        val (forwardRecWeights, backwardRecWeights) = inputList[2].splitWithAxis(2)
        val (forwardBias, backwardBias) = inputList.getOrNull(3)?.splitWithAxis(2) ?: listOf(null, null)

        val inputMatrices = inputTensor.as2DCollection().toList()

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
        return Tensor(null, newBuffer, mainForwardOutput.first().type)
    }

    @Suppress("UNCHECKED_CAST")
    private fun stateOutputHelper(lastForwardState: State<T>, lastBackwardState: State<T>): List<Tensor> {
        val (batchSize, hiddenSize) = lastForwardState.output.data.shape
        val type = lastForwardState.output.type

        val newShape = intArrayOf(2, batchSize, hiddenSize)
        val newStrides = TensorStrides(newShape)

        val lastOutputs = listOf(lastForwardState.output, lastBackwardState.output)
        val newOutputData = VirtualBuffer(newStrides.linearSize) { i ->
            val indices = newStrides.index(i)
            val (numDirection, rowNum, colNum) = indices
            lastOutputs[numDirection].data[rowNum, colNum]
        }
        val newOutputBuffer = BufferNDStructure(newStrides, newOutputData)

        val lastCellGates = listOf(lastForwardState.cellGate, lastBackwardState.cellGate)
        val newCellGatesData = VirtualBuffer(newStrides.linearSize) { i ->
            val indices = newStrides.index(i)
            val (numDirection, rowNum, colNum) = indices
            lastCellGates[numDirection].data[rowNum, colNum]
        }
        val newCellGateBuffer = BufferNDStructure(newStrides, newCellGatesData)

        val outputTensor = Tensor(null, newOutputBuffer, type)
        val cellGateTensor = Tensor(null, newCellGateBuffer, type)
        return listOf(outputTensor, cellGateTensor)
    }
}
