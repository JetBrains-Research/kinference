package io.kinference.operators.layer.recurrent.lstm

import io.kinference.ndarray.Strides
import io.kinference.primitives.types.DataType
import io.kinference.data.tensors.Tensor
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.ndarray.extensions.splitParts
import io.kinference.operators.layer.recurrent.RecurrentLayer

abstract class LSTMBase(hiddenSize: Int, activations: List<String>, direction: String) : RecurrentLayer(hiddenSize, activations, direction) {
    protected var weights: NDArray? = null
    protected var recurrentWeights: NDArray? = null
    protected var bias: NDArray? = null
    protected var peepholes: NDArray? = null
    protected var initialOutput: NDArray? = null
    protected var initialCellState: NDArray? = null

    protected var seqLength: Int? = null
    protected var batchSize: Int? = null
    protected var type: DataType? = null

    abstract fun apply(inputs: List<NDArray>, sequenceLens: IntArray, outputArray: MutableNDArray, startOffset: Int): List<Tensor>

    @ExperimentalUnsignedTypes
    override fun apply(inputList: List<Tensor?>): List<Tensor?> {
        require(inputList.toMutableList().also { if (4 in it.indices) it.removeAt(4) }.all { it?.data?.type == inputList[0]!!.data.type })

        val input = inputList[0]!!

        seqLength = input.data.shape[0]
        batchSize = input.data.shape[1]
        type = input.data.type

        val weights = inputList[1]!!
        val recurrentWeights = inputList[2]!!
        val bias = inputList.getOrNull(3)

        val sequenceLens = inputList.getOrNull(4)
        if (sequenceLens != null) require(sequenceLens.data.type == DataType.INT)

        val initialOutput = inputList.getOrNull(5)
        val initialCellState = inputList.getOrNull(6)
        val peepholes = inputList.getOrNull(7)

        parseTempInputs(weights, recurrentWeights, bias, initialOutput, initialCellState, peepholes)
        val outputShape = intArrayOf(seqLength!!, 1, batchSize!!, hiddenSize)
        if (direction == "bidirectional") outputShape[1] = 2
        val outputStrides = Strides(outputShape)
        val outputArray = allocateNDArray(type!!, outputStrides)
        return apply(parseInput(input), parseSequenceLength(sequenceLens), outputArray, 0)
    }

    private fun parseInput(input: Tensor): List<MutableNDArray> =
        input.data.splitParts(input.data.shape[0] * input.data.shape[1], Strides(intArrayOf(1, input.data.shape[2])))

    @ExperimentalUnsignedTypes
    private fun parseSequenceLength(input: Tensor?): IntArray {
        return if (input?.data == null) {
            IntArray(batchSize!!) { seqLength!! }
        } else {
            val pointer = (input.data as IntNDArray).array.pointer()
            IntArray(input.data.linearSize) { pointer.getAndIncrement() }
        }

    }

    protected abstract fun parseTempInputs(weights: Tensor, recurrentWeights: Tensor, bias: Tensor?, initialOutput: Tensor?,
                                           initialCellState: Tensor?, peepholes: Tensor?)
}
