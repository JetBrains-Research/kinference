package org.jetbrains.research.kotlin.inference.operators.layer.recurrent.lstm

import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.ndarray.allocateNDArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.splitArray
import org.jetbrains.research.kotlin.inference.onnx.TensorProto
import org.jetbrains.research.kotlin.inference.operators.layer.recurrent.RecurrentLayer

abstract class LSTMBase(hiddenSize: Int, activations: List<String>, direction: String) : RecurrentLayer(hiddenSize, activations, direction) {
    protected var weights: NDArray<Any>? = null
    protected var recurrentWeights: NDArray<Any>? = null
    protected var bias: NDArray<Any>? = null
    protected var peepholes: NDArray<Any>? = null
    protected var initialOutput: NDArray<Any>? = null
    protected var initialCellState: NDArray<Any>? = null

    protected var seqLength: Int? = null
    protected var batchSize: Int? = null
    protected var type: TensorProto.DataType? = null

    abstract fun apply(inputs: List<NDArray<Any>>, sequenceLens: IntArray, outputArray: NDArray<Any>, startOffset: Int): List<Tensor>

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
        if (sequenceLens != null) require(sequenceLens.data.type == TensorProto.DataType.INT32)

        val initialOutput = inputList.getOrNull(5)
        val initialCellState = inputList.getOrNull(6)
        val peepholes = inputList.getOrNull(7)

        parseTempInputs(weights, recurrentWeights, bias, initialOutput, initialCellState, peepholes)
        val outputShape = intArrayOf(seqLength!!, 1, batchSize!!, hiddenSize)
        if (direction == "bidirectional") outputShape[1] = 2
        val outputStrides = Strides(outputShape)
        val outputArray = allocateNDArray<Any>(type!!, outputStrides)
        return apply(parseInput(input), parseSequenceLens(sequenceLens), outputArray, 0)
    }

    private fun parseInput(input: Tensor): List<NDArray<Any>> =
        input.data.splitArray(input.data.shape[0] * input.data.shape[1], Strides(intArrayOf(1, input.data.shape[2])))

    private fun parseSequenceLens(input: Tensor?) = input?.data?.array as? IntArray ?: IntArray(batchSize!!) { seqLength!! }

    protected abstract fun parseTempInputs(weights: Tensor, recurrentWeights: Tensor, bias: Tensor?, initialOutput: Tensor?,
                                           initialCellState: Tensor?, peepholes: Tensor?)
}
