package io.kinference.operators.layer.attention

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.data.tensors.asTensor
import io.kinference.graph.Context
import io.kinference.graph.ProfilingContext
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.ndarray.runBlocking
import io.kinference.onnx.AttributeProto
import io.kinference.onnx.TensorProto
import io.kinference.operators.*
import io.kinference.primitives.types.DataType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlin.time.ExperimentalTime

@ExperimentalTime
class QAttention(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<Tensor, Tensor>(INFO, attributes, inputs, outputs) {

    companion object {
        private val FLOATS = setOf(TensorProto.DataType.FLOAT, TensorProto.DataType.FLOAT16)
        private val BYTES = setOf(TensorProto.DataType.INT8, TensorProto.DataType.UINT8)
        private val DEFAULT_SCALE = FloatNDArray.scalar(1f)

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("num_heads", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("unidirectional", setOf(AttributeProto.AttributeType.INT), false, default = 0)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, BYTES, "input", optional = false),
            IOInfo(1, BYTES, "weight", optional = false, divider = 3),
            IOInfo(2, FLOATS, "bias", optional = false, divider = 3),
            IOInfo(3, FLOATS, "input_scale", optional = false),
            IOInfo(4, FLOATS, "weight_scale", optional = false),
            IOInfo(5, setOf(TensorProto.DataType.INT32), "mask_index", optional = true),
            IOInfo(6, BYTES, "input_zero_point", optional = true),
            IOInfo(7, BYTES, "weight_zero_point", optional = true),
            IOInfo(8, FLOATS, "past", optional = true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, FLOATS, "output", optional = false),
            IOInfo(1, FLOATS, "present", optional = true)
        )

        private val INFO = OperatorInfo("QAttention", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }

    private val numHeads: Int by attribute("num_heads") { it: Number -> it.toInt() }
    private val unidir: Boolean by attribute("unidirectional") { it: Number -> it.toInt() == 1 }

    private fun initQueryKeyValue(input: NumberNDArray, weights: NumberNDArray, bias: FloatNDArray,
                                  batchSize: Int, seqLen: Int, hiddenSize: Int, numHeads: Int,
                                  inputZeroPoint: Int, weightsZeroPoint: Int, deqScale: Float): Array<MutableNDArray> {
        val headSize = hiddenSize / numHeads
        val qkvWeights = weights.splitHorizontalByBlocks(3).map { it.splitHorizontalByBlocks(numHeads) }
        val qkvBias = bias.splitHorizontalByBlocks(3).map { it.splitHorizontalByBlocks(numHeads) }

        val qkv = Array(3) { allocateNDArray(DataType.FLOAT, Strides(intArrayOf(batchSize, numHeads, seqLen, headSize))) }

        runBlocking(Dispatchers.Default) {
            for (qkvIdx in 0 until 3) {
                launch {
                    val output = qkv[qkvIdx]
                    val weights = qkvWeights[qkvIdx]
                    val bias = qkvBias[qkvIdx]

                    for (batchNum in 0 until batchSize) {
                        val inputMatrix = input.view(batchNum)
                        for (numHead in 0 until numHeads) {
                            val outputMatrix = output.viewMutable(batchNum, numHead) as MutableFloatNDArray
                            val weightsMatrix = weights[numHead]
                            val biasMatrix = bias[numHead] as FloatNDArray

                            when {
                                inputMatrix is ByteNDArray && weightsMatrix is ByteNDArray -> inputMatrix.quantizeDot(weightsMatrix, outputMatrix, inputZeroPoint, weightsZeroPoint, deqScale)
                                inputMatrix is ByteNDArray && weightsMatrix is UByteNDArray -> inputMatrix.quantizeDot(weightsMatrix, outputMatrix, inputZeroPoint, weightsZeroPoint, deqScale)
                                inputMatrix is UByteNDArray && weightsMatrix is ByteNDArray -> inputMatrix.quantizeDot(weightsMatrix, outputMatrix, inputZeroPoint, weightsZeroPoint, deqScale)
                                inputMatrix is UByteNDArray && weightsMatrix is UByteNDArray -> inputMatrix.quantizeDot(weightsMatrix, outputMatrix, inputZeroPoint, weightsZeroPoint, deqScale)
                            }
                            outputMatrix.plusAssign(biasMatrix)
                        }
                    }
                }
            }
        }

        return qkv
    }

    override fun apply(context: Context, inputs: List<Tensor?>, profilingContext: ProfilingContext?): List<Tensor?> {
        val input = inputs[0]!!.data as NumberNDArray
        val weights = inputs[1]!!.data as NumberNDArray

        val inputScale = inputs[3]!!.data.singleValue() as Float
        val weightsScale = inputs[4]!!.data.singleValue() as Float
        val deqScale = inputScale * weightsScale

        val inputZeroPointNumber = inputs.getOrNull(6)?.data?.singleValue()
        val weightsZeroPointNumber = inputs.getOrNull(7)?.data?.singleValue()

        val inputZeroPoint = when(inputZeroPointNumber) {
            is UByte -> inputZeroPointNumber.toInt()
            is Byte -> inputZeroPointNumber.toInt()
            else -> 0
        }

        val weightsZeroPoint = when(weightsZeroPointNumber) {
            is UByte -> weightsZeroPointNumber.toInt()
            is Byte -> weightsZeroPointNumber.toInt()
            else -> 0
        }


        val (batchSize, seqLen, hiddenSize) = input.shape
        val bias = inputs[2]!!.data as FloatNDArray


        val (queries, keys, values) = initQueryKeyValue(input, weights, bias, batchSize, seqLen, hiddenSize, numHeads, inputZeroPoint, weightsZeroPoint, deqScale)

        val maskIndices = inputs.elementAtOrNull(5)?.data as IntNDArray?
        val past = inputs.elementAtOrNull(8)?.data
        val (scores, present) = Attention.getScores(unidir, queries, keys, values, maskIndices, past, batchSize, seqLen, numHeads, hiddenSize)
        return listOf(scores.asTensor(), present.asTensor())
    }
}
