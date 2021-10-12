package io.kinference.core.operators.layer.normalization

import io.kinference.core.attributes.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.arrays.DoubleNDArray
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.pointers.acceptTriple
import io.kinference.ndarray.arrays.pointers.forEach
import io.kinference.ndarray.arrays.tiled.DoubleTiledArray
import io.kinference.ndarray.arrays.tiled.FloatTiledArray
import io.kinference.ndarray.extensions.indexAxis
import io.kinference.core.operators.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto.AttributeType
import io.kinference.protobuf.message.TensorProto
import kotlin.math.sqrt
import kotlin.time.ExperimentalTime

@ExperimentalTime
class LayerNormalization(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(INFO, attributes, inputs, outputs) {
    private val axis: Int by attribute { it: Number -> it.toInt() }
    private val epsilon: Float by attribute()

    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.FLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeType.INT), false, -1),
            AttributeInfo("epsilon", setOf(AttributeType.FLOAT), false, 0.00001f)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", false),
            IOInfo(1, TYPE_CONSTRAINTS, "scale", false),
            IOInfo(2, TYPE_CONSTRAINTS, "B", false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", false),
            //Unsupported
            IOInfo(1, TYPE_CONSTRAINTS, "mean", true),
            IOInfo(2, TYPE_CONSTRAINTS, "inv_std_var", true)
        )

        private val INFO = OperatorInfo("LayerNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO)
    }


    override fun apply(context: Context, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val input = inputs[0]!!.data
        val scale = inputs[1]!!.data
        val bias = inputs[2]!!.data

        require(input.type == scale.type && input.type == bias.type)
        val type = input.type

        val axis = input.indexAxis(axis)

        var normCount = 1
        var normSize = 1

        for (i in input.shape.indices) {
            if (i < axis) {
                normCount *= input.shape[i]
            } else {
                normSize *= input.shape[i]
            }

        }

        return when (type) {
            DataType.FLOAT -> {
                input as FloatNDArray; scale as FloatNDArray; bias as FloatNDArray

                val outputArray = FloatTiledArray(input.strides)
                for (i in 0 until normCount) {
                    val offset = i * normSize

                    var mean = 0.0f
                    var meanSquare = 0.0f
                    input.array.pointer(offset).forEach(normSize) {
                        mean += it
                        meanSquare += it * it
                    }

                    mean = mean / normSize
                    meanSquare = sqrt(meanSquare / normSize - mean * mean + epsilon)

                    val scalePointer = scale.array.pointer()
                    val biasPointer = bias.array.pointer()
                    val inputPointer = input.array.pointer(offset)

                    outputArray.pointer(offset).acceptTriple(inputPointer, scalePointer, biasPointer, normSize) { _, inp, sc, b ->
                        (inp - mean) / meanSquare * sc + b
                    }
                }
                listOf(
                    FloatNDArray(outputArray, input.strides).asTensor()
                )
            }
            DataType.DOUBLE -> {
                input as DoubleNDArray; scale as DoubleNDArray; bias as DoubleNDArray

                val outputArray = DoubleTiledArray(input.strides)
                for (i in 0 until normCount) {
                    val offset = i * normSize

                    var mean = 0.0
                    var meanSquare = 0.0
                    input.array.pointer(offset).forEach(normSize) {
                        mean += it
                        meanSquare += it * it
                    }

                    mean = mean / normSize
                    meanSquare = sqrt(meanSquare / normSize - mean * mean + epsilon)

                    val scalePointer = scale.array.pointer()
                    val biasPointer = bias.array.pointer()
                    val inputPointer = input.array.pointer(offset)

                    outputArray.pointer(offset).acceptTriple(inputPointer, scalePointer, biasPointer, normSize) { _, inp, sc, b ->
                        (inp - mean) / meanSquare * sc + b
                    }
                }
                listOf(
                    DoubleNDArray(outputArray, input.strides).asTensor()
                )
            }
            else -> throw UnsupportedOperationException()
        }
    }
}
