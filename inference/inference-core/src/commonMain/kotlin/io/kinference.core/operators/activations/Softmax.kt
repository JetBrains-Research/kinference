package io.kinference.core.operators.activations

import io.kinference.attribute.Attribute
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.ndarray.runBlocking
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto
import kotlinx.coroutines.*
import kotlin.math.exp
import kotlin.math.min
import kotlin.time.ExperimentalTime

sealed class Softmax(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Activation(info, attributes, inputs, outputs) {
    companion object {
        private fun resolveDims(dims: IntArray?): Int {
            return if (dims == null || dims.isEmpty()) 1 else dims.reduce(Int::times)
        }

        fun exp(type: DataType) = when (type) {
            DataType.FLOAT -> object : FloatMap {
                override fun apply(value: Float): Float = exp(value)
            }
            DataType.DOUBLE -> object : DoubleMap {
                override fun apply(value: Double): Double = exp(value)
            }
            else -> error("Unsupported data type: $type")
        }

        fun softmax(input: NDArray, axis: Int = 0, strides: Strides = input.strides): MutableNDArray {
            val actualAxis = input.indexAxis(axis)
            val shape = input.shape
            val (rowIdx, columnIdx) = (shape.indices).partition { it < actualAxis }

            val rows = resolveDims(shape.sliceArray(rowIdx))
            val columns = resolveDims(shape.sliceArray(columnIdx))

            val matrixRows = (input.reshape(intArrayOf(rows, columns)) as NumberNDArray).rows

            fun MutableNumberNDArray.softmax() {
                minusAssign(createScalarNDArray(input.type, max()))
                mapMutable(exp(type))
                divAssign(createScalarNDArray(input.type, sum()))
            }

            if (matrixRows.size > 128) {
                runBlocking(Dispatchers.Default) {
                    for (i in matrixRows.indices step 32) {
                        val end = min(i + 32, matrixRows.size)
                        launch {
                            for (row in i until end) {
                                matrixRows[row].softmax()
                            }
                        }
                    }
                }
            } else {
                for (i in matrixRows.indices) {
                    matrixRows[i].softmax()
                }
            }

            val step = matrixRows[0].linearSize
            val array = allocateNDArray(input.type, strides)
            repeat(matrixRows.size) { i ->
                array.copyFrom(i * step, matrixRows[i])
            }
            return array
        }

        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in SoftmaxVer1.VERSION.asRange() -> SoftmaxVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of Softmax operator: $version")
        }
    }
}

//only for float and double types
@ExperimentalTime
class SoftmaxVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Softmax(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = 1)
        )

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)
        private val INFO = OperatorInfo("Softmax", ATTRIBUTES_INFO, INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }

    override fun activate(input: NDArray): NDArray {
        return softmax(input, axis, input.strides)
    }
}
