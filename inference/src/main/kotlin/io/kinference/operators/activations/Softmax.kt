package io.kinference.operators.activations

import io.kinference.ndarray.MutableNDArray
import io.kinference.ndarray.MutableNumberNDArray
import io.kinference.ndarray.NDArray
import io.kinference.ndarray.Strides
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.ndarray.extensions.createScalarNDArray
import io.kinference.ndarray.extensions.indexAxis
import io.kinference.ndarray.extensions.rows
import io.kinference.primitives.types.DataType
import io.kinference.attributes.Attribute
import io.kinference.ndarray.*
import io.kinference.ndarray.extensions.*
import io.kinference.onnx.AttributeProto
import io.kinference.operators.AttributeInfo
import io.kinference.operators.IOInfo
import io.kinference.operators.OperatorInfo
import kotlin.math.exp

//only for float and double types
@ExperimentalUnsignedTypes
class Softmax(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Activation(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = 1L)
        )

        private val INFO = OperatorInfo("Softmax", ATTRIBUTES_INFO,
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false)),
            listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))
        )

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
            else -> error("Unsupported data type")
        }

        private fun expMatrixRows(input: MutableNDArray, axis: Int): Array<MutableNumberNDArray> {
            val actualAxis = input.indexAxis(axis)
            val shape = input.shape
            val (rowIdx, columnIdx) = (shape.indices).partition { it < actualAxis }

            val rows = resolveDims(shape.sliceArray(rowIdx))
            val columns = resolveDims(shape.sliceArray(columnIdx))

            val matrixRows = input.reshape(intArrayOf(rows, columns)).rows
            return Array(matrixRows.size) { i ->
                (matrixRows[i] as MutableNumberNDArray).apply {
                    val max = createScalarNDArray(input.type, this.max())
                    minusAssign(max)
                    mapMutable(exp(type))
                }
            }
        }

        fun softmax(input: MutableNDArray, axis: Int = 0, strides: Strides = input.strides): MutableNDArray {
            val matrixRows = expMatrixRows(input, axis)

            val step = matrixRows[0].linearSize
            val array = allocateNDArray(input.type, strides)
            repeat(matrixRows.size) { i ->
                val sum = matrixRows[i].sum()
                matrixRows[i].divAssign(createScalarNDArray(input.type, sum))
                array.placeAllFrom(i * step, matrixRows[i])
            }
            return array
        }
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }

    override fun activate(input: NDArray): NDArray {
        return softmax(input.toMutable(), axis, input.strides)
    }
}
