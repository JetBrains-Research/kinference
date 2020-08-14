package org.jetbrains.research.kotlin.inference.operators.activations

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ndarray.*
import org.jetbrains.research.kotlin.inference.data.tensors.Strides
import org.jetbrains.research.kotlin.inference.extensions.ndarray.*
import org.jetbrains.research.kotlin.inference.extensions.primitives.exp
import org.jetbrains.research.kotlin.inference.extensions.primitives.max
import org.jetbrains.research.kotlin.inference.extensions.primitives.sum
import org.jetbrains.research.kotlin.inference.onnx.AttributeProto
import org.jetbrains.research.kotlin.inference.operators.AttributeInfo
import org.jetbrains.research.kotlin.inference.operators.IOInfo
import org.jetbrains.research.kotlin.inference.operators.OperatorInfo

//only for float and double types
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

        private fun expMatrixRows(input: MutableTypedNDArray<Any>, axis: Int): Array<MutableTypedNDArray<Any>> {
            val actualAxis = input.indexAxis(axis)
            val shape = input.shape
            val (rowIdx, columnIdx) = (shape.indices).partition { it < actualAxis }

            val rows = resolveDims(shape.sliceArray(rowIdx))
            val columns = resolveDims(shape.sliceArray(columnIdx))

            val matrixRows = input.reshape(intArrayOf(rows, columns)).rows.map { it.toMutable() }
            return Array(matrixRows.size) { i ->
                val max = matrixRows[i].max()!!
                matrixRows[i] -= createScalarNDArray(input.type, max)
                matrixRows[i].exp()
            }
        }

        fun softmax(input: MutableTypedNDArray<Any>, axis: Int = 0, strides: Strides = input.strides): MutableTypedNDArray<Any> {
            val matrixRows = expMatrixRows(input, axis)

            val step = matrixRows[0].linearSize
            val array = allocateNDArray<Any>(input.type, strides)
            repeat(matrixRows.size) { i ->
                val sum = matrixRows[i].sum()
                matrixRows[i].divAssign(createScalarNDArray(input.type, sum))
                array.placeAll(i * step, matrixRows[i].array)
            }
            return array
        }
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }

    override fun activate(input: TypedNDArray<Any>): TypedNDArray<Any> {
        return softmax(input.toMutable(), axis, input.strides)
    }
}
