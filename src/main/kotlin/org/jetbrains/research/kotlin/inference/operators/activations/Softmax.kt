package org.jetbrains.research.kotlin.inference.operators.activations

import AttributeProto
import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ndarray.NDArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.allocateNDArray
import org.jetbrains.research.kotlin.inference.extensions.ndarray.createScalarNDArray
import org.jetbrains.research.kotlin.inference.extensions.primitives.*
import org.jetbrains.research.kotlin.inference.operators.*

//only for float and double types
class Softmax(attributes: Map<String, Attribute<Any>>, usedOutputsNum: Int = 1) : Activation(INFO, attributes, usedOutputsNum) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = 1L)
        )

        private val INFO = OperatorInfo("Softmax", ATTRIBUTES_INFO,
            listOf(InputInfo(0, TYPE_CONSTRAINTS, "input", true)),
            listOf(OutputInfo(0, TYPE_CONSTRAINTS, "output"))
        )
    }

    private fun resolveDims(dims: List<Int>?): Int {
        return if (dims.isNullOrEmpty()) 1 else dims.reduce(Int::times)
    }

    private fun expMatrixRows(input: NDArray, axis: Int): Array<NDArray> {
        val actualAxis = input.indexAxis(axis)
        val shape = input.shape
        val (rowIdx, columnIdx) = (shape.indices).partition { it < actualAxis }

        val rows = resolveDims(shape.slice(rowIdx))
        val columns = resolveDims(shape.slice(columnIdx))

        val matrixRows = input.reshape(intArrayOf(rows, columns)).rows

        return Array(matrixRows.size) { i ->
            val max = matrixRows[i].max()!!
            matrixRows[i].minus(createScalarNDArray(input.type, max)).exp()
        }
    }

    override fun activate(input: NDArray): NDArray {
        val axis = getAttributeValue("axis") as? Long
        val matrixRows = expMatrixRows(input, axis?.toInt() ?: 0)

        val step = matrixRows[0].linearSize
        val array = allocateNDArray(input.type, input.strides)
        repeat(matrixRows.size) { i ->
            val sum = matrixRows[i].sum()
            val row = matrixRows[i].div(createScalarNDArray(input.type, sum))
            array.placeAll(i * step, row.array)
        }
        return array
    }
}
