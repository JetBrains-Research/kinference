package org.jetbrains.research.kotlin.inference.operators.activations

import AttributeProto
import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.extensions.*
import org.jetbrains.research.kotlin.inference.operators.*
import scientifik.kmath.structures.*

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

    private fun expMatrixRows(input: Tensor, axis: Int): Array<NDBuffer<Any>> {
        val actualAxis = input.indexAxis(axis)
        val shape = input.data.shape
        val (rowIdx, columnIdx) = (shape.indices).partition { it < actualAxis }

        val rows = resolveDims(shape.slice(rowIdx))
        val columns = resolveDims(shape.slice(columnIdx))

        val matrixRows = input.reshape(intArrayOf(rows, columns)).bufferRows

        return Array(matrixRows.size) { i ->
            val max = matrixRows[i].max()!!
            matrixRows[i].minusScalar(max).exp()
        }
    }

    override fun activate(input: Tensor): Tensor {
        val axis = getAttributeValue("axis") as? Long
        val matrixRows = expMatrixRows(input, axis?.toInt() ?: 0)

        val buffer = allocateMutableBuffer(input.info.type, input.data.buffer.size)
        val step = matrixRows[0].buffer.size
        repeat(matrixRows.size) { i ->
            val sum = matrixRows[i].sum()
            val row = matrixRows[i].divScalar(sum)
            buffer.placeAll(row.buffer, i * step)
        }
        return Tensor("output", BufferNDStructure(input.data.strides, buffer as Buffer<Any>), input.info.type)
    }
}
