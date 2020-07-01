package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import AttributeProto
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.*
import org.jetbrains.research.kotlin.mpp.inference.operators.AttributeInfo
import scientifik.kmath.linear.VirtualMatrix
import scientifik.kmath.structures.*
import kotlin.math.exp

//only for float and double types
class Softmax(attributes: Map<String, Attribute<Any>>) : Activation("Softmax", TYPE_CONSTRAINTS, attributes, ATTRIBUTES_INFO) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = 1L)
        )
    }

    private fun resolveDims(dims: List<Int>?): Int {
        return if (dims.isNullOrEmpty()) 1 else dims.reduce(Int::times)
    }

    private fun castToExpMatrix(input: Tensor, axis: Int): VirtualMatrix<Double> {
        val actualAxis = input.indexAxis(axis)
        val shape = input.data.shape
        val (rowIdx, columnIdx) = (shape.indices).partition { it < actualAxis }

        val rows = resolveDims(shape.slice(rowIdx))
        val columns = resolveDims(shape.slice(columnIdx))

        val elements = (input.elementsList as List<Number>).toDoubleList()
        val localMax = elements.chunked(columns).map { it.max() ?: 0.0 }
        return VirtualMatrix(rowNum = rows, colNum = columns) { i, j -> exp((elements[i * columns + j] - localMax[i])) }
    }

    override fun activate(input: Tensor): Tensor {
        val axis = getAttributeValue("axis") as? Long
        val matrix = castToExpMatrix(input, axis?.toInt() ?: 0)
        val rows = matrix.rows.asIterable().toList()

        val rowSums = rows.map { it.array.sum() }
        val resultElements = rows.mapIndexed { i, buf -> buf.asSequence().map { it / rowSums[i] }.toList() }

        val buf = BufferNDStructure(TensorStrides(input.data.shape), resultElements.flatten().asBuffer()) as BufferNDStructure<Any>
        return Tensor("output", buf, input.info.type)
    }
}
