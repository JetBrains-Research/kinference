package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import AttributeProto
import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.AttributeInfo
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.tensors.TensorStrides
import scientifik.kmath.linear.VirtualMatrix
import scientifik.kmath.structures.*
import kotlin.math.exp

//only for float and double types
class Softmax(attributes: Map<String, Attribute<Any>>) : Activation("Softmax", TYPE_CONSTRAINTS, attributes, ATTRIBUTES_INFO) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.BFLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = 1L)
        )
    }

    private fun resolveDims(dims: List<Int>?): Int {
        return if (dims.isNullOrEmpty()) 1 else dims.reduce(Int::times)
    }

    private fun castToExpMatrix(input: Tensor, axis: Long): VirtualMatrix<Double> {
        val actualAxis = when {
            axis >= 0 -> axis
            else -> input.rank + axis
        }

        val (rowIdx, columnIdx) = (input.data.shape.indices).partition { it < actualAxis }
        val rows = resolveDims(input.data.shape.slice(rowIdx))
        val columns = resolveDims(input.data.shape.slice(columnIdx))

        val localMax = input.elementsList.chunked(columns).map { (it as? List<Double>)?.max() ?: 0.0 }
        return VirtualMatrix(rowNum = rows, colNum = columns) { i, j ->
            exp(((input.elementsList[i * columns + j] as Number).toDouble() - localMax[i]))
        }
    }

    override fun activate(input: Tensor): Tensor {
        val matrix = castToExpMatrix(input, getAttributeValue("axis") as Long)
        val rows = matrix.rows.asIterable().toList()

        val rowSums = rows.map { it.array.sum() }.toList()
        val resultElements = rows.mapIndexed { i, buf -> buf.asSequence().map { it / rowSums[i] }.toList() }

        val buf = BufferNDStructure(TensorStrides(input.data.shape), resultElements.flatten().asBuffer()) as BufferNDStructure<Any>
        return Tensor("output", buf, input.type)
    }
}
