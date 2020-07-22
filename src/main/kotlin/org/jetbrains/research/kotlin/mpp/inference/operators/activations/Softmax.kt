package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import AttributeProto
import org.jetbrains.research.kotlin.mpp.inference.FloatBuffer
import org.jetbrains.research.kotlin.mpp.inference.arrayRows
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.TensorStrides
import org.jetbrains.research.kotlin.mpp.inference.operators.*
import scientifik.kmath.linear.BufferMatrix
import scientifik.kmath.structures.Buffer
import scientifik.kmath.structures.BufferNDStructure
import kotlin.collections.*
import kotlin.math.exp

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

    private fun expMatrixRows(input: Tensor, axis: Int): Array<FloatArray> {
        val actualAxis = input.indexAxis(axis)
        val shape = input.data.shape
        val (rowIdx, columnIdx) = (shape.indices).partition { it < actualAxis }

        val rows = resolveDims(shape.slice(rowIdx))
        val columns = resolveDims(shape.slice(columnIdx))

        val matrixRows = BufferMatrix(rows, columns, input.data.buffer as Buffer<out Float>).arrayRows
        val localMax = FloatArray(matrixRows.size) { i -> matrixRows[i].max() ?: 0.0f }

        return Array(matrixRows.size) { i ->
            FloatArray(columns) { j -> exp(matrixRows[i][j] - localMax[i]) }
        }
    }

    override fun activate(input: Tensor): Tensor {
        val axis = getAttributeValue("axis") as? Long
        val matrix = expMatrixRows(input, axis?.toInt() ?: 0)

        val rowSums = FloatArray(matrix.size) { i -> matrix[i].sum() }
        val resArray = Array(matrix.size) { i ->
            FloatArray(matrix[0].size) { j -> matrix[i][j] / rowSums[i] }
        }.reduce(FloatArray::plus)
        val buf = BufferNDStructure(TensorStrides(input.data.shape), FloatBuffer(resArray)) as BufferNDStructure<Any>
        return Tensor("output", buf, input.info.type)
    }
}
