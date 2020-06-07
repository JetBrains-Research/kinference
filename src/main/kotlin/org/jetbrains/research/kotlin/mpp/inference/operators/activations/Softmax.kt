package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import org.jetbrains.research.kotlin.mpp.inference.space.SpaceStrides
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import scientifik.kmath.linear.VirtualMatrix
import scientifik.kmath.structures.*
import kotlin.math.exp

//only for float and double types
class Softmax<T : Number>(private val axis: Long?) : Activation<T>() {
    private fun resolveDims(dims: List<Int>?): Int {
        return if (dims.isNullOrEmpty()) 1 else dims.reduce(Int::times)
    }

    private fun castToExpMatrix(input: Tensor<T>): VirtualMatrix<Double> {
        val actualAxis = when {
            axis == null -> 1
            axis >= 0 -> axis
            else -> input.rank + axis
        }

        val (rowIdx, columnIdx) = (input.data.shape.indices).partition { it < actualAxis }
        val rows = resolveDims(input.data.shape.slice(rowIdx))
        val columns = resolveDims(input.data.shape.slice(columnIdx))

        val localMax = input.elementsList.chunked(columns).map { (it as? List<Double>)?.max() ?: 0.0 }
        return VirtualMatrix(rowNum = rows, colNum = columns) { i, j ->
            exp((input.elementsList[i * columns + j].toDouble() - localMax[i]))
        }
    }

    override fun activate(input: Tensor<T>): Tensor<T> {
        val matrix = castToExpMatrix(input)
        val rows = matrix.rows.asIterable().asIterable().toList()

        val rowSums = rows.map { it.array.sum() }.toList()
        val resultElements = rows.mapIndexed { i, buf -> buf.asSequence().map { it / rowSums[i] }.toList() }

        val buf = BufferNDStructure(SpaceStrides(input.space!!.shape), resultElements.flatten().asBuffer()) as BufferNDStructure<T>
        return Tensor("output", buf, input.type, input.space)
    }
}
