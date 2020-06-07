package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import org.jetbrains.research.kotlin.mpp.inference.operators.Operator
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import scientifik.kmath.structures.BufferNDStructure
import scientifik.kmath.structures.NDStructure
import java.math.BigDecimal

@Suppress("UNCHECKED_CAST")
abstract class Activation<T : Number> : Operator<T>() {
    abstract fun activate(input: Tensor<T>): Tensor<T>

    override fun apply(inputs: Collection<Tensor<T>>): Collection<Tensor<T>> {
        val toActivate = inputs.singleOrNull()
        requireNotNull(toActivate) { "Multiple inputs are not allowed" }

        return listOf(activate(toActivate))
    }

    companion object {
        fun <T : Number> max(x: Number, y: T): T {
            val a = BigDecimal(x.toString())
            val b = BigDecimal(y.toString())
            return a.max(b) as T
        }

        inline fun <T : Number, reified R : Number> reduceTensorAxis(tensor: NDStructure<T>, axis: Int, crossinline func: (Int, (Int) -> T) -> R): BufferNDStructure<R> {
            val shape = tensor.shape
            if (axis < 0 || axis >= shape.size)
                throw IllegalArgumentException("Illegal axis: $axis")
            val newShape = IntArray(shape.size) { if (it == axis) 1 else shape[it] }
            val index = IntArray(shape.size)
            return NDStructure.auto(newShape) { newIndex: IntArray ->
                for (i in index.indices) index[i] = newIndex[i]
                func(shape[axis]) {
                    index[axis] = it
                    tensor[index]
                }
            }
        }
    }
}
