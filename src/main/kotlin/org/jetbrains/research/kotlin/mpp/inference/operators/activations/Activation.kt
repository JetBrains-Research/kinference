package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import org.jetbrains.research.kotlin.mpp.inference.operators.Operator
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import java.math.BigDecimal
import kotlin.math.exp

@Suppress("UNCHECKED_CAST")
sealed class Activation<T : Number>(private val func: (T) -> T) : Operator<T>() {
    override fun apply(inputs: Collection<Tensor<T>>): Collection<Tensor<T>> {
        val toActivate = inputs.singleOrNull()
        requireNotNull(toActivate) { "Multiple inputs are not allowed" }

        return listOf(toActivate.mapElements(func))
    }

    class Identity<T : Number> : Activation<T>(func = { x -> x })

    class Relu<T : Number> : Activation<T>(func = { x -> max(0, x) })

    //only for float and double types
    class Sigmoid<T : Number> : Activation<T>(func = { x -> (1.0 / (1.0 + exp(-x.toDouble()))) as T })

    //only for float and double types
    class Tanh<T : Number> : Activation<T>(func = { x ->
        ((exp(2.0 * x.toDouble()) - 1.0) /
            (exp(2.0 * x.toDouble()) + 1.0)) as T
    })

    companion object {
        private fun <T : Number> max(x: Number, y: T): T {
            val a = BigDecimal(x.toString())
            val b = BigDecimal(y.toString())
            return a.max(b) as T
        }
    }
}
