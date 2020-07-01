package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.*
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor
import java.math.BigDecimal

@Suppress("UNCHECKED_CAST")
abstract class Activation(name: String,
                          constraints: Set<TensorProto.DataType>,
                          attributes: Map<String, Attribute<Any>> = emptyMap(),
                          attributesInfo: Collection<AttributeInfo> = emptyList()
) : Operator<Tensor, Tensor>(name, attributes, attributesInfo,
    listOf(InputInfo(0, constraints, "input")),
    listOf(OutputInfo(0, constraints, "output"))) {

    abstract fun activate(input: Tensor): Tensor

    override fun apply(inputs: Collection<Tensor>, numOutputs: Int): Collection<Tensor> {
        return listOf(activate(inputs.first()))
    }

    companion object {
        fun <T : Number> max(x: Number, y: T): T {
            val a = BigDecimal(x.toString())
            val b = BigDecimal(y.toString())
            return a.max(b) as T
        }
    }
}
