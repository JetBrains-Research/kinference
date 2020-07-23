package org.jetbrains.research.kotlin.mpp.inference.operators.activations

import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.operators.Operator
import org.jetbrains.research.kotlin.mpp.inference.operators.OperatorInfo

@Suppress("UNCHECKED_CAST")
abstract class Activation(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, usedOutputsNum: Int)
    : Operator<Tensor, Tensor>(info, usedOutputsNum, attributes) {

    abstract fun activate(input: Tensor): Tensor

    override fun apply(inputs: List<Tensor>): List<Tensor> {
        return listOf(activate(inputs.first()))
    }
}
