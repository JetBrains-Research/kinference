package org.jetbrains.research.kotlin.mpp.inference.operators.tensor

import TensorProto
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.operators.*
import org.jetbrains.research.kotlin.mpp.inference.data.tensors.Tensor

class Reshape(attributes: Map<String, Attribute<Any>>) : Operator<Tensor, Tensor>("Reshape", attributes, emptyList(), INPUTS_INFO, OUTPUTS_INFO) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(
            InputInfo(0, TYPE_CONSTRAINTS, "data", true),
            InputInfo(1, setOf(TensorProto.DataType.INT64), "shape", true)
        )

        private val OUTPUTS_INFO = listOf(OutputInfo(0, TYPE_CONSTRAINTS, "reshaped"))
    }

    override fun apply(inputs: Collection<Tensor>, numOutputs: Int): Collection<Tensor> {
        return listOf(inputs.elementAt(0).reshape(inputs.elementAt(1)))
    }
}
