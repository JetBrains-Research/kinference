package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.BooleanNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.utils.inlines.InlineInt

sealed class Dropout(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 12)  // last version. Other versions: 1, 6, 7, 10.

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in Dropout12.VERSION.asRange() -> Dropout12(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Dropout operator: $version")
            }
    }
}

class Dropout12(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Dropout(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS_T = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.BFLOAT16
        )

        private val TYPE_CONSTRAINTS_T1 = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE
        )

        private val TYPE_CONSTRAINTS_T2 = setOf(
            TensorProto.DataType.BOOL
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("seed", setOf(AttributeProto.AttributeType.INT), false, 1)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS_T, "data", optional = false, differentiable = true),
            IOInfo(1, TYPE_CONSTRAINTS_T1, "ratio", optional = true, differentiable = false),
            IOInfo(2, TYPE_CONSTRAINTS_T2, "training_mode", optional = true, differentiable = false),
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS_T, "output", optional = false, differentiable = true),
            IOInfo(1, TYPE_CONSTRAINTS_T2, "mask", optional = true, differentiable = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 12)
        private val INFO = OperatorInfo("Dropout", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val data = inputs[0]!!

        if (outputs.size == 1)
            return listOf(data.rename("output"))

        val mask = BooleanNDArray(data.data.shape) { _: InlineInt -> true }
        return listOf(data.rename("output"), mask.asTensor("mask"))
    }
}
