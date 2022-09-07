package io.kinference.tfjs.operators.logical

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.extensions.equal
import io.kinference.ndarray.extensions.tidyNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class Equal(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 7)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in EqualVer7.VERSION.asRange() -> EqualVer7(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Equal operator: $version")
            }
    }
}

class EqualVer7(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Less(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = PRIMITIVE_DATA_TYPES + TensorProto.DataType.BFLOAT16

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.BOOL), "C", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 7)
        private val INFO = OperatorInfo("Equal", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

    }

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val output = tidyNDArray {
            val left = inputs[0]!!.data
            val right = inputs[1]!!.data
            return@tidyNDArray left.equal(right)
        }

        return listOf(output.asTensor("C"))
    }
}
