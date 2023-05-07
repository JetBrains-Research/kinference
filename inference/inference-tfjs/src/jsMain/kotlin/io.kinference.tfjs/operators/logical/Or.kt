package io.kinference.tfjs.operators.logical

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.BooleanNDArrayTFJS
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class Or(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 7)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): Or {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in OrVer7.VERSION.asRange() -> OrVer7(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Or operator: $version")
            }
        }
    }
}

class OrVer7(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
): Or(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(TensorProto.DataType.BOOL)

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "C", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 7)
        private val INFO = OperatorInfo("Or", emptySet(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val left = inputs[0]!!.data as BooleanNDArrayTFJS
        val right = inputs[1]!!.data as BooleanNDArrayTFJS

        val result = left.or(right)
        return listOf(result.asTensor("C"))
    }
}
