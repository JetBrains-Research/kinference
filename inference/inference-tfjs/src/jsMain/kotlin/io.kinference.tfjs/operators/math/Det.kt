package io.kinference.tfjs.operators.math

import io.kinference.attribute.Attribute
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.det
import io.kinference.operator.*
import io.kinference.tfjs.data.tensors.asTensor

sealed class Det(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): Det {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in DetVer11.VERSION.asRange() -> DetVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Det operator: $version")
            }
        }
    }
}


class DetVer11(
    name: String,
    attributes:
    Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Det(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = PRIMITIVE_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", differentiable = true, optional = false),
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", differentiable = true, optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("Det", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        return listOf((inputs[0]!!.data as NumberNDArrayTFJS).det().asTensor("Y"))
    }
}

