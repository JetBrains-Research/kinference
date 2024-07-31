package io.kinference.core.operators.activations

import io.kinference.attribute.Attribute
import io.kinference.core.KIONNXData
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayCore
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.operator.*

sealed class Identity(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<KIONNXData<*>, KIONNXData<*>>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in IdentityVer1.VERSION.asRange() -> IdentityVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Identity operator: $version")
        }
    }
}


class IdentityVer1 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>> = emptyMap(),
    inputs: List<String>,
    outputs: List<String>
) : Identity(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val INPUT_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "input", optional = false, onnxDataTypes = setOf(ONNXDataType.ONNX_SEQUENCE, ONNXDataType.ONNX_TENSOR))
        )
        private val OUTPUT_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "output", optional = false, onnxDataTypes = setOf(ONNXDataType.ONNX_SEQUENCE, ONNXDataType.ONNX_TENSOR))
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("Identity", emptyMap(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KIONNXData<*>?>): List<KIONNXData<*>?> {
        return listOf(inputs[0]!!.clone("output"))
    }
}
