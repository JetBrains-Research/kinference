package io.kinference.webgpu.operators.logical

import io.kinference.attribute.Attribute
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.webgpu.ndarray.NDArrayInfo
import io.kinference.webgpu.ndarray.WebGPUDataType
import io.kinference.webgpu.operators.common.BroadcastingBinaryOperator

sealed class Less(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : BroadcastingBinaryOperator(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 7)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in LessVer7.VERSION.asRange() -> LessVer7(attributes, inputs, outputs)
            else -> error("Unsupported version of Less operator: $version")
        }
    }
}

class LessVer7(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Less(INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = PRIMITIVE_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.BOOL), "C", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 7)
        private val INFO = OperatorInfo("Less", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun operation(input0: String, input1: String, output: String): String = "$output = i32($input0 < $input1);"

    override fun outputType(inputInfo: List<NDArrayInfo?>): WebGPUDataType = WebGPUDataType.INT32
}
