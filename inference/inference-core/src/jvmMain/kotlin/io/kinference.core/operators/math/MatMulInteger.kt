package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.optimizer.rules.context.MatMulIntegerContextRule
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.operator.*
import io.kinference.optimizer.GraphOptimizer.Companion.isOpt
import io.kinference.protobuf.message.TensorProto

sealed class MatMulInteger(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 10)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in MatMulIntegerVer10.VERSION.asRange() -> MatMulIntegerVer10(name, attributes, inputs, outputs)
            else -> error("Unsupported version of MatMulInteger operator: $version")
        }
    }
}


class MatMulIntegerVer10(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : MatMulInteger(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val IN_TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT8,
            TensorProto.DataType.INT8
        )

        private val OUT_TYPE_CONSTRAINTS = setOf(TensorProto.DataType.INT32)

        private val INPUTS_INFO = listOf(
            IOInfo(0, IN_TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, IN_TYPE_CONSTRAINTS, "B", optional = false),
            IOInfo(2, IN_TYPE_CONSTRAINTS, "a_zero_point", optional = true),
            IOInfo(3, IN_TYPE_CONSTRAINTS, "b_zero_point", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, OUT_TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 10)
        private val INFO = OperatorInfo("MatMulInteger", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }


    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val first = inputs[0]!!
        val second = inputs[1]!!
        val firstZero = inputs.getOrNull(2)
        val secondZero = inputs.getOrNull(3)

        val firstPrepared = first.takeIf { isOpt(it.name) }
            ?: MatMulIntegerContextRule.prepareTensor(first, firstZero)
        val secondPrepared = second.takeIf { isOpt(it.name) }
            ?: MatMulIntegerContextRule.prepareTensor(second, secondZero)

        val output = (firstPrepared.data as NumberNDArrayCore)
            .matmul(secondPrepared.data as NumberNDArrayCore)
        return listOf(output.asTensor("y"))
    }
}
