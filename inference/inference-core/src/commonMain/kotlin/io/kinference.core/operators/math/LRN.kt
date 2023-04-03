package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.BooleanNDArray
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.extensions.lrn
import io.kinference.ndarray.extensions.utils.calculateInnerShapeSize
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.toIntArray
import kotlin.time.ExperimentalTime

sealed class LRN(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 13)  // last version. Other versions: 1.

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in LRN13.VERSION.asRange() -> LRN13(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Greater operator: $version")
            }
    }
}

@ExperimentalTime
class LRN13(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    LRN(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.BFLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("alpha", setOf(AttributeProto.AttributeType.FLOAT), false, 0.0001f),
            AttributeInfo("beta", setOf(AttributeProto.AttributeType.FLOAT), false, 0.75f),
            AttributeInfo("bias", setOf(AttributeProto.AttributeType.FLOAT), false, 1.0f),
            AttributeInfo("size", setOf(AttributeProto.AttributeType.INT), true),
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false, differentiable = true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false, differentiable = true)
        )

        internal val VERSION = VersionInfo(sinceVersion = 13)
        private val INFO = OperatorInfo("LRN", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val size: Int by attribute("size") { it: Number -> it.toInt() }
    private val alpha: Float by attribute("alpha")
    private val beta: Float by attribute("beta")
    private val bias: Float by attribute("bias")

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val x = inputs[0]!!.data

        x as FloatNDArray
        val y = x.lrn(alpha, beta, bias, size)

        return listOf(y.asTensor("y"))
    }
}
