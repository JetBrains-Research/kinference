package io.kinference.tfjs.operators.math

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import kotlin.time.ExperimentalTime

sealed class FusedMatMul(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in FusedMatMulVer1.VERSION.asRange() -> FusedMatMulVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of FusedMatMul operator: $version")
            }
    }
}

@ExperimentalTime
class FusedMatMulVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    FusedMatMul(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.BFLOAT16
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("alpha", setOf(AttributeProto.AttributeType.FLOAT), required = true),
            AttributeInfo("transA", setOf(AttributeProto.AttributeType.INT), required = true),
            AttributeInfo("transB", setOf(AttributeProto.AttributeType.INT), required = true),
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("FusedMatMul", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")
    }

    private val alpha: Float by attribute()

    private val transposeLeft: Boolean by attribute("transA") { it: Number -> it.toInt() == 1 }
    private val transposeRight: Boolean by attribute("transB") { it: Number -> it.toInt() == 1 }

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val left = inputs[0]!!.data as NumberNDArrayTFJS
        val right = inputs[1]!!.data as NumberNDArrayTFJS

        val output = tidyNDArray {
            val matMulResult = left.matmul(right, transposeLeft, transposeRight)
            val scalar = NDArrayTFJS.floatScalar(alpha)
            return@tidyNDArray matMulResult * scalar
        }
        return listOf(output.asTensor("Y"))
    }
}
