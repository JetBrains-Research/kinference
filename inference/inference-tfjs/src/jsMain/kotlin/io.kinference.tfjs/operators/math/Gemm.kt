package io.kinference.tfjs.operators.math

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class Gemm(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>, outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Gemm {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in GemmVer11.VERSION.asRange() -> GemmVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Gemm operator: $version")
            }
        }
    }
}

class GemmVer11(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Gemm(name, INFO, attributes, inputs, outputs) {

    private val alpha: Double by attribute { it: Number -> it.toDouble() }
    private val beta: Double by attribute { it: Number -> it.toDouble() }

    private val transA: Boolean by attribute { it: Number -> it.toInt() != 0 }
    private val transB: Boolean by attribute { it: Number -> it.toInt() != 0 }

    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT16,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
            TensorProto.DataType.BFLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("alpha", setOf(AttributeProto.AttributeType.FLOAT), false, 1.0),
            AttributeInfo("beta", setOf(AttributeProto.AttributeType.FLOAT), false, 1.0),
            AttributeInfo("transA", setOf(AttributeProto.AttributeType.INT), false, 0),
            AttributeInfo("transB", setOf(AttributeProto.AttributeType.INT), false, 0)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false),
            IOInfo(2, TYPE_CONSTRAINTS, "C", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("Gemm", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val a = (inputs[0]!!.data as NumberNDArrayTFJS)
        val b = (inputs[1]!!.data as NumberNDArrayTFJS)
        val c = inputs.getOrNull(2)?.data as? NumberNDArrayTFJS

        val result = tidyNDArray {
            val alphaScalar = NumberNDArrayTFJS(tensor(arrayOf(alpha), emptyArray(), a.dtype))
            val matMulResult = alphaScalar * a.matmul(b, transposeLeft = transA, transposeRight = transB)

            if (c == null) {
                matMulResult
            } else {
                val betaScalar = NumberNDArrayTFJS(tensor(arrayOf(beta), emptyArray(), c.dtype))
                matMulResult + betaScalar * c
            }
        }

        return listOf(result.asTensor())
    }
}
