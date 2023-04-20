package io.kinference.tfjs.operators.math

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.arrays.indexAxis
import io.kinference.ndarray.extensions.tidyNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class CumSum(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>, outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(
            name: String,
            version: Int?,
            attributes: Map<String, Attribute<Any>>,
            inputs: List<String>, outputs: List<String>
        ): CumSum {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in CumSumVer11.VERSION.asRange() -> CumSumVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of CumSum operator: $version")
            }
        }
    }
}


class CumSumVer11(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>, outputs: List<String>
) : CumSum(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT32, TensorProto.DataType.UINT64, TensorProto.DataType.INT32, TensorProto.DataType.INT64,
            TensorProto.DataType.FLOAT, TensorProto.DataType.DOUBLE, TensorProto.DataType.BFLOAT16, TensorProto.DataType.FLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("exclusive", setOf(AttributeProto.AttributeType.INT), false, 0),
            AttributeInfo("reverse", setOf(AttributeProto.AttributeType.INT), false, 0)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "x", optional = false),
            IOInfo(1, setOf(TensorProto.DataType.INT32, TensorProto.DataType.INT64), "axis", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "y", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("CumSum", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val exclusive by attribute { ex: Number -> ex.toInt() != 0 }
    private val reverse by attribute { r: Number -> r.toInt() != 0 }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data as NumberNDArrayTFJS
        val axis = (inputs[1]!!.data.singleValue() as Number).toInt()
        val actualAxis = input.indexAxis(axis)

        val result = tidyNDArray {
            input.cumulativeSum(actualAxis, exclusive, reverse)
        }

        return listOf(result.asTensor("y"))
    }
}
