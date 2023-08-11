package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.extensions.shapeArray
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class EyeLike(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 9)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): EyeLike {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in EyeLikeVer9.VERSION.asRange() -> EyeLikeVer9(name, attributes, inputs, outputs)
                else -> error("Unsupported version of EyeLike operator: $version")
            }
        }
    }
}

class EyeLikeVer9 internal constructor(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : EyeLike(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("dtype", setOf(AttributeProto.AttributeType.INT), required = false, default = null),
            AttributeInfo("k", setOf(AttributeProto.AttributeType.INT), required = false, default = 0L)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, PRIMITIVE_DATA_TYPES, "input", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, PRIMITIVE_DATA_TYPES, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 9)
        private val INFO = OperatorInfo("EyeLike", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val dtype: TensorProto.DataType? by attributeOrNull { it: Number? -> TensorProto.DataType.fromValue(it?.toInt() ?: -1) }
    private val k: Int by attribute { it: Number -> it.toInt() }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!

        val shape = input.data.shapeArray
        val outputDtype = when {
            dtype != null -> dtype!!
            input.info.type != TensorProto.DataType.UNDEFINED -> input.info.type
            else -> TensorProto.DataType.FLOAT
        }

        val output = when (outputDtype) {
            in INT_DATA_TYPES + UINT_DATA_TYPES -> NDArrayTFJS.intEyeLike(shape, k)
            in FLOAT_DATA_TYPES -> NDArrayTFJS.floatEyeLike(shape, k)
            TensorProto.DataType.BOOL -> NDArrayTFJS.booleanEyeLike(shape, k)
            else -> error("EyeLike is only supported for numeric and boolean tensors, current type: $type")
        }

        return listOf(output.asTensor(outputDtype, "output"))
    }
}
