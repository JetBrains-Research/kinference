package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class ConstantOfShape(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 9)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in ConstantOfShapeVer9.VERSION.asRange() -> ConstantOfShapeVer9(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}


class ConstantOfShapeVer9(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : ConstantOfShape(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = PRIMITIVE_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo(
                name = "value",
                types = setOf(AttributeProto.AttributeType.TENSOR),
                default = NDArrayTFJS.float(floatArrayOf(0f), arrayOf(1)).asTensor("value"),
                required = false
            )
        )

        private val INPUTS_INFO = listOf(IOInfo(0, setOf(TensorProto.DataType.INT64), "input", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 9)
        private val INFO = OperatorInfo("ConstantOfShape", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        private fun empty(type: DataType, shape: Array<Int>): NDArrayTFJS {
            return when (type) {
                DataType.FLOAT -> NDArrayTFJS.float(FloatArray(0), shape)
                DataType.INT -> NDArrayTFJS.int(IntArray(0), shape)
                DataType.BOOLEAN -> NDArrayTFJS.boolean(BooleanArray(0), shape)
                DataType.ALL -> NDArrayTFJS.string(emptyArray(), shape)
                else -> error("")
            }
        }
    }

    private val value: TFJSTensor by attribute()

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val output = tidyNDArray {
            val shape = inputs[0]!!.data.dataInt().toTypedArray()
            if (shape.contains(0)) {
                return@tidyNDArray empty(value.data.type, shape)
            }

            return@tidyNDArray value.data.broadcastTo(shape)
        }

        return listOf(output.asTensor("output"))
    }
}

