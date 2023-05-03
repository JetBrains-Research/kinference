package io.kinference.tfjs.operators.activations

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.tidyNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.tfjs.data.tensors.*

sealed class Softmax(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): Softmax {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in SoftmaxVer1.VERSION.asRange() -> SoftmaxVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of Softmax operator: $version")
            }
        }
    }
}

//only for float and double types

class SoftmaxVer1(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Softmax(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), false, default = 1)
        )

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)
        private val INFO = OperatorInfo("Softmax", ATTRIBUTES_INFO, INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        private suspend fun NumberNDArrayTFJS.softmaxNonLastAxis(axis: Int): NumberNDArrayTFJS {
            return tidyNDArray {
                val rows = this.computeBlockSize(toDim = axis)
                val columns = this.computeBlockSize(fromDim = axis)
                val matrixShape = intArrayOf(rows,columns)

                val matrixLike = this.reshape(matrixShape).softmax(axis = -1)
                matrixLike.reshape(this.shape)
            }
        }
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data as NumberNDArrayTFJS
        val actualAxis = input.indexAxis(axis)

        val result = if (actualAxis == input.shape.lastIndex) {
            input.softmax(actualAxis)
        } else {
            input.softmaxNonLastAxis(actualAxis)
        }

        return listOf(result.asTensor("output"))
    }
}
