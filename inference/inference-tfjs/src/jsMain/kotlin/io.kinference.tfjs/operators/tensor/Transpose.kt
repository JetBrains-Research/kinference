package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.arrays.transpose
import io.kinference.ndarray.extensions.tidyNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import kotlin.time.ExperimentalTime

sealed class Transpose(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in TransposeVer1.VERSION.asRange() -> TransposeVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}

@ExperimentalTime
class TransposeVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Transpose(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("perm", setOf(AttributeProto.AttributeType.INTS), false)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = true))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "transposed", optional = false, differentiable = true))

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("Transpose", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val perm: Array<Int>? by attributeOrNull { array: LongArray? -> array?.let { Array(array.size) { array[it].toInt() } } }

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val output = tidyNDArray {
            val input = inputs[0]!!.data
            return@tidyNDArray input.transpose(perm?.toIntArray()) as NDArrayTFJS
        }

        return listOf(output.asTensor("transposed"))
    }
}
