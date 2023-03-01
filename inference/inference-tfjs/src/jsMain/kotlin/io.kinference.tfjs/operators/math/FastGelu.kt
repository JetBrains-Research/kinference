package io.kinference.tfjs.operators.math

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.tanh
import io.kinference.ndarray.extensions.tidyNDArray
import io.kinference.operator.*
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.utils.closeAll

sealed class FastGelu(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in FastGeluVer1.VERSION.asRange() -> FastGeluVer1(name, attributes, inputs, outputs)
                else -> error("Unsupported version of FastGelu operator: $version")
            }
    }
}

class FastGeluVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    FastGelu(name, INFO, attributes, inputs, outputs) {
    companion object {

        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "bias", optional = true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "Y", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("FastGelu", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")
    }

    private val COEF_1 = NDArrayTFJS.floatScalar(0.5f)
    private val COEF_2 = NDArrayTFJS.floatScalar(0.035677408136300125f)
    private val COEF_3 = NDArrayTFJS.floatScalar(0.7978845608028654f)


    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val output = tidyNDArray {
            val input = inputs.first()!!.data as NumberNDArrayTFJS
            val bias = inputs.getOrNull(1)?.data as? NumberNDArrayTFJS

            val inputWithBias = if (bias != null) input + bias else input

            return@tidyNDArray inputWithBias * (COEF_1 + COEF_1 * (inputWithBias * (COEF_2 * inputWithBias * inputWithBias + COEF_3)).tanh())
        }

        return listOf(output.asTensor("Y"))
    }

    override fun close() {
        super.close()
        closeAll(COEF_1, COEF_2, COEF_3)
    }
}

