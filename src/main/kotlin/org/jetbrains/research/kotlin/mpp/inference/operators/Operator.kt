package org.jetbrains.research.kotlin.mpp.inference.operators

import AttributeProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.mpp.inference.attributes.Attribute
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor


class AttributeInfo(val name: String, val types: Set<AttributeProto.AttributeType>, val required: Boolean = false, val default: Any? = null) {
    init {
        require(types.isNotEmpty()) { "Attribute info must have at least one type constraint!" }
    }
}

class InputInfo(val index: Int, val types: Set<DataType>, val name: String? = null, val required: Boolean = false, val default: Any? = null) {
    init {
        require(types.isNotEmpty()) { "Input info must have at least one type constraint!" }
    }
}

class OutputInfo(val index: Int, val types: Set<DataType>, val name: String? = null) {
    init {
        require(types.isNotEmpty()) { "Output info must have at least one type constraint!" }
    }
}


@Suppress("UNCHECKED_CAST")
abstract class Operator(val name: String,
                        val attributes: Map<String, Attribute<Any>>,
                        val attributesInfo: Collection<AttributeInfo>,
                        val inputsInfo: Collection<InputInfo>,
                        val outputsInfo: Collection<OutputInfo>) {

    init {
        // TODO check attributes
    }

    fun applyWithCheck(inputs: Collection<Tensor>): Collection<Tensor> {
        // TODO check inputs
        val outputs = apply(inputs)
        // TODO check outputs
        return outputs
    }

    fun getAttributeValue(key: String): Any {
        val info = attributesInfo.find { it.name == key }
        require(info != null) { "Attribute '$key' not specified in the '$name' operator" }

        val value = attributes[key]?.value ?: if (!info.required) info.default else null
        require(value != null) { "Attribute '$key' not found or don't have a default value" }
        return value
    }

    abstract fun apply(inputs: Collection<Tensor>): Collection<Tensor>
    open fun apply(vararg inputs: Tensor): Collection<Tensor> = apply(inputs.toList())
}
