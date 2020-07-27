package org.jetbrains.research.kotlin.inference.operators

import AttributeProto
import TensorProto
import TensorProto.DataType
import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ONNXData
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor

class AttributeInfo(val name: String, val types: Set<AttributeProto.AttributeType>, val required: Boolean = false, val default: Any? = null) {
    init {
        require(types.isNotEmpty()) { "Attribute info must have at least one type constraint!" }
    }
}

open class InputInfo(val index: Int, val types: Set<DataType>, val name: String? = null, val required: Boolean = false, val scalar: Boolean = false) {
    init {
        require(types.isNotEmpty()) { "Input info must have at least one type constraint!" }
    }
}

class VariadicInputInfo(startIndex: Int, types: Set<DataType>, name: String? = null, requiredOne: Boolean = false, val heterogeneous: Boolean = true)
    : InputInfo(startIndex, types, name, requiredOne)

open class OutputInfo(val index: Int, val types: Set<DataType>, val name: String? = null) {
    init {
        require(types.isNotEmpty()) { "Output info must have at least one type constraint!" }
    }
}

class VariadicOutputInfo(startIndex: Int, types: Set<DataType>, name: String? = null, val heterogeneous: Boolean = true) : OutputInfo(startIndex, types, name)


data class OperatorInfo(val name: String, val attributes: Map<String, AttributeInfo>, val inputs: List<InputInfo>, val outputs: List<OutputInfo>) {
    constructor(name: String, attributes: Collection<AttributeInfo>, inputs: List<InputInfo>, outputs: List<OutputInfo>)
        : this(name, attributes.map { it.name to it }.toMap(), inputs, outputs)

    init {
        val variadicInputIndex = inputs.indexOfFirst { it is VariadicInputInfo }
        require(variadicInputIndex == -1 || variadicInputIndex == inputs.size - 1) { "Variadic input must be last" }

        val variadicOutputIndex = outputs.indexOfFirst { it is VariadicOutputInfo }
        require(variadicOutputIndex == -1 || variadicOutputIndex == outputs.size - 1) { "Variadic output must be last" }
    }
}

@Suppress("UNCHECKED_CAST")
abstract class Operator<in T : ONNXData, out U : ONNXData>(val info: OperatorInfo, val usedOutputsNum: Int, val attributes: Map<String, Attribute<Any>> = emptyMap()) {
    init {
        for (info in info.attributes.values) {
            if (info.required) require(info.name in attributes) { "Required attribute '${info.name}' not specified in ${info.name} operator" }

            attributes[info.name]?.let { attribute ->
                require(attribute.type in info.types) { "Attribute '${attribute.name}' type doesn't match specification\nPresent: ${attribute.type}, Expected: one of ${info.types}" }
            }
        }

        for (attribute in attributes.values) {
            if (attribute.name !in info.attributes) {
                System.err.println("Unknown attribute '${attribute.name}' in ${info.name} operator")
            }
        }
    }

    fun applyWithCheck(inputs: List<T>): List<U> {
        val inputConstraints = sequence<InputInfo?> {
            for (constraint in info.inputs) {
                while (constraint is VariadicInputInfo) yield(constraint)
                yield(constraint)
            }

            yield(null)
        }

        var variadicCounter = 0
        var variadicType: TensorProto.DataType? = null
        inputConstraints.zip(inputs.asSequence().plusElement(null)) { constraint, input ->
            if (input == null) {
                require(constraint == null || !constraint.required || (constraint is VariadicInputInfo && variadicCounter > 0)) {
                    "Required input '${constraint!!.name}' for '${info.name}' operator not provided"
                }
                return@zip
            }

            require(constraint != null) { "Unexpected input '${input.info.name}' for '${info.name}' operator" }

            if (constraint is VariadicInputInfo) {
                if (variadicCounter == 0) variadicType = input.info.type
                variadicCounter++

                if (!constraint.heterogeneous) require(input.info.type == variadicType) { "All inputs for '${constraint.name}' must have same type\nPresent: ${input.info.type}, Expected: $variadicType" }
            }

            require(input.info.type in constraint.types) { "Wrong input type '${input.info.name}' for '${info.name}' operator\nPresent: ${input.info.type}, Expected: ${constraint.types}" }
            if (constraint.scalar) require((input as Tensor).data.isScalar()) { "Input '${input.info.name}' must be a scalar for '${info.name}' operator" }
        }

        val outputs = apply(inputs)

        require(outputs.size >= usedOutputsNum) { "Operator '${info.name}' doesn't provide expected output size\nPresent: ${outputs.size}, Expected: at least $usedOutputsNum" }

        val outputConstraints = sequence<OutputInfo?> {
            for (constraint in info.outputs) {
                while (constraint is VariadicOutputInfo) yield(constraint)
                yield(constraint)
            }

            yield(null)
        }

        variadicCounter = 0
        variadicType = null
        outputConstraints.zip(outputs.asSequence()) { constraint, output ->
            require(constraint != null) { "Unexpected output '${output.info.name}' for '${info.name}' operator" }

            if (constraint is VariadicOutputInfo) {
                if (variadicCounter == 0) variadicType = output.info.type
                variadicCounter++

                if (!constraint.heterogeneous) require(output.info.type == variadicType) { "All outputs for '${constraint.name}' must have same type\nPresent: ${output.info.type}, Expected: $variadicType" }
            }

            require(output.info.type in constraint.types) { "Wrong output type '${output.info.name}' for '${info.name}' operator\nPresent: ${output.info.type}, Expected: ${constraint.types}" }
        }

        return outputs
    }

    fun getAttributeValue(key: String): Any {
        val value = getAttributeValueOrNull(key)
        require(value != null) { "Attribute '$key' not found or don't have a default value" }
        return value
    }

    fun getAttributeValueOrNull(key: String): Any? {
        val info = info.attributes[key]
        require(info != null) { "Attribute '$key' not specified in the '${this.info.name}' operator" }

        return attributes[key]?.value ?: if (!info.required) info.default else null
    }

    abstract fun apply(inputs: List<T>): List<U>
    open fun apply(vararg inputs: T): Collection<U> = apply(inputs.toList())

    companion object {
        val ALL_DATA_TYPES = DataType.values().toHashSet() - DataType.UNDEFINED
        val FLOAT_DATA_TYPES = setOf(DataType.BFLOAT16, DataType.FLOAT16, DataType.FLOAT, DataType.DOUBLE)
    }
}
