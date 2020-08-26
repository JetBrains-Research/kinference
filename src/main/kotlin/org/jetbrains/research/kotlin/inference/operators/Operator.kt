package org.jetbrains.research.kotlin.inference.operators

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ONNXData
import org.jetbrains.research.kotlin.inference.data.ONNXDataType
import org.jetbrains.research.kotlin.inference.data.seq.TensorSeq
import org.jetbrains.research.kotlin.inference.data.tensors.Tensor
import org.jetbrains.research.kotlin.inference.graph.Context
import org.jetbrains.research.kotlin.inference.ndarray.extensions.isScalar
import org.jetbrains.research.kotlin.inference.onnx.AttributeProto
import org.jetbrains.research.kotlin.inference.onnx.TensorProto.DataType
import kotlin.properties.ReadOnlyProperty
import kotlin.reflect.KProperty

class AttributeInfo(val name: String, val types: Set<AttributeProto.AttributeType>, val required: Boolean = false, val default: Any? = null) {
    init {
        require(types.isNotEmpty()) { "Attribute info must have at least one type constraint!" }
    }
}

open class IOInfo(val index: Int, val types: Set<DataType>, val name: String,
                  val optional: Boolean = false, val onnxDataType: ONNXDataType = ONNXDataType.ONNX_TENSOR,
                  val scalar: Boolean = false, val differentiable: Boolean? = null /* null == undefined, TODO */) {
    init {
        require(types.isNotEmpty()) { "Input info must have at least one type constraint!" }
    }
}

class VariadicIOInfo(startIndex: Int, types: Set<DataType>, name: String,
                     val minimumArity: Int = 0, onnxDataType: ONNXDataType = ONNXDataType.ONNX_TENSOR,
                     scalar: Boolean = false, differentiable: Boolean? = null,
                     val heterogeneous: Boolean = true) : IOInfo(startIndex, types, name, minimumArity == 0, onnxDataType, scalar, differentiable)

data class OperatorInfo(val name: String, val attributes: Map<String, AttributeInfo>, val inputs: List<IOInfo>, val outputs: List<IOInfo>) {
    constructor(name: String, attributes: Collection<AttributeInfo>, inputs: List<IOInfo>, outputs: List<IOInfo>)
        : this(name, attributes.map { it.name to it }.toMap(), inputs, outputs)

    init {
        val variadicInputIndex = inputs.indexOfFirst { it is VariadicIOInfo }
        require(variadicInputIndex == -1 || variadicInputIndex == inputs.size - 1) { "Variadic input must be last" }

        val variadicOutputIndex = outputs.indexOfFirst { it is VariadicIOInfo }
        require(variadicOutputIndex == -1 || variadicOutputIndex == outputs.size - 1) { "Variadic output must be last" }
    }
}

@Suppress("UNCHECKED_CAST")
abstract class Operator<in T : ONNXData, out U : ONNXData>(val info: OperatorInfo, val attributes: Map<String, Attribute<Any>> = emptyMap(), val inputs: List<String>, val outputs: List<String>) {
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

    private fun check(constraints: List<IOInfo>, values: List<ONNXData?>, what: String) {
        fun infos(constraints: List<IOInfo>) = sequence<IOInfo?> {
            for (constraint in constraints) {
                while (constraint is VariadicIOInfo) yield(constraint)
                yield(constraint)
            }

            yield(null)
        }

        var variadicCounter = 0
        var variadicType: DataType? = null
        infos(constraints).zip(values.asSequence().plusElement(null)) { constraint, value ->
            // TODO check for not null variadic
            if (value == null) {
                require(constraint == null || constraint.optional || (constraint is VariadicIOInfo && variadicCounter >= constraint.minimumArity)) {
                    "Required $what '${constraint!!.name}' for '${info.name}' operator not provided"
                }
                return@zip
            }

            require(constraint != null) { "Unexpected $what '${value.info.name}' for '${info.name}' operator" }

            if (constraint is VariadicIOInfo) {
                if (variadicCounter == 0) variadicType = value.info.type
                variadicCounter++

                if (!constraint.heterogeneous) require(value.info.type == variadicType) { "All ${what}s for '${constraint.name}' must have same type\nPresent: ${value.info.type}, Expected: $variadicType" }
            }

            require(value.type == constraint.onnxDataType) { "Wrong $what ONNX data type '${value.info.name}' for '${info.name}' operator\nPresent: ${value.type}, Expected: ${constraint.onnxDataType}" }
            require(value.info.type in constraint.types) { "Wrong $what type '${value.info.name}' for '${info.name}' operator\nPresent: ${value.info.type}, Expected: ${constraint.types}" }
            if (constraint.scalar) {
                when (value.type) {
                    ONNXDataType.ONNX_TENSOR -> require((value as Tensor).data.isScalar()) { "${what.capitalize()} '${value.info.name}' must be a scalar for '${info.name}' operator" }
                    ONNXDataType.ONNX_SEQUENCE -> require((value as TensorSeq).data.all { it.data.isScalar() }) { "${what.capitalize()} '${value.info.name}' must be a list of scalars for '${info.name}' operator" }
                }
            }
        }
    }

    fun applyWithCheck(context: Context, inputs: List<T?>): List<U?> {
        check(info.inputs, inputs, "input")
        val outputs = apply(context, inputs)
        require(outputs.size >= outputs.size) { "Operator '${info.name}' doesn't provide expected output size\nPresent: ${outputs.size}, Expected: at least ${outputs.size}" }
        check(info.outputs, outputs, "output")
        return outputs
    }

    class AttributeValueDelegate<Input, Output>(val name: String?, val transform: (Input) -> Output,
                                                val getValue: Operator<*, *>.(String) -> Any?) : ReadOnlyProperty<Operator<*, *>, Output> {
        private var initialized: Boolean = false
        private var value: Output? = null

        override fun getValue(thisRef: Operator<*, *>, property: KProperty<*>): Output {
            if (!initialized) {
                value = transform(thisRef.getValue(name ?: property.name) as Input)
                initialized = true
            }
            return value as Output
        }
    }

    fun <O> attribute(name: String? = null) = AttributeValueDelegate<O, O>(name, { it }, { getAttribute(it) })
    fun <I, O> attribute(name: String? = null, transform: (I) -> O) = AttributeValueDelegate(name, transform, { getAttribute(it) })

    /**
     * Get attribute from operator info
     *
     * Consider using [attribute] that performs caching on delegate side.
     */
    fun <T> getAttribute(key: String): T {
        val value = getAttributeOrNull<T>(key)
        require(value != null) { "Attribute '$key' not found or don't have a default value" }
        return value
    }

    fun <O> attributeOrNull(name: String? = null) = AttributeValueDelegate<O, O?>(name, { it }, { getAttributeOrNull(it) })
    fun <I, O> attributeOrNull(name: String? = null, transform: (I?) -> O?) = AttributeValueDelegate(name, transform, { getAttributeOrNull(it) })

    /**
     * Get attribute from operator info or null if no default
     *
     * Consider using [attributeOrNull] that performs caching on delegate side
     */
    fun <T> getAttributeOrNull(key: String): T? {
        val info = info.attributes[key]
        require(info != null) { "Attribute '$key' not specified in the '${this.info.name}' operator" }

        return attributes[key]?.value as T? ?: if (!info.required) info.default as T? else null
    }

    abstract fun apply(context: Context, inputs: List<T?>): List<U?>
    open fun apply(context: Context, vararg inputs: T?): Collection<U?> = apply(context, inputs.toList())

    companion object {
        val ALL_DATA_TYPES = DataType.values().toHashSet() - DataType.UNDEFINED
        val FLOAT_DATA_TYPES = setOf(DataType.BFLOAT16, DataType.FLOAT16, DataType.FLOAT, DataType.DOUBLE)
    }
}
