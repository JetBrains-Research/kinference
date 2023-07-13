package io.kinference.ndarray.arrays

import io.kinference.ndarray.*
import io.kinference.ndarray.activateDefaultBackend
import io.kinference.ndarray.extensions.*
import io.kinference.primitives.types.DataType
import io.kinference.utils.toFloatArray
import io.kinference.utils.toIntArray
import kotlinx.coroutines.await

abstract class NDArrayTFJS internal constructor(internal var tfjsArray: ArrayTFJS) : NDArray {
    init {
        if (!isActivated) {
            activateDefaultBackend()
            isActivated = true
        }
    }

    override val strides
        get() = Strides(tfjsArray.shape.toIntArray())

    override val type: DataType = tfjsArray.dtype.resolveTFJSDataType()

    override fun close() {
        tfjsArray.dispose()
    }

    override suspend fun gather(indices: NDArray, axis: Int, batchDims: Int): NDArrayTFJS {
        indices as NDArrayTFJS
        return tfjsArray.gather(indices.tfjsArray, axis, batchDims).toNDArray()
    }

    override suspend fun squeeze(vararg axes: Int): NDArrayTFJS {
        return tfjsArray.squeeze(axes.toTypedArray()).toNDArray()
    }

    override suspend fun unsqueeze(vararg axes: Int): NDArrayTFJS {
        fun indexAxisForUnsqueeze(axis: Int, shapeSize: Int): Int = if (axis < 0) shapeSize + axis else axis

        val actualAxes = axes.map { indexAxisForUnsqueeze(it, rank + axes.size) }.sorted()
        val newShape = shape.toMutableList()
        for (axis in actualAxes) {
            newShape.add(axis, 1)
        }

        return this.reshape(newShape.toIntArray())
    }

    override suspend fun reshape(strides: Strides): NDArrayTFJS {
        return reshape(strides.shape)
    }

    override suspend fun reshape(shape: IntArray): NDArrayTFJS {
        return tfjsArray.reshape(shape.toTypedArray()).toNDArray()
    }

    override suspend fun nonZero(): NumberNDArrayTFJS {
        return tidyNDArray {
            val zero = zero(tfjsArray.dtype)
            val zeroFlags = this.notEqual(zero)
            zeroFlags.where().await().transpose() as NumberNDArrayTFJS
        }
    }

    @Suppress("UNCHECKED_CAST")
    override suspend fun stack(others: List<NDArray>, axis: Int): NDArrayTFJS {
        others as List<NDArrayTFJS>
        return tfjsArray.stack(*others.getArrays(), axis = axis).toNDArray()
    }

    @Suppress("UNCHECKED_CAST")
    override suspend fun concat(others: List<NDArray>, axis: Int): NDArrayTFJS {
        others as List<NDArrayTFJS>
        return tfjsArray.concat(*others.getArrays(), axis = axis).toNDArray()
    }

    override suspend fun tile(repeats: IntArray): NDArrayTFJS {
        return tfjsArray.tile(repeats.toTypedArray()).toNDArray()
    }

    override suspend fun transpose(permutations: IntArray): NDArrayTFJS {
        return tfjsArray.transpose(permutations.toTypedArray()).toNDArray()
    }

    override suspend fun transpose2D(): NDArrayTFJS = transpose(intArrayOf(1, 0))

    override suspend fun slice(starts: IntArray, ends: IntArray, steps: IntArray): NDArrayTFJS {
        val result = tfjsArray.slice(starts.toTypedArray(), ends.toTypedArray(), steps.toTypedArray())
        return result.toNDArray()
    }

    override suspend fun split(parts: Int, axis: Int): List<NDArrayTFJS> {
        return tfjsArray.split(parts, axis).map { it.toNDArray() }
    }

    override suspend fun split(split: IntArray, axis: Int): List<NDArrayTFJS> {
        return tfjsArray.split(split.toTypedArray(), axis).map { it.toNDArray() }
    }

    private fun edgePad(pads: Array<Array<Int>>): NDArrayTFJS {
        fun symmetricPadSingle(i: Int, input: ArrayTFJS): ArrayTFJS {
            val padsAtI = Array(pads.size) { j -> Array(2) { if (i < pads[j][it]) 1 else 0 } }
            return input.symmetricPad(padsAtI)
        }

        val maxPadNum = pads.flatten().max()

        var padded: ArrayTFJS = tfjsArray
        for (i in 0 until maxPadNum) {
            padded = symmetricPadSingle(i, padded)
        }

        return padded.toNDArray()
    }

    suspend fun pad(pads: Array<Array<Int>>, mode: PadMode, constantValue: NDArray?): NDArrayTFJS {
        require(constantValue == null || constantValue is NDArrayTFJS)

        return tidyNDArray {
            when (mode) {
                PadMode.CONSTANT -> {
                    val value = constantValue as? NDArrayTFJS ?: zero(dtype)
                    tfjsArray.pad(pads, value.singleValue()).toNDArray()
                }
                PadMode.REFLECT -> {
                    tfjsArray.reflectPad(pads).toNDArray()
                }
                PadMode.EDGE -> {
                    edgePad(pads)
                }
            }
        }
    }

    override suspend fun pad(pads: Array<Pair<Int, Int>>, mode: PadMode, constantValue: NDArray?): NDArrayTFJS {
        val padsArray = Array(pads.size) {
            val currentPad = pads[it]
            arrayOf(currentPad.first, currentPad.second)
        }
        return pad(padsArray, mode, constantValue)
    }

    override fun clone() = tfjsArray.clone().toNDArray()

    companion object {
        private var isActivated = false

        private fun Array<Int>.times() = this.fold(1, Int::times)

        internal fun zero(dtype: String): NDArrayTFJS {
            val zero = when (dtype) {
                "int32" -> intScalar(0)
                "float32" -> floatScalar(0f)
                "bool" -> booleanScalar(false)
                else -> error("Unsupported data type: $dtype")
            }
            return zero
        }



        fun float(values: FloatArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values, shape, "float32"))
        fun float(values: Array<Float>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values, shape, "float32"))
        fun float(values: DoubleArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toTypedArray(), shape, "float32"))
        fun float(values: Array<Double>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values, shape, "float32"))
        fun float(values: LongArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toFloatArray(), shape, "float32"))
        fun float(values: Array<Long>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toFloatArray(), shape, "float32"))
        fun float(values: IntArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values, shape, "float32"))
        fun float(values: Array<Int>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values, shape, "float32"))
        fun float(values: ShortArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toTypedArray(), shape, "float32"))
        fun float(values: Array<Short>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values, shape, "float32"))
        fun float(values: ByteArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toTypedArray(), shape, "float32"))
        fun float(values: Array<Byte>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values, shape, "float32"))
        fun float(values: UByteArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toFloatArray(), shape, "float32"))
        fun float(values: Array<UByte>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toFloatArray(), shape, "float32"))
        fun float(values: UShortArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toFloatArray(), shape, "float32"))
        fun float(values: Array<UShort>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toFloatArray(), shape, "float32"))
        fun float(values: UIntArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toFloatArray(), shape, "float32"))
        fun float(values: Array<UInt>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toFloatArray(), shape, "float32"))
        fun float(values: ULongArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toFloatArray(), shape, "float32"))
        fun float(values: Array<ULong>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toFloatArray(), shape, "float32"))

        fun int(values: IntArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values, shape, "int32"))
        fun int(values: Array<Int>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values, shape, "int32"))
        fun int(values: ShortArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toTypedArray(), shape, "int32"))
        fun int(values: Array<Short>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values, shape, "int32"))
        fun int(values: ByteArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toTypedArray(), shape, "int32"))
        fun int(values: Array<Byte>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values, shape, "int32"))
        fun int(values: LongArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toIntArray(), shape, "int32"))
        fun int(values: Array<Long>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toIntArray(), shape, "int32"))
        fun int(values: UByteArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toIntArray(), shape, "int32"))
        fun int(values: Array<UByte>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toIntArray(), shape, "int32"))
        fun int(values: UShortArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toIntArray(), shape, "int32"))
        fun int(values: Array<UShort>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toIntArray(), shape, "int32"))
        fun int(values: UIntArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toIntArray(), shape, "int32"))
        fun int(values: Array<UInt>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toIntArray(), shape, "int32"))
        fun int(values: ULongArray, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toIntArray(), shape, "int32"))
        fun int(values: Array<ULong>, shape: Array<Int>) = NumberNDArrayTFJS(tensor(values.toIntArray(), shape, "int32"))

        fun boolean(values: Array<Boolean>, shape: Array<Int>) = BooleanNDArrayTFJS(tensor(values, shape))
        fun boolean(values: BooleanArray, shape: Array<Int>) = BooleanNDArrayTFJS(tensor(values.toTypedArray(), shape))
        fun string(values: Array<String>, shape: Array<Int>) = StringNDArrayTFJS(tensor(values, shape))

        fun float(shape: Array<Int>, init: (Int) -> Float) = NumberNDArrayTFJS(tensor(FloatArray(shape.times(), init), shape, "float32"))
        fun int(shape: Array<Int>, init: (Int) -> Int) = NumberNDArrayTFJS(tensor(IntArray(shape.times(), init), shape, "int32"))
        fun boolean(shape: Array<Int>, init: (Int) -> Boolean) = BooleanNDArrayTFJS(tensor(Array(shape.times()) { init(it) }, shape))
        fun string(shape: Array<Int>, init: (Int) -> String) = StringNDArrayTFJS(tensor(Array(shape.times()) { init(it) }, shape))

        fun float(shape: Array<Int>, init: (IntArray) -> Float): NumberNDArrayTFJS {
            val ndIterator = NDIndexer(shape.toIntArray())
            val array = FloatArray(shape.times()) { init(ndIterator.next()) }
            return NumberNDArrayTFJS(tensor(array, shape, "float32"))
        }

        fun int(shape: Array<Int>, init: (IntArray) -> Int): NumberNDArrayTFJS {
            val ndIterator = NDIndexer(shape.toIntArray())
            val array = IntArray(shape.times()) { init(ndIterator.next()) }
            return NumberNDArrayTFJS(tensor(array, shape, "int32"))
        }

        fun boolean(shape: Array<Int>, init: (IntArray) -> Boolean): BooleanNDArrayTFJS {
            val ndIterator = NDIndexer(shape.toIntArray())
            val array = Array(shape.times()) { init(ndIterator.next()) }
            return BooleanNDArrayTFJS(tensor(array, shape))
        }

        fun string(shape: Array<Int>, init: (IntArray) -> String): StringNDArrayTFJS {
            val ndIterator = NDIndexer(shape.toIntArray())
            val array = Array(shape.times()) { init(ndIterator.next()) }
            return StringNDArrayTFJS(tensor(array, shape))
        }

        fun floatScalar(value: Number) = NumberNDArrayTFJS(scalar(value.toFloat(), "float32"))
        fun intScalar(value: Number) = NumberNDArrayTFJS(scalar(value.toInt(), "int32"))
        fun booleanScalar(value: Boolean) = BooleanNDArrayTFJS(scalar(value))
        fun stringScalar(value: String) = StringNDArrayTFJS(scalar(value))

        fun floatZeros(shape: Array<Int>) = NumberNDArrayTFJS(zeros(shape, "float32"))
        fun intZeros(shape: Array<Int>) = NumberNDArrayTFJS(zeros(shape, "int32"))
        fun booleanZeros(shape: Array<Int>) = BooleanNDArrayTFJS(zeros(shape, "bool"))

        fun floatOnes(shape: Array<Int>) = NumberNDArrayTFJS(ones(shape, "float32"))
        fun intOnes(shape: Array<Int>) = NumberNDArrayTFJS(ones(shape, "int32"))
        fun booleanOnes(shape: Array<Int>) = BooleanNDArrayTFJS(ones(shape, "bool"))

        fun floatRange(start: Number, stop: Number, step: Number) = NumberNDArrayTFJS(range(start.toFloat(), stop.toFloat(), step.toFloat(), "float32"))
        fun intRange(start: Number, stop: Number, step: Number) = NumberNDArrayTFJS(range(start.toInt(), stop.toInt(), step.toInt(), "int32"))

        fun floatFill(shape: Array<Int>, value: Number) = NumberNDArrayTFJS(fill(shape, value.toFloat(), "float32"))
        fun intFill(shape: Array<Int>, value: Number) = NumberNDArrayTFJS(fill(shape, value.toFloat(), "int32"))
        fun stringFill(shape: Array<Int>, value: String) = StringNDArrayTFJS(fill(shape, value, "string"))

        fun onesLike(tensor: NumberNDArrayTFJS) = NumberNDArrayTFJS(onesLike(tensor.tfjsArray))
        fun zerosLike(tensor: NumberNDArrayTFJS) = NumberNDArrayTFJS(zerosLike(tensor.tfjsArray))

        fun oneHotFloat(indices: NumberNDArrayTFJS, depth: Int, onValue: Number = 1f, offValue: Number = 0f) =
            NumberNDArrayTFJS(oneHot(indices.tfjsArray, depth, onValue.toFloat(), offValue.toFloat(), "float32"))

        fun oneHotInt(indices: NumberNDArrayTFJS, depth: Int, onValue: Number = 1, offValue: Number = 0) =
            NumberNDArrayTFJS(oneHot(indices.tfjsArray, depth, onValue.toInt(), offValue.toInt(), "int32"))

        fun oneHotBool(indices: NumberNDArrayTFJS, depth: Int, onValue: Boolean = true, offValue: Boolean = false) =
            NumberNDArrayTFJS(oneHot(indices.tfjsArray, depth, onValue.toInt(), offValue.toInt(), "bool"))
    }
}
