package org.jetbrains.research.kotlin.mpp.inference.space

import scientifik.kmath.linear.GenericMatrixContext
import scientifik.kmath.linear.MatrixContext
import scientifik.kmath.operations.*
import scientifik.kmath.structures.*

class IntTensorRing(shape: IntArray) : TensorRing<Int>(shape, IntRing) {
    override val zero: NDBuffer<Int> = NDStructure.auto(shape) { elementContext.zero }
    override val one: NDBuffer<Int> = NDStructure.auto(shape) { elementContext.one }
    override val matrixContext: GenericMatrixContext<Int, Ring<Int>> = MatrixContext.auto(IntRing)

    override fun rebuild(newDims: IntArray): TensorRing<Int> = IntTensorRing(newDims)

    override val bufferBuilder: (Int, (Int) -> Int) -> Buffer<Int>
        get() = { size, initializer -> IntBuffer(IntArray(size) { initializer(it) }) }
}

class LongTensorRing(shape: IntArray) : TensorRing<Long>(shape, LongRing) {
    override val zero: NDBuffer<Long> = NDStructure.auto(shape) { elementContext.zero }
    override val one: NDBuffer<Long> = NDStructure.auto(shape) { elementContext.one }
    override val matrixContext: GenericMatrixContext<Long, Ring<Long>> = MatrixContext.auto(LongRing)

    override fun rebuild(newDims: IntArray): TensorRing<Long> = LongTensorRing(newDims)

    override val bufferBuilder: (Int, (Int) -> Long) -> Buffer<Long>
        get() = { size, initializer -> LongBuffer(LongArray(size) { initializer(it) }) }
}

class FloatTensorRing(shape: IntArray) : TensorRing<Float>(shape, FloatField) {
    override val zero: NDBuffer<Float> = NDStructure.auto(shape) { elementContext.zero }
    override val one: NDBuffer<Float> = NDStructure.auto(shape) { elementContext.one }
    override val matrixContext: GenericMatrixContext<Float, Ring<Float>> = MatrixContext.auto(FloatField)

    override fun rebuild(newDims: IntArray): TensorRing<Float> = FloatTensorRing(newDims)

    override val bufferBuilder: (Int, (Int) -> Float) -> Buffer<Float>
        get() = { size, initializer -> Buffer.auto(Float::class, size) { initializer(it) } }
}

class DoubleTensorRing(shape: IntArray) : TensorRing<Double>(shape, RealField) {
    override val zero: NDBuffer<Double> = NDStructure.auto(shape) { elementContext.zero }
    override val one: NDBuffer<Double> = NDStructure.auto(shape) { elementContext.one }
    override val matrixContext: GenericMatrixContext<Double, Ring<Double>> = MatrixContext.auto(RealField)

    override fun rebuild(newDims: IntArray): TensorRing<Double> = DoubleTensorRing(newDims)

    override val bufferBuilder: (Int, (Int) -> Double) -> Buffer<Double>
        get() = { size, initializer -> DoubleBuffer(DoubleArray(size) { initializer(it) }) }
}

@Suppress("UNCHECKED_CAST")
inline fun <reified T : Any> tryResolveSpace(dims: IntArray) = when (T::class) {
    Double::class -> DoubleTensorRing(dims)
    Float::class -> FloatTensorRing(dims)
    Long::class -> LongTensorRing(dims)
    Int::class -> IntTensorRing(dims)
    else -> error("Unsupported data type")
} as TensorRing<T>

inline fun <reified T : Any> resolveSpace(dims: List<Long>) = tryResolveSpace<T>(dims.toIntArray())

fun Collection<Long>.toIntArray() = this.map { it.toInt() }.toIntArray()
