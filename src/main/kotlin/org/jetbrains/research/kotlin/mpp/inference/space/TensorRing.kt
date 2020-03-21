package org.jetbrains.research.kotlin.mpp.inference.space

import scientifik.kmath.linear.GenericMatrixContext
import scientifik.kmath.operations.Ring
import scientifik.kmath.operations.RingElement
import scientifik.kmath.structures.*

abstract class TensorRing<T : Any>(
    override val shape: IntArray,
    override val elementContext: Ring<T>,
    override val strides: Strides = DefaultStrides(shape)
) : BufferedNDRing<T, Ring<T>> {
    abstract val bufferBuilder: (Int, (Int) -> T) -> Buffer<T>
    abstract val matrixContext : GenericMatrixContext<T, Ring<T>>

    abstract fun rebuild(newDims: IntArray): TensorRing<T>

    override fun map(arg: NDBuffer<T>, transform: Ring<T>.(T) -> T): NDBuffer<T> {
        val array = bufferBuilder(arg.strides.linearSize) { offset -> elementContext.transform(arg.buffer[offset]) }
        return BufferedNDSpaceElement(this, array)
    }

    override fun mapIndexed(arg: NDBuffer<T>, transform: Ring<T>.(index: IntArray, T) -> T): NDBuffer<T> {
        return BufferedNDSpaceElement(
            this,
            bufferBuilder(arg.strides.linearSize) { offset ->
                elementContext.transform(arg.strides.index(offset), arg.buffer[offset])
            }
        )
    }

    override fun produce(initializer: Ring<T>.(IntArray) -> T): NDBuffer<T> {
        val array = bufferBuilder(strides.linearSize) { offset -> elementContext.initializer(strides.index(offset)) }
        return BufferedNDSpaceElement(this, array)
    }

    override fun NDBuffer<T>.toElement(): RingElement<NDBuffer<T>, *, out BufferedNDRing<T, Ring<T>>> {
        return BufferedNDRingElement(this@TensorRing, buffer)
    }

    override fun combine(a: NDBuffer<T>, b: NDBuffer<T>, transform: Ring<T>.(T, T) -> T): NDBuffer<T> {
        return BufferedNDSpaceElement(
            this,
            bufferBuilder(strides.linearSize) { offset -> elementContext.transform(a.buffer[offset], b.buffer[offset]) }
        )
    }
}
