package io.kinference.ndarray.arrays

import io.kinference.ndarray.extensions.dataFloat
import kotlin.test.Test
import kotlinx.coroutines.test.runTest
import kotlin.test.assertContentEquals

class FloatCreateTest {
    @Test
    fun createFromFloatArray() = runTest {
        val tensor = NDArrayTFJS.float(floatArrayOf(1f, 2f, 3f, 4f), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from float array")

        tensor.close()
    }

    @Test
    fun createFromFloatArrayTyped() = runTest {
        val tensor = NDArrayTFJS.float(arrayOf(1f, 2f, 3f, 4f), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from float array")

        tensor.close()
    }

    @Test
    fun createFromDoubleArray() = runTest {
        val tensor = NDArrayTFJS.float(doubleArrayOf(1.0, 2.0, 3.0, 4.0), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from double array")

        tensor.close()
    }

    @Test
    fun createFromDoubleArrayTyped() = runTest {
        val tensor = NDArrayTFJS.float(arrayOf(1.0, 2.0, 3.0, 4.0), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from double array")

        tensor.close()
    }

    @Test
    fun createFromIntArray() = runTest {
        val tensor = NDArrayTFJS.float(intArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from int array")

        tensor.close()
    }

    @Test
    fun createFromIntArrayTyped() = runTest {
        val tensor = NDArrayTFJS.float(arrayOf<Int>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from int array")

        tensor.close()
    }

    @Test
    fun createFromByteArray() = runTest {
        val tensor = NDArrayTFJS.float(byteArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from byte array")

        tensor.close()
    }

    @Test
    fun createFromByteArrayTyped() = runTest {
        val tensor = NDArrayTFJS.float(arrayOf<Byte>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from byte array")

        tensor.close()
    }

    @Test
    fun createFromShortArray() = runTest {
        val tensor = NDArrayTFJS.float(shortArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from short array")

        tensor.close()
    }

    @Test
    fun createFromShortArrayTyped() = runTest {
        val tensor = NDArrayTFJS.float(arrayOf<Short>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from short array")

        tensor.close()
    }

    @Test
    fun createFromLongArray() = runTest {
        val tensor = NDArrayTFJS.float(longArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from long array")

        tensor.close()
    }

    @Test
    fun createFromLongArrayTyped() = runTest {
        val tensor = NDArrayTFJS.float(arrayOf<Long>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from long array")

        tensor.close()
    }

    @Test
    fun createFromUByteArray() = runTest {
        val tensor = NDArrayTFJS.float(ubyteArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from ubyte array")

        tensor.close()
    }

    @Test
    fun createFromUByteArrayTyped() = runTest {
        val tensor = NDArrayTFJS.float(arrayOf<UByte>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from ubyte array")

        tensor.close()
    }

    @Test
    fun createFromUShortArray() = runTest {
        val tensor = NDArrayTFJS.float(ushortArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from ushort array")

        tensor.close()
    }

    @Test
    fun createFromUShortArrayTyped() = runTest {
        val tensor = NDArrayTFJS.float(arrayOf<UShort>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from ushort array")

        tensor.close()
    }

    @Test
    fun createFromUIntArray() = runTest {
        val tensor = NDArrayTFJS.float(uintArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from uint array")

        tensor.close()
    }

    @Test
    fun createFromUIntArrayTyped() = runTest {
        val tensor = NDArrayTFJS.float(arrayOf<UInt>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from uint array")

        tensor.close()
    }

    @Test
    fun createFromULongArray() = runTest {
        val tensor = NDArrayTFJS.float(ulongArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from ulong array")

        tensor.close()
    }

    @Test
    fun createFromULongArrayTyped() = runTest {
        val tensor = NDArrayTFJS.float(arrayOf<ULong>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from ulong array")

        tensor.close()
    }
}
