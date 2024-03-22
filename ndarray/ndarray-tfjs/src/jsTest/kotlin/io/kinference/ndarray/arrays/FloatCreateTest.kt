package io.kinference.ndarray.arrays

import io.kinference.ndarray.extensions.dataFloat
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.test.assertContentEquals

class FloatCreateTest {
    @Test
    fun createFromFloatArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(floatArrayOf(1f, 2f, 3f, 4f), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from float array")

        tensor.close()
    }

    @Test
    fun createFromFloatArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(arrayOf(1f, 2f, 3f, 4f), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from float array")

        tensor.close()
    }

    @Test
    fun createFromDoubleArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(doubleArrayOf(1.0, 2.0, 3.0, 4.0), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from double array")

        tensor.close()
    }

    @Test
    fun createFromDoubleArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(arrayOf(1.0, 2.0, 3.0, 4.0), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from double array")

        tensor.close()
    }

    @Test
    fun createFromIntArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(intArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from int array")

        tensor.close()
    }

    @Test
    fun createFromIntArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(arrayOf<Int>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from int array")

        tensor.close()
    }

    @Test
    fun createFromByteArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(byteArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from byte array")

        tensor.close()
    }

    @Test
    fun createFromByteArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(arrayOf<Byte>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from byte array")

        tensor.close()
    }

    @Test
    fun createFromShortArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(shortArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from short array")

        tensor.close()
    }

    @Test
    fun createFromShortArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(arrayOf<Short>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from short array")

        tensor.close()
    }

    @Test
    fun createFromLongArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(longArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from long array")

        tensor.close()
    }

    @Test
    fun createFromLongArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(arrayOf<Long>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from long array")

        tensor.close()
    }

    @Test
    fun createFromUByteArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(ubyteArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from ubyte array")

        tensor.close()
    }

    @Test
    fun createFromUByteArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(arrayOf<UByte>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from ubyte array")

        tensor.close()
    }

    @Test
    fun createFromUShortArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(ushortArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from ushort array")

        tensor.close()
    }

    @Test
    fun createFromUShortArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(arrayOf<UShort>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from ushort array")

        tensor.close()
    }

    @Test
    fun createFromUIntArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(uintArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from uint array")

        tensor.close()
    }

    @Test
    fun createFromUIntArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(arrayOf<UInt>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from uint array")

        tensor.close()
    }

    @Test
    fun createFromULongArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(ulongArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from ulong array")

        tensor.close()
    }

    @Test
    fun createFromULongArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.float(arrayOf<ULong>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataFloat()
        assertContentEquals(floatArrayOf(1f, 2f, 3f, 4f), data, "Failed to create float tensor from ulong array")

        tensor.close()
    }
}
