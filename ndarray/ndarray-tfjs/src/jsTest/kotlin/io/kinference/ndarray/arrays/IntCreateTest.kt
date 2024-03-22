package io.kinference.ndarray.arrays

import io.kinference.ndarray.extensions.dataInt
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.test.assertContentEquals

class IntCreateTest {
    @Test
    fun createFromIntArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(intArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from int array")

        tensor.close()
    }

    @Test
    fun createFromIntArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(arrayOf<Int>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from int array")

        tensor.close()
    }

    @Test
    fun createFromByteArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(byteArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from byte array")

        tensor.close()
    }

    @Test
    fun createFromByteArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(arrayOf<Byte>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from byte array")

        tensor.close()
    }

    @Test
    fun createFromShortArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(shortArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from short array")

        tensor.close()
    }

    @Test
    fun createFromShortArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(arrayOf<Short>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from short array")

        tensor.close()
    }

    @Test
    fun createFromLongArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(longArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from long array")

        tensor.close()
    }

    @Test
    fun createFromLongArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(arrayOf<Long>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from long array")

        tensor.close()
    }

    @Test
    fun createFromUIntArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(uintArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from uint array")

        tensor.close()
    }

    @Test
    fun createFromUIntArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(arrayOf<UInt>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from uint array")

        tensor.close()
    }

    @Test
    fun createFromUByteArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(ubyteArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from ubyte array")

        tensor.close()
    }

    @Test
    fun createFromUByteArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(arrayOf<UByte>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from ubyte array")

        tensor.close()
    }

    @Test
    fun createFromUShortArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(ushortArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from ushort array")

        tensor.close()
    }

    @Test
    fun createFromUShortArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(arrayOf<UShort>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from ushort array")

        tensor.close()
    }

    @Test
    fun createFromULongArray() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(ulongArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from ulong array")

        tensor.close()
    }

    @Test
    fun createFromULongArrayTyped() = TestRunner.runTest {
        val tensor = NDArrayTFJS.int(arrayOf<ULong>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Failed to create int tensor from ulong array")

        tensor.close()
    }
}
