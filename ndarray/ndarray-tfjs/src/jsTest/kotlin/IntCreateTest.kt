import io.kinference.ndarray.arrays.NDArrayTFJS
import io.kinference.ndarray.extensions.dataInt
import kotlin.test.Test
import kotlin.test.assertContentEquals

class IntCreateTest {
    @Test
    fun createFromIntArray() {
        val tensor = NDArrayTFJS.int(intArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from int array")

        tensor.close()
    }

    @Test
    fun createFromIntArrayTyped() {
        val tensor = NDArrayTFJS.int(arrayOf<Int>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from int array")

        tensor.close()
    }

    @Test
    fun createFromByteArray() {
        val tensor = NDArrayTFJS.int(byteArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from byte array")

        tensor.close()
    }

    @Test
    fun createFromByteArrayTyped() {
        val tensor = NDArrayTFJS.int(arrayOf<Byte>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from byte array")

        tensor.close()
    }

    @Test
    fun createFromShortArray() {
        val tensor = NDArrayTFJS.int(shortArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from short array")

        tensor.close()
    }

    @Test
    fun createFromShortArrayTyped() {
        val tensor = NDArrayTFJS.int(arrayOf<Short>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from short array")

        tensor.close()
    }

    @Test
    fun createFromLongArray() {
        val tensor = NDArrayTFJS.int(longArrayOf(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from long array")

        tensor.close()
    }

    @Test
    fun createFromLongArrayTyped() {
        val tensor = NDArrayTFJS.int(arrayOf<Long>(1, 2, 3, 4), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from long array")

        tensor.close()
    }

    @Test
    fun createFromUIntArray() {
        val tensor = NDArrayTFJS.int(uintArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from uint array")

        tensor.close()
    }

    @Test
    fun createFromUIntArrayTyped() {
        val tensor = NDArrayTFJS.int(arrayOf<UInt>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from uint array")

        tensor.close()
    }

    @Test
    fun createFromUByteArray() {
        val tensor = NDArrayTFJS.int(ubyteArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from ubyte array")

        tensor.close()
    }

    @Test
    fun createFromUByteArrayTyped() {
        val tensor = NDArrayTFJS.int(arrayOf<UByte>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from ubyte array")

        tensor.close()
    }

    @Test
    fun createFromUShortArray() {
        val tensor = NDArrayTFJS.int(ushortArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from ushort array")

        tensor.close()
    }

    @Test
    fun createFromUShortArrayTyped() {
        val tensor = NDArrayTFJS.int(arrayOf<UShort>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from ushort array")

        tensor.close()
    }

    @Test
    fun createFromULongArray() {
        val tensor = NDArrayTFJS.int(ulongArrayOf(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from ulong array")

        tensor.close()
    }

    @Test
    fun createFromULongArrayTyped() {
        val tensor = NDArrayTFJS.int(arrayOf<ULong>(1u, 2u, 3u, 4u), arrayOf(2, 2))

        val data = tensor.dataInt()
        assertContentEquals(intArrayOf(1, 2, 3, 4), data, "Problem with creating int tensor from ulong array")

        tensor.close()
    }
}
