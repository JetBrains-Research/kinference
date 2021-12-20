package io.kinference.utils.webgpu

import io.kinference.TestLoggerFactory
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.test.assertContentEquals

class ApiTest {
    @Test
    fun testAdapterSupportedLimits() = TestRunner.runTest {
        val adapter = WebGPUInstance.requestAdapter()
        logger.info { "Adapter limits: ${adapter.limits.asString()}" }
    }

    @Test
    fun testDeviceSupportedLimits() = TestRunner.runTest {
        val device = WebGPUInstance.requestAdapter().requestDevice()
        logger.info { "Device limits: ${device.limits.asString()}" }
    }

    @Test
    fun testBasic() = TestRunner.runTest {
        val numbers = intArrayOf(1, 2, 3, 4)

        val device = WebGPUInstance.requestAdapter().requestDevice()
        val shader = device.createShaderModule(ShaderModuleDescriptor(testShader))

        val stagingBuffer = device.createBuffer(
            BufferDescriptor(
                size = numbers.sizeBytes,
                usage = BufferUsageFlags(BufferUsage.MapRead, BufferUsage.CopyDst)
            )
        )
        val storageBuffer = device.createBuffer(
            BufferDescriptor(
                size = numbers.sizeBytes,
                usage = BufferUsageFlags(BufferUsage.Storage, BufferUsage.CopyDst, BufferUsage.CopySrc)
            )
        )

        val bindGroupLayout = device.createBindGroupLayout(
            BindGroupLayoutDescriptor(
                listOf(BindGroupLayoutEntry(0, BufferBindingLayout(BufferBindingType.Storage)))
            )
        )
        val bindGroup = device.createBindGroup(
            BindGroupDescriptor(
                layout = bindGroupLayout,
                entries = listOf(BindGroupEntry(0, BufferBinding(storageBuffer)))
            )
        )
        val pipelineLayout = device.createPipelineLayout(
            PipelineLayoutDescriptor(
                bindGroupLayouts = listOf(bindGroupLayout)
            )
        )
        val computePipeline = device.createComputePipeline(
            ComputePipelineDescriptor(
                layout = pipelineLayout,
                compute = ProgrammableStage(
                    module = shader,
                    entryPoint = "main"
                )
            )
        )

        val encoder = device.createCommandEncoder()
        val computePass = encoder.beginComputePass()

        computePass.setPipeline(computePipeline)
        computePass.setBindGroup(0, bindGroup, listOf())
        computePass.dispatch(numbers.size)
        computePass.endPass()
        encoder.copyBufferToBuffer(storageBuffer, 0, stagingBuffer, 0, storageBuffer.size)

        val queue = device.queue
        val cmdBuffer = encoder.finish()
        queue.writeBuffer(
            storageBuffer,
            0,
            BufferData(numbers)
        )
        queue.submit(listOf(cmdBuffer))

        stagingBuffer.mapAsync(MapModeFlags(MapMode.Read))
        val times = stagingBuffer.getMappedRange().toIntArray()
        stagingBuffer.unmap()

        assertContentEquals(intArrayOf(0, 1, 7, 2), times)
    }

    companion object {
        private val logger = TestLoggerFactory.create("ApiTest")

        private const val testShader = """
[[block]]
struct PrimeIndices {
    data: [[stride(4)]] array<u32>;
}; // this is used as both input and output for convenience

[[group(0), binding(0)]]
var<storage, read_write> v_indices: PrimeIndices;

// The Collatz Conjecture states that for any integer n:
// If n is even, n = n/2
// If n is odd, n = 3n+1
// And repeat this process for each new n, you will always eventually reach 1.
// Though the conjecture has not been proven, no counterexample has ever been found.
// This function returns how many times this recurrence needs to be applied to reach 1.
fn collatz_iterations(n_base: u32) -> u32{
    var n: u32 = n_base;
    var i: u32 = 0u;
    loop {
        if (n <= 1u) {
            break;
        }
        if (n % 2u == 0u) {
            n = n / 2u;
        }
        else {
            // Overflow? (i.e. 3*n + 1 > 0xffffffffu?)
            if (n >= 1431655765u) {   // 0x55555555u
                return 4294967295u;   // 0xffffffffu
            }

            n = 3u * n + 1u;
        }
        i = i + 1u;
    }
    return i;
}

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    v_indices.data[global_id.x] = collatz_iterations(v_indices.data[global_id.x]);
}"""
    }
}
