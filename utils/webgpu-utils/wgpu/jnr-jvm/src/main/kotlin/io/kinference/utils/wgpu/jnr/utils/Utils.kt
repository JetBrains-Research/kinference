package io.kinference.utils.wgpu.jnr.utils

import io.kinference.utils.wgpu.internal.createPointerTo
import io.kinference.utils.wgpu.internal.getPointerTo
import io.kinference.utils.wgpu.jnr.WGPUSType
import io.kinference.utils.wgpu.jnr.WGPUShaderModuleWGSLDescriptor
import io.kinference.utils.wgpu.jnr.WGPUShaderModuleDescriptor

fun loadWgsl(shaderSource: String): WGPUShaderModuleDescriptor {
    val wgslDescriptor = WGPUShaderModuleWGSLDescriptor.allocateDirect().apply {
        chain.apply {
            sType = WGPUSType.ShaderModuleWGSLDescriptor
        }
        source = shaderSource.createPointerTo()
    }
    return WGPUShaderModuleDescriptor.allocateDirect().apply {
        nextInChain = wgslDescriptor.getPointerTo()
    }
}
