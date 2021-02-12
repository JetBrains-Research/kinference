package io.kinference
import io.kinference.externals.REGL
import io.kinference.externals.*
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.time.ExperimentalTime
import kotlin.time.measureTime

class Playground {

    private val random = Random(42)
    private val regl = REGL(object : InitializationOptions {
        init {
            extensions = arrayOf("oes_texture_float")
        }
    })

    @OptIn(ExperimentalTime::class)
//    @Test
    fun play() {
        val n = 1024
        val m = 1024
        val t = 1024

        val count = 1

        println("Start matrix multiplication test M: $m, N: $n, T: $t:")

        val left = FloatArray(n * t) { random.nextFloat() }
        val right = FloatArray(t * m) { random.nextFloat() }
        var destBaseline = FloatArray(0)

        val baselineTime = measureTime {
            repeat(count) {
                destBaseline = FloatArray(n * m)
                dotBaseline(left, right, destBaseline, m, n, t)
            }
        }

        println("Baseline: ${baselineTime / count}")

        val gpu = object : DrawConfig<MMUniforms, MMAttributes, MMProps, Any, DefaultContext> {
            init {
                vert = """
                    precision mediump float;
                    attribute vec2 position;
                    varying vec2 vUv;
    
                    void main () {
                        vUv = 0.5 * (position + 1.0);
                        gl_Position = vec4(position, 0, 1);
                    }
                """.trimIndent()

                frag = """
                    precision mediump float;
                    
                    uniform sampler2D leftMatrix;
                    uniform sampler2D rightMatrix;
                    
                    varying vec2 vUv;
                    
                    #define destDim vec2($n,$m)
                    #define leftDim vec2($n,$t)
                    #define rightDim vec2($t,$m)
                    
                    #define leftDimInt ivec2($n,$t)
                    
                    void main () {
                        vec2 uv = vUv;
                        
                        float x = uv.x * destDim.x;
                        float y = uv.y * destDim.y;
                        float sum = 0.0;
                        
                        for (int i = 0; i < leftDimInt.x; i++) {
                            float left = texture2D(leftMatrix, vec2(float(i) / leftDim.x, y / leftDim.y)).x;
                            float right = texture2D(rightMatrix, vec2(x / rightDim.x, float(i) / rightDim.y)).x;
                            
                            sum += left * right;
                        }
                        
                        gl_FragColor = vec4(sum);
                    }
                """.trimIndent()

                uniforms = object : MMUniforms {
                    init {
                        _leftMatrix = regl.prop("leftMatrix")
                        _rightMatrix = regl.prop("rightMatrix")
                    }
                }

                attributes = object : MMAttributes {
                    init {
                        _position = arrayOf(-4, -4, 4, -4, 0, 4)
                    }
                }

                this.count = 3
            }
        }

        val exec = regl(gpu)

        val destTexture = createTexture(n, m)
        val fbo = regl.framebuffer(object : FramebufferOptions {
            init {
                color = destTexture
            }
        })

        val matmul = fun (left: Texture2D, right: Texture2D): FloatArray {
            fbo(object : FramebufferOptions {
                init {
                    color = destTexture
                }
            })

            val result = FloatArray(n * m)
            fbo.use<DefaultContext, Any> { _, _, _ ->
                exec(object : MMProps {
                    init {
                        _leftMatrix = left
                        _rightMatrix = right
                    }
                })

                val array = regl.read<FloatArray>()

                var count = 0
                for (i in array.indices step 4) {
                    result[count++] = array[i]
                }
            }

            return result
        }

        val leftTexture = createTextureFromArray(n, t, left)
        val rightTexture = createTextureFromArray(t, m, right)
        var destBaselineGPU = FloatArray(0)

        val baselineGPUTime = measureTime {
            repeat(count) {
                destBaselineGPU = matmul(leftTexture, rightTexture)
            }
        }

        println("Baseline GPU: ${baselineGPUTime / count}")

        assertTrue(destBaseline.contentEquals(destBaselineGPU))
    }

    interface MMUniforms {
        var _leftMatrix: DynamicVariable<Any>
            get() = this.asDynamic()["leftMatrix"]
            set(v) { this.asDynamic()["leftMatrix"] = v}

        var _rightMatrix: DynamicVariable<Any>
            get() = this.asDynamic()["rightMatrix"]
            set(v) { this.asDynamic()["rightMatrix"] = v}
    }

    interface MMAttributes {
        var _position: Array<Number>
            get() = this.asDynamic()["position"]
            set(v) { this.asDynamic()["position"] = v}
    }

    interface MMProps {
        var _leftMatrix: Texture2D
            get() = this.asDynamic()["leftMatrix"]
            set(v) { this.asDynamic()["leftMatrix"] = v}

        var _rightMatrix: Texture2D
            get() = this.asDynamic()["rightMatrix"]
            set(v) { this.asDynamic()["rightMatrix"] = v}
    }

    private fun createTexture(width: Int, height: Int): Texture2D {
        return regl.texture(object : Texture2DOptions {
            init {
                this.width = width
                this.height = height
                format = "rgba"
                type = "float"
                mag = "nearest"
                min = "nearest"
            }
        })
    }

    private fun createTextureFromArray(width: Int, height: Int, array: FloatArray): Texture2D {
        val data = FloatArray(width * height * 4) { 0f }

        repeat(width * height) {
            data[it * 4] = array[it]
            data[it * 4 + 1] = array[it]
            data[it * 4 + 2] = array[it]
            data[it * 4 + 3] = array[it]
        }

        return regl.texture(object : Texture2DOptions {
            init {
                this.width = width
                this.height = height
                this.data = data
                format = "rgba"
                type = "float"
                mag = "nearest"
                min = "nearest"
            }
        })
    }

    private fun dotBaseline(left: FloatArray, right: FloatArray, dest: FloatArray, m: Int, n: Int, t: Int) {
        for (i in 0 until n) {
            val dInd = i * m
            val lInd = i * t
            for (k in 0 until t) {
                val temp = left[lInd + k]
                val rInd = k * m
                for (j in 0 until m) {
                    dest[dInd + j] += temp * right[rInd + j]
                }
            }
        }
    }
}

