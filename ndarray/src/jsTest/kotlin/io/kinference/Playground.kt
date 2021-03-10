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

        val count = 3

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
                // language=glsl
                vert = """
                    precision highp float;
                    attribute vec2 position;
    
                    void main () {
                        gl_Position = vec4(position, 0, 1);
                    }
                """.trimIndent()

                // language=glsl
                frag = """
                    precision highp float;
                    
                    uniform sampler2D leftMatrix;
                    uniform sampler2D rightMatrix;
                    
                    #define N ($n.0)
                    #define M ($m.0)
                    #define T ($t.0)
                    
                    void main () {                         
                        const int isize = int(T) / 4;
                        const float size = float(isize);
                        const float step = 1.0 / size;
                                               
                        float x = (gl_FragCoord.x - 0.5) * 4.;
                        float y = (gl_FragCoord.y - 0.5);
                        
                        float ind = 0.0;
                        float lind = y / N;
                        float rstep = 1.0 / M;
                        
                        float x1 = x / M;
                        float x2 = x1 + rstep;
                        float x3 = x2 + rstep;
                        float x4 = x3 + rstep;
                        
                        vec4 acc = vec4(0.);
                        for (int i = 0; i < isize; i++) {
                            vec4 left = texture2D(leftMatrix, vec2(ind, lind));
                            
                            acc.x += dot(left, texture2D(rightMatrix, vec2(ind, x1)));
                            acc.y += dot(left, texture2D(rightMatrix, vec2(ind, x2)));
                            acc.z += dot(left, texture2D(rightMatrix, vec2(ind, x3)));
                            acc.w += dot(left, texture2D(rightMatrix, vec2(ind, x4)));
                            
                            ind += step;
                        }
                        
                        gl_FragColor = acc;
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
                        _position = arrayOf(-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1)
                    }
                }

                this.count = 6
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

            var result: FloatArray? = null
            fbo.use<DefaultContext, Any> { _, _, _ ->
                exec(object : MMProps {
                    init {
                        _leftMatrix = left
                        _rightMatrix = right
                    }
                })

                result = regl.read<FloatArray>()
            }

            return result!!
        }

        val leftTexture = createTextureFromArray(n, t, left)
        val rightTexture = createTextureFromArray(t, m, right, true)
        var destBaselineGPU = FloatArray(0)

        val baselineGPUTime = measureTime {
            repeat(count) {
                destBaselineGPU = matmul(leftTexture, rightTexture)
            }
        }

        println("Baseline GPU: ${baselineGPUTime / count}")

//        println(destBaseline.toString())
//        println(destBaselineGPU.toString())
//        assertTrue(destBaseline.contentEquals(destBaselineGPU))
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
        require(width % 4 == 0) { "Width must be multiple of 4" }

        return regl.texture(object : Texture2DOptions {
            init {
                this.width = width / 4
                this.height = height
                format = "rgba"
                type = "float"
                mag = "nearest"
                min = "nearest"
            }
        })
    }

    private fun createTextureFromArray(width: Int, height: Int, array: FloatArray, transpose: Boolean = false): Texture2D {
        require(width % 4 == 0) { "Width must be multiple of 4" }

        val data = if (transpose) {
            FloatArray(width * height) { array[(it % height) * width + it / height] }
        } else {
            array
        }

        val w = if (transpose) height else width
        val h = if (transpose) width else height

        return regl.texture(object : Texture2DOptions {
            init {
                this.width = w / 4
                this.height = h
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

