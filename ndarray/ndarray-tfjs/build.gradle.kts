import io.kinference.gradle.configureTests

group = rootProject.group
version = rootProject.version

kotlin {
    js(IR) {
        browser()
        configureTests()
    }

    sourceSets {
        val jsMain by getting {
            dependencies {
                implementation(npm("@tensorflow/tfjs-core", libs.versions.tfjs.get()))
                implementation(npm("@tensorflow/tfjs-backend-webgl", libs.versions.tfjs.get()))

                api(project(":ndarray:ndarray-api"))
                api(project(":utils:utils-common"))
                implementation(libs.kotlinx.coroutines.core)
            }
        }

        val jsTest by getting {
            dependencies {
                implementation(project(":utils:utils-testing"))
            }
        }
    }
}
