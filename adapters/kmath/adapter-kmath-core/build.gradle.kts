import io.kinference.gradle.configureTests

group = rootProject.group
version = rootProject.version

repositories {
    maven("https://repo.kotlin.link")
}

kotlin {
    jvm {
        configureTests()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-core"))

                api(project(":inference:inference-api"))
                api(project(":inference:inference-core"))

                api(libs.kmath.core)
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(project(":utils:utils-testing"))
            }
        }
    }
}
