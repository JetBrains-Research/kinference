import io.kinference.gradle.*

group = rootProject.group
version = rootProject.version

kotlin {
    js(IR) {
        browser()
    }

    jvm()

    sourceSets {
        commonMain {
            dependencies {
                api(project(":inference:inference-api"))
                api(project(":utils:utils-logger"))
                api(project(":serialization:serializer-protobuf"))
                api(project(":utils:utils-profiling"))
            }
        }
    }
}
