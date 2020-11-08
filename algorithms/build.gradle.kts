import io.kinference.gradle.useHeavyTests

group = rootProject.group
version = rootProject.version

useHeavyTests()

dependencies {
    api(project(":inference"))

    implementation("com.fasterxml.jackson.core", "jackson-databind", "2.11.3")

    testImplementation("org.junit.jupiter", "junit-jupiter", "5.6.2")
}
