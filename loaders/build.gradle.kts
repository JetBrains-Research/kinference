group = rootProject.group
version = rootProject.version

dependencies {
    api(kotlin("stdlib"))

    api(project(":ndarray"))

    api("com.amazonaws", "aws-java-sdk-s3", "1.11.896")
}
