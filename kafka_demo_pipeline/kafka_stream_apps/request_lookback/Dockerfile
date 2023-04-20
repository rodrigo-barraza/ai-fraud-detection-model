# FROM openjdk:8-jre-alpine
FROM maven:3.5.3-jdk-8-alpine

# Build related environment variables.
ENV SERVICE_NAME einstein-request-lookback

# Create the target folder
WORKDIR /opt/$SERVICE_NAME/
# Set the time zone
ENV TZ="America/Vancouver"
RUN echo "America/Vancouver" > /etc/timezone && \
        apk update && \
        apk add tzdata && \
        cp /usr/share/zoneinfo/$TZ /etc/localtime && \
        echo $TZ > /etc/timezone && \
        apk del tzdata



# Copy artifacts
COPY ./ /opt/$SERVICE_NAME/

# build the jar inside the docker image
RUN cd /opt/$SERVICE_NAME/ && mvn clean && mvn package

ENTRYPOINT java -jar /opt/$SERVICE_NAME/target/einstein-0.1-jar-with-dependencies.jar
