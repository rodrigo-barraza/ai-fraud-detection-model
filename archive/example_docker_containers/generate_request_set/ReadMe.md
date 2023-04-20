# Create Request Set

Prototype Docker container that takes a single request id, collects the events for the request's user for the hour prior to the request and creates a new record in the `requestEvents60` table in mongo.

Usage:

To trigger a job send a post request like so:

```{
    "request_id": <id>
}```

Building:

In the current directory, run `docker build -t <container_name>:latest .`

Running:

`docker run -p <host_port>:<container_port> <container_name>:latest`

