
package einstein;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.common.utils.Bytes;
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.Topology;
import org.apache.kafka.streams.kstream.Materialized;
import org.apache.kafka.streams.kstream.Produced;
import org.apache.kafka.streams.kstream.KStream;
import org.apache.kafka.streams.kstream.Joined;
import org.apache.kafka.streams.kstream.JoinWindows;
import org.apache.kafka.streams.Consumed;
import org.apache.kafka.streams.state.KeyValueStore;

import java.util.Arrays;
import java.util.Locale;
import java.util.Properties;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;


public class SlidingWindowAggregation {

    public static String joinRequestEvent(String request, String event){

        ObjectMapper mapper = new ObjectMapper();

        ObjectNode newNode = mapper.createObjectNode();

        newNode.put("request", request);
        newNode.put("event", event);

        String outputString = newNode.toString();
        System.out.println(outputString);

        return outputString;

    }

    public static void main(String[] args) throws Exception {
        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, "streams-sliding-window-aggregation");
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "kafka:29092");
        props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());

        final StreamsBuilder builder = new StreamsBuilder();


        KStream<String, String> left = builder.stream(
                "requests", /* input topic */
                Consumed.with(
                        Serdes.String(), /* key serde */
                        Serdes.String()   /* value serde */
                ));

        KStream<String, String> right = builder.stream(
                "events", /* input topic */
                Consumed.with(
                        Serdes.String(), /* key serde */
                        Serdes.String()   /* value serde */
                ));

        // Java 8+ example, using lambda expressions
        KStream<String, String> joined = left.leftJoin(right,
                (leftValue, rightValue) -> (joinRequestEvent(leftValue, rightValue)), /* ValueJoiner */
                JoinWindows.of(TimeUnit.MINUTES.toMillis(5))
                        .after(0)
                        .before(TimeUnit.MINUTES.toMillis(5))
                        .until(TimeUnit.DAYS.toMillis(1)),
                Joined.with(
                        Serdes.String(), /* key */
                        Serdes.String(),   /* left value */
                        Serdes.String())  /* right value */
        );

        joined.to("request-events", Produced.with(Serdes.String(), Serdes.String()));

           //-----------------------------------------------------------------

        final Topology topology = builder.build();
        final KafkaStreams streams = new KafkaStreams(topology, props);
        final CountDownLatch latch = new CountDownLatch(1);

        // attach shutdown handler to catch control-c
        Runtime.getRuntime().addShutdownHook(new Thread("streams-shutdown-hook") {
            @Override
            public void run() {
                streams.close();
                latch.countDown();
            }
        });

        try {
            streams.start();
            latch.await();
        } catch (Throwable e) {
            System.exit(1);
        }
        System.exit(0);
    }
}