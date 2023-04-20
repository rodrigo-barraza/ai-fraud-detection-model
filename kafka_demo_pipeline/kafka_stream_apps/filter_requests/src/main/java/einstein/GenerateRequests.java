
package einstein;

import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.Topology;
import org.apache.kafka.streams.kstream.Produced;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.CountDownLatch;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;

public class GenerateRequests {

    public static boolean isRequest(String eventString) {

        ObjectMapper mapper = new ObjectMapper();
        boolean isRequest = false;

        try {
            JsonNode event = mapper.readTree(eventString);
            String eventCategory = event.get("eventCategory").toString();
            String eventAction = event.get("eventAction").toString();
            System.out.println(eventCategory);
            
            // needed to escape quotes because result in eventCategory is a string including quotes
            if ((eventCategory.equals("\"buy\"") || eventCategory.equals("\"interac\"")) && eventAction.equals("\"request\"")) {
                System.out.println("request found:");
                System.out.println(eventCategory);
                isRequest = true;
            }
        } catch (IOException e) {
            System.out.println("Error parsing event");
        }

        return isRequest;
    }

    public static void main(String[] args) throws Exception {
        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, "streams-generate-requests");
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "kafka:29092");
        props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());

        final StreamsBuilder builder = new StreamsBuilder();

        builder.<String, String>stream("events")
               .filter((String key, String value) -> (isRequest(value)))
               .to("requests", Produced.with(Serdes.String(), Serdes.String()));

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
